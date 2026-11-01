"""Independent recorded-evidence audit and native capacity gate; no policy execution."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch_geometric.data import Data

from gnn_pruning.models import build_model
from .flickr_dense import digest, sha, tensors_hash, write
from .v5_campaign import matrix, validation


def read(path):
    return json.loads(path.read_text())


def project(dense, kept, features, classes):
    """Slice original frozen tensors directly, independently of surgery code."""
    axes = [list(range(features)), *kept, list(range(classes))]; result = {}
    for name, value in dense.items():
        i = int(name.split('.')[1])
        selected = value.index_select(0, torch.tensor(axes[i+1], dtype=torch.long))
        if value.ndim == 2:
            selected = selected.index_select(1, torch.tensor(axes[i], dtype=torch.long))
        result[name] = selected.contiguous()
    return result


class Oracle:
    """Exhaustive integer two-hidden-layer geometry, frozen projection and rewards."""
    def __init__(self, cp, target):
        self.cp = cp; self.cfg = cp['model_config']; self.target = target
        self.factor = 2 if cp['model_name'] == 'graphsage' else 1
        self.features = self.cfg['in_channels']; self.classes = self.cfg['out_channels']
        self.p0 = self.count((32,32)); self.horizon = 56
        distance = {w:abs(Fraction(self.p0-self.count(w),self.p0)-Fraction(str(target)))
                    for a in range(4,33) for b in range(4,33) if (w:=(a,b)) != (32,32)}
        self.goals = {w for w,d in distance.items() if d <= Fraction(1,100)}
        self.fallback = not bool(self.goals)
        if self.fallback:
            self.goals = {w for w,d in distance.items() if d == min(distance.values())}

    def count(self, widths):
        a,b = widths
        return self.factor*(self.features*a+a*b+b*self.classes)+a+b+self.classes

    def goal(self, widths):
        return tuple(widths) in self.goals

    @lru_cache(None)
    def legal(self, widths, step, terminal=False):
        if terminal:return []
        result=[]
        for i in range(2):
            for k in (1,2,4,8):
                w=list(widths);w[i]-=k
                if w[i] < 4:continue
                assert self.horizon-step-1 >= sum(x-4 for x in w)
                if any(a <= w[0] and b <= w[1] for a,b in self.goals):result.append(f'prune:{i}:{k}')
        return result

    def state(self, ids, steps, accuracy, terminal=False):
        widths=list(map(len,ids))
        return ['structural_budget_v5',str(self.target),'0.01',widths,self.p0-self.count(widths),
                self.horizon-steps,math.floor(accuracy/.01),ids,terminal]

    @lru_cache(maxsize=4096)
    def tensor_hash(self, ids):
        return tensors_hash(project(self.cp['model_state_dict'],ids,self.features,self.classes))

    def transition(self, ids, step, dense_acc, row):
        action=row['action']; assert action in self.legal(tuple(map(len,ids)),step)
        _, layer, size=action.split(':');layer=int(layer);size=int(size)
        assert row['widths_before']==list(map(len,ids))
        new=row['original_kept_ids']
        assert all(new[i]==ids[i] for i in range(2) if i!=layer)
        projected=project(self.cp['model_state_dict'],ids,self.features,self.classes)
        scores=torch.zeros(len(ids[layer]),dtype=torch.float64)
        for name,value in projected.items():
            index=int(name.split('.')[1]);value=value.abs()
            if index==layer:scores+=(value.sum(1) if value.ndim==2 else value).double()
            elif index==layer+1 and value.ndim==2:scores+=value.sum(0).double()
        ranked=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
        keep=sorted(ranked[:-size]);assert new[layer]==[ids[layer][i] for i in keep]
        widths=list(map(len,new));terminal=self.goal(widths)
        assert row['widths_after']==widths and row['successful_deletions']==step+1
        sparse=1-self.count(widths)/self.p0;old=1-self.count(tuple(map(len,ids)))/self.p0
        drop=max(0.,dense_acc-row['validation_accuracy'])
        penalty=-2*max(0.,abs(sparse-self.target)-.01) if terminal else 0.
        reward=.6*(sparse-old)-.4*drop+penalty
        assert row['sparsity']==sparse and row['accuracy_drop']==drop
        assert row['terminal_penalty']==penalty and row['reward']==reward
        assert row['terminal_reason']==(('closest_feasible' if self.fallback else 'target_band') if terminal else None)
        return new,terminal


def behavior(directory, cp, c, stats, technical):
    """Audit actual decisions and recorded alternatives; no counterfactual forwards."""
    o=Oracle(cp,c['target']);trace=read(directory/'deployment.json');training=read(directory/'training.json')
    decisions=read(directory/'decisions.json');events=read(directory/'evaluation_cache.json')
    if c['method'].startswith('static_'):
        assert not training and not decisions and not events
        return dict(training_steps=0,deployment_steps=len(trace),transitions_checked=0)
    assert events
    dense_acc=events[0]['validation_accuracy'];observed={tensors_hash(cp['model_state_dict']):dense_acc}
    table={};visited={};rng=random.Random(c['policy_seed']);transitions=0
    key=lambda s:json.dumps(s,separators=(',',':'))
    def checked(row,ids,step):
        nonlocal transitions
        new,terminal=o.transition(ids,step,dense_acc,row);transitions+=1
        identity=o.tensor_hash(tuple(map(tuple,new)));acc=row['validation_accuracy']
        assert identity not in observed or observed[identity]==acc
        observed[identity]=acc
        return new,terminal
    def selection(decision,ids,step,acc,explore):
        state=o.state(ids,step,acc);actions=o.legal(tuple(map(len,ids)),step)
        assert decision['state']==state and decision['legal_actions']==actions
        row=table.get(key(state),{});values=[row.get(a,0.) for a in actions]
        tied=[a for a,v in zip(actions,values) if v==max(values)]
        assert decision['q_values']==values and decision['known']==[a in row for a in actions]
        assert decision['tied_maxima']==tied
        exploration=explore and rng.random()<.2
        selected=rng.choice(actions) if exploration else tied[0]
        assert decision['exploration']==exploration and decision['selected_action']==selected
        visited[key(state)]=len(actions)
        return state,row
    if c['method']=='tabular_q':
        episodes=3 if technical else 30
        assert {r['episode'] for r in training}==set(range(episodes))
        for episode in range(episodes):
            ids=[list(range(32)) for _ in range(2)];acc=dense_acc
            rows=[r for r in training if r['episode']==episode];assert rows
            for step,row in enumerate(rows):
                state,values=selection(row,ids,step,acc,True)
                assert row['selected_action']==row['action']
                ids,terminal=checked(row,ids,step);acc=row['validation_accuracy']
                following=o.state(ids,step+1,acc,terminal)
                actions=o.legal(tuple(map(len,ids)),step+1,terminal)
                assert row['next_state']==following and row['next_legal_actions']==actions
                previous=values.get(row['action'],0.);future=table.get(key(following),{})
                boot=0. if terminal else max((future.get(a,0.) for a in actions),default=0.)
                updated=previous+.1*(row['reward']+.9*boot-previous)
                assert row['q_before']==previous and row['q_after']==updated
                table.setdefault(key(state),{})[row['action']]=updated
                assert terminal==(step==len(rows)-1)
        assert {key(r['state']):r['actions'] for r in read(directory/'q_table.json')}==table
    else:
        assert not training and not read(directory/'q_table.json')
    ids=[list(range(32)) for _ in range(2)];acc=dense_acc;unseen=ties=0
    assert len(trace)==len(decisions)
    for step,(row,decision) in enumerate(zip(trace,decisions)):
        state=o.state(ids,step,acc);actions=o.legal(tuple(map(len,ids)),step)
        assert decision['state']==state and decision['legal_actions']==actions
        if c['method']=='tabular_q':
            _,values=selection(decision,ids,step,acc,False)
            unseen+=row['action'] not in values;ties+=len(decision['tied_maxima'])>1
        elif c['method']=='sequential_random':
            assert decision['selected_action']==rng.choice(actions)
        else:
            first={r['action']:r['transition'] for r in decision['first_transitions']}
            assert list(first)==actions
            for transition in first.values():checked(transition,ids,step)
            paths=decision['candidate_paths'];expected_paths=[]
            for action in actions:
                child=first[action];child_ids=child['original_kept_ids'];terminal=o.goal(child['widths_after'])
                second=[] if terminal or c['method']=='adaptive_greedy' else o.legal(tuple(child['widths_after']),step+1)
                expected_paths += [[action,b] for b in second] if second else [[action]]
            assert [path['actions'] for path in paths]==expected_paths
            for path in paths:
                first_reward=first[path['actions'][0]]['reward']
                if len(path['actions'])==1:
                    assert path['leaf']==first[path['actions'][0]] and path['score']==first_reward
                    assert path['rewards']==[first_reward]
                else:
                    parent=first[path['actions'][0]]
                    checked(path['leaf'],parent['original_kept_ids'],step+1)
                    assert path['leaf']['action']==path['actions'][1]
                    assert path['rewards']==[first_reward,path['leaf']['reward']]
                    assert path['score']==first_reward+.9*path['leaf']['reward']
            best=max(paths,key=lambda x:x['score'])
            assert decision['selected_action']==best['actions'][0] and decision['best_planned_path']==best['actions']
            assert row==first[decision['selected_action']]
        assert decision['selected_action']==row['action']
        ids,terminal=checked(row,ids,step);acc=row['validation_accuracy']
        if terminal:assert step==len(trace)-1
    if not (technical and c['method']=='tabular_q'):assert terminal
    else:assert len(trace)<=2
    seen=set()
    for index,event in enumerate(events):
        identity=event['tensor_sha256']
        assert event['request']==index+1 and identity in observed
        assert event['validation_accuracy']==observed[identity] and event['reused']==(identity in seen)
        seen.add(identity)
    assert stats['validation_callback_requests']==len(events)
    assert stats['actual_validation_forwards']==len(seen)
    assert stats['exact_cache_hits']==len(events)-len(seen)
    if c['method']=='tabular_q':
        assert len(events)==1+len(training)+len(trace)
        assert stats['q_entries']==sum(map(len,table.values())) and stats['unique_states']==len(visited)
        assert stats['legal_state_action_slots']==sum(visited.values())
        assert stats['q_table_coverage']==stats['q_entries']/sum(visited.values())
        assert stats['deployment_unseen_choices']==unseen and stats['deployment_tied_maxima']==ties
    elif c['method'] in ('adaptive_greedy','lookahead_depth2'):
        assert len(events)==1+sum(d['candidate_evaluations'] for d in decisions)
    return dict(training_steps=len(training),deployment_steps=len(trace),transitions_checked=transitions)


def audit_one(root,p,c,group,forward=True):
    """Hashes, final frozen projection/reload/metrics plus independent trace oracle."""
    directory=root/group/c['id'];marker=read(directory/'completed.json')
    assert marker=={f.name:sha(f) for f in directory.iterdir() if f.is_file() and f.name!='completed.json'}
    r=read(directory/'report.json');pin=p['dense'][c['dense_id']]
    assert r['condition']==c and r['source_commit']==p['source_commit'] and r['protocol_sha256']==digest(p)
    assert r['test_labels_accessed'] is False and r['dense_checkpoint_sha256']==pin['checkpoint_sha256']
    assert r['split_sha256']==pin['split_sha256'] and r['threads'] in (1,4)
    cp_path=root/'dense'/c['dense_id']/'checkpoint.pt';assert sha(cp_path)==pin['checkpoint_sha256']
    cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    g=p['graphs'][pin['graph']];graph_path=root/'inputs'/g['file'];assert sha(graph_path)==g['sha256']
    tensors=torch.load(graph_path,map_location='cpu',weights_only=True)
    split=read(root/'inputs'/g['split_file']);assert split==cp['split'] and bool((tensors['y'][split['test']]==-1).all())
    assert tensors_hash(tensors)==g['tensor_sha256']==r['graph_tensor_sha256']
    final=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert sha(directory/'checkpoint.pt')==r['checkpoint_sha256'] and final['original_kept_ids']==r['retained_original_ids']
    ids=r['retained_original_ids']
    assert len(ids)==2 and all(x==sorted(set(x)) and 4<=len(x)<=32 and all(0<=i<32 for i in x) for x in ids)
    expected=project(cp['model_state_dict'],ids,pin['model_config']['in_channels'],pin['model_config']['out_channels'])
    assert expected.keys()==final['model_state_dict'].keys() and all(torch.equal(v,final['model_state_dict'][k]) for k,v in expected.items())
    model=build_model(c['model'],**final['model_config']).eval();model.load_state_dict(final['model_state_dict'],strict=True)
    assert tensors_hash(model.state_dict())==r['tensor_sha256']
    o=Oracle(cp,c['target']);after=sum(v.numel() for v in model.parameters());achieved=1-after/o.p0
    assert r['parameters_before']==o.p0 and r['parameters_after']==after==o.count(tuple(map(len,ids)))<o.p0
    assert r['achieved_compression']==achieved and r['final_widths']==list(map(len,ids))
    assert r['target_miss']==(abs(achieved-c['target'])>.01+1e-14) and r['fallback']==o.fallback
    technical=r['technical'];stats=read(directory/'statistics.json');assert stats==r['statistics']
    if not (technical and c['method']=='tabular_q'):
        assert o.goal(tuple(map(len,ids))) and stats['complete']
    assert r['scientific_complete']==(not technical and stats['complete'])
    assert len(r['inference']['full_graph_forward_seconds'])==20 and r['inference']['warmup']==5
    assert all(math.isfinite(r[k]) and r[k]>0 for k in ('search_seconds','search_cpu_seconds','peak_rss_bytes'))
    if forward:
        measured=validation(model,Data(**tensors),split)
        assert all(abs(measured[k]-r['validation'][k]) <= (5e-5 if k=='loss' else 1e-12) for k in measured)
    diagnostics=behavior(directory,cp,c,stats,technical)
    return dict(id=c['id'],status='PASS',**diagnostics,report=r)


def capacity(root,p):
    """Audit all168 native profiles before resource-only common thread selection."""
    rows=[c for c in matrix(p) if c['dense_seed']==101];reports={};comparison=[]
    for n in (1,4):
        native=read(root/f'dense-native-t{n}.json')
        assert native['status']=='PASS' and native['source_commit']==p['source_commit']
        assert native['verified']=={k:v['checkpoint_sha256'] for k,v in p['dense'].items()}
        for c in rows:
            reports[n,c['id']]=audit_one(root,p,c,f'preflight-t{n}')
    for c in rows:
        left=root/'preflight-t1'/c['id'];right=root/'preflight-t4'/c['id']
        exact=all(read(left/(f+'.json'))==read(right/(f+'.json')) for f in ('training','deployment','decisions','q_table','evaluation_cache'))
        r1=reports[1,c['id']]['report'];r4=reports[4,c['id']]['report']
        exact=exact and r1['tensor_sha256']==r4['tensor_sha256'] and r1['validation']==r4['validation']
        comparison.append(dict(id=c['id'],exact_scientific_equal=exact))
    cost={n:sum(r['report']['search_seconds'] for (t,_),r in reports.items() if t==n) for n in (1,4)}
    threads=min((1,4),key=lambda n:cost[n]) if all(r['exact_scientific_equal'] for r in comparison) else 1
    # Do not manufacture a fresh timestamp if any of these assertions fails.
    estimates={};envelopes={}
    for method in p['methods']:
        for target in p['targets']:
            chosen=[r['report'] for (n,_),r in reports.items() if n==threads and r['report']['condition']['method']==method and r['report']['condition']['target']==target]
            workers=[]
            for r in chosen:
                batch=read(root/(f'batch-preflight-t{threads}-'+r['condition']['dense_id']+'.json'))
                workers.append(next(x['worker_seconds'] for x in batch if x['id']==r['condition']['id']))
            search=max(r['search_seconds'] for r in chosen)
            overhead=max(w-r['search_seconds'] for w,r in zip(workers,chosen))
            multiplier=31/3 if method=='tabular_q' else 1
            safe=2*search*multiplier+2*overhead+10
            memory=math.ceil(max(r['peak_rss_bytes'] for r in chosen)*3/1024**2+2048)
            assert safe>0 and memory<=8192
            envelopes[method,str(target)]=dict(safe_worker_seconds=safe,memory_mb=memory,
                                             measured_max_search_seconds=search,measured_max_overhead_seconds=overhead)
    for c in matrix(p):estimates[c['id']]=envelopes[c['method'],str(c['target'])]
    return dict(status='PASS',source_commit=p['source_commit'],protocol_sha256=digest(p),checked_unix=time.time(),
                dense_verified=18,technical_verified=168,threads=threads,conditions=estimates,
                cross_thread_comparison=comparison,measured_search_seconds_by_threads=cost,
                forecast_formula='2*max_native_search*(31/3 for Q else1)+2*max_worker_overhead+10; cross-stratum/seed envelope',
                actual_allocated_slots=20,requested_slots=4,host='polus-c4-ib.bmc.hpc.cs.msu.ru')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--capacity',action='store_true');args=a.parse_args()
    p=read(args.root/'protocol.json')
    if args.capacity:write(args.output,capacity(args.root,p))
    else:
        rows=[audit_one(args.root,p,c,'runs') for c in matrix(p)]
        write(args.output,dict(status='PASS',expected=252,verified=len(rows),runs=rows))
