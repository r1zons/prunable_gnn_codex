"""Independent offline audit of completed thread profiles; never runs a policy.

Checks stored decisions/Bellman arithmetic and hashes. Optional forward verification
loads frozen tensors and validation labels only; no fitting/search/test evaluation.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from gnn_pruning.experiments.flickr_dense import load_inputs, sha, tensors_hash
from gnn_pruning.models import build_model

P0=522503
THREADS=(4,8,12,20)
TARGETS=(.3,.5)
ARCH='flickr-graphsage-l4-h256-s101'


def read(path):return json.loads(path.read_text())

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def count(widths):
    a,b,c=widths
    return 1001*a+(2*a+1)*b+(2*b+15)*c+7


def key(state):return json.dumps(state,separators=(',',':'))


@lru_cache(None)
def reachable(widths,target):
    """Independent exact integer oracle: enumerate two axes, solve third interval."""
    from fractions import Fraction
    lo=math.ceil((1-Fraction(str(target))-Fraction('0.01'))*P0)
    hi=math.floor((1-Fraction(str(target))+Fraction('0.01'))*P0)
    a=np.arange(4,widths[0]+1,dtype=np.int64)[:,None]
    b=np.arange(4,widths[1]+1,dtype=np.int64)[None,:]
    base=1001*a+(2*a+1)*b+7;coef=2*b+15
    cmin=np.maximum(4,-((-(lo-base))//coef))
    cmax=np.minimum(widths[2],(hi-base)//coef)
    return bool(np.any(cmin<=cmax))


def goal(widths,target):
    from fractions import Fraction
    return abs(Fraction(P0-count(widths),P0)-Fraction(str(target)))<=Fraction('0.01')


def legal(widths,remaining,target,terminal=False):
    if terminal:return []
    actions=[]
    for layer in range(3):
        for size in (1,2,4,8):
            w=list(widths);w[layer]-=size
            if w[layer]<4:continue
            assert remaining-1>=sum(x-4 for x in w)
            if reachable(tuple(w),target):actions.append(f'prune:{layer}:{size}')
    return actions


def expected_state(ids,steps,accuracy,target,terminal=False):
    widths=list(map(len,ids))
    return ['structural_budget_v5',str(target),'0.01',widths,P0-count(widths),756-steps,
            math.floor(accuracy/.01),ids,terminal]


def projected_state(dense,kept):
    axes=[list(range(500)),*kept,list(range(7))];result={}
    for name,value in dense.items():
        layer=int(name.split('.')[1]);out=torch.as_tensor(axes[layer+1],dtype=torch.long)
        item=value.index_select(0,out)
        if value.ndim==2:item=item.index_select(1,torch.as_tensor(axes[layer],dtype=torch.long))
        result[name]=item.contiguous()
    return result


def verify_trace(directory,target,dense_state):
    """Replay recorded arithmetic/RNG, not environment steps or model evaluations."""
    training=read(directory/'training.json');deployment=read(directory/'deployment.json')
    decisions=read(directory/'decisions.json');events=read(directory/'evaluation_cache.json')
    table={};rng=random.Random(101);dense_accuracy=events[0]['validation_accuracy'];visited={}
    assert len(events)==1+len(training)+len(deployment) and len(deployment)==len(decisions)==2
    @lru_cache(None)
    def projected_hash(ids):return tensors_hash(projected_state(dense_state,[list(x) for x in ids]))
    requests=[([list(range(256)) for _ in range(3)],dense_accuracy)]
    def check_transition(row,ids,steps,accuracy):
        parts=row['action'].split(':');assert parts[0]=='prune'
        layer,size=map(int,parts[1:]);new=row['original_kept_ids']
        assert row['widths_before']==list(map(len,ids))
        assert all(new[i]==ids[i] for i in range(3) if i!=layer)
        assert new[layer]==sorted(set(new[layer])) and set(new[layer])<=set(ids[layer])
        assert len(ids[layer])-len(new[layer])==size
        widths=list(map(len,new));terminal=goal(widths,target)
        assert row['widths_after']==widths and row['successful_deletions']==steps+1
        sparse=1-count(widths)/P0;old_s=1-count(list(map(len,ids)))/P0
        drop=max(0.,dense_accuracy-row['validation_accuracy'])
        penalty=-2.*max(0.,abs(sparse-target)-.01) if terminal else 0.
        reward=.6*(sparse-old_s)-.4*drop+penalty
        assert row['sparsity']==sparse and row['accuracy_drop']==drop
        assert row['terminal_penalty']==penalty and row['reward']==reward
        assert row['terminal_reason']==('target_band' if terminal else None)
        requests.append((new,row['validation_accuracy']))
        return new,terminal
    for episode in range(3):
        entries=[r for r in training if r['episode']==episode];assert entries
        ids=[list(range(256)) for _ in range(3)];acc=dense_accuracy
        for step,row in enumerate(entries):
            state=expected_state(ids,step,acc,target);assert row['state']==state
            actions=legal(list(map(len,ids)),756-step,target);assert row['legal_actions']==actions
            values=table.get(key(state),{});q=[values.get(a,0.) for a in actions]
            assert row['q_values']==q and row['known']==[a in values for a in actions]
            tied=[i for i,v in enumerate(q) if v==max(q)];explore=rng.random()<.2
            index=rng.randrange(len(actions)) if explore else tied[0]
            assert row['exploration']==explore and row['selected_action']==actions[index]==row['action']
            assert row['tied_maxima']==[actions[i] for i in tied]
            visited[key(state)]=len(actions)
            ids,terminal=check_transition(row,ids,step,acc);acc=row['validation_accuracy']
            next_state=expected_state(ids,step+1,acc,target,terminal);assert row['next_state']==next_state
            next_actions=legal(list(map(len,ids)),755-step,target,terminal);assert row['next_legal_actions']==next_actions
            previous=values.get(row['action'],0.);future=table.get(key(next_state),{})
            bootstrap=0. if terminal else max((future.get(a,0.) for a in next_actions),default=0.)
            updated=previous+.1*(row['reward']+.9*bootstrap-previous)
            assert row['q_before']==previous and row['q_after']==updated
            table.setdefault(key(state),{})[row['action']]=updated
            assert terminal==(step==len(entries)-1)
        assert terminal
    unseen=ties=0;ids=[list(range(256)) for _ in range(3)];acc=dense_accuracy
    for step,(decision,row) in enumerate(zip(decisions,deployment)):
        state=expected_state(ids,step,acc,target);assert decision['state']==state
        actions=legal(list(map(len,ids)),756-step,target);values=table.get(key(state),{})
        q=[values.get(a,0.) for a in actions];tied=[i for i,v in enumerate(q) if v==max(q)]
        assert decision['legal_actions']==actions and decision['q_values']==q
        assert decision['known']==[a in values for a in actions] and not decision['exploration']
        assert decision['tied_maxima']==[actions[i] for i in tied]
        assert decision['selected_action']==row['action']==actions[tied[0]]
        unseen+=row['action'] not in values;ties+=len(tied)>1;visited[key(state)]=len(actions)
        ids,terminal=check_transition(row,ids,step,acc);acc=row['validation_accuracy']
        assert not terminal
    saved={key(r['state']):r['actions'] for r in read(directory/'q_table.json')};assert table==saved
    seen=set()
    for i,((ids,accuracy),event) in enumerate(zip(requests,events)):
        identity=projected_hash(tuple(map(tuple,ids)))
        assert event['request']==i+1 and event['tensor_sha256']==identity
        assert event['validation_accuracy']==accuracy and event['reused']==(identity in seen)
        seen.add(identity)
    for i,row in enumerate(training):assert row['validation_evaluation_reused']==events[i+1]['reused']
    for i,row in enumerate(decisions):assert row['validation_evaluation_reused']==events[1+len(training)+i]['reused']
    return dict(training_steps=len(training),deployment_steps=2,requests=len(events),forwards=len(seen),
        cache_hits=len(events)-len(seen),q_entries=sum(len(v) for v in table.values()),
        unique_states=len(visited),legal_state_action_slots=sum(visited.values()),deployment_unseen_choices=unseen,
        deployment_tied_maxima=ties,training_episodes=3,arithmetic_rng_state_masks_and_hashes='PASS')


def audit(raw,submission,historical,dense_collection,output):
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    protocol=read(raw/'protocol.json');config=read(raw/'config.json')
    assert sha(raw/'protocol.json')==config['scientific_protocol_sha256']
    assert config['benchmark_source_commit']=='10d778223f049935fd5d4ae554494da15377aaba'
    assert protocol['q_learning']==dict(episodes=30,epsilon=.2,alpha=.1,gamma=.9)
    assert protocol['technical_q_episodes']==3 and protocol['policy_seed']==101
    source=raw/('source-'+config['benchmark_source_commit']);source_files=read(raw/'source-files.json')
    assert source_files==read(submission/'source-files.json')
    preparation=read(Path(__file__).resolve().parent.parent/'PROVENANCE.json')
    assert sha(raw/'source-files.json')==preparation['source_file_inventory']['sha256']
    assert sha(raw/'source.tar.gz')==preparation['source_archive']['sha256']
    assert {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}==set(source_files)
    for name,expected in source_files.items():assert sha(source/name)==expected,name
    receipts=read(raw/'submission-receipts.json');assert [r['job_id'] for r in receipts]==[1612044,1612045,1612046]
    for r in receipts:
        claim=raw/'submissions'/r['job']['id']
        assert r['accepted'] and r['status']=='ACCEPTED' and r['source_commit']==config['benchmark_source_commit']
        assert sha(claim/'launcher.sh')==r['launcher_sha256'] and sha(raw/'config.json')==r['config_sha256']
        assert r['protocol_sha256']==sha(raw/'protocol.json')
    topology=[]
    for path in raw.glob('topology-*/topology.json'):
        d=read(path);cores={(r['socket'],r['core']) for r in d['cpu_topology']}
        assert len(cores)==d['physical_cores_visible']==20 and d['logical_cpus']==160
        assert len(d['cpu_topology'])==160
        sibling_counts=[sum((r['socket'],r['core'])==c for r in d['cpu_topology']) for c in cores]
        assert set(sibling_counts)=={8}
        topology.append(dict(host=d['host'],physical_cores=20,logical_cpus=160,smt=8,sockets=len({r['socket'] for r in d['cpu_topology']}),numa_nodes=len({r['numa'] for r in d['cpu_topology']}),job_id=d['lsf']['LSB_JOBID']))
    assert len(topology)==3
    placement=read(raw/'placement.json')['fixed20_physical_cpu_ids'];assert len(set(placement))==20
    data,split,manifest=load_inputs(dense_collection/'inputs')
    assert sha(dense_collection/'inputs/manifest.json')==protocol['input_manifest_sha256']
    assert (data.y[split['test']]==-1).all()
    dense_path=dense_collection/'runs'/ARCH/'checkpoint.pt';assert sha(dense_path)==protocol['dense'][ARCH]['checkpoint_sha256']
    dense=torch.load(dense_path,map_location='cpu',weights_only=True)['model_state_dict']
    assert tensors_hash(dense)==protocol['dense'][ARCH]['tensor_sha256'] and sum(x.numel() for x in dense.values())==P0
    statuses=read(raw/'worker_statuses.json');assert len(statuses)==8 and all(r['returncode']==0 and r['status']=='DONE' for r in statuses)
    rows=[];audits=[];base_reports={};traces={};loss_errors=[]
    for n in THREADS:
        for target in TARGETS:
            path=raw/'runs'/f't{n}-target{round(target*100)}';marker=read(path/'completed.json')
            assert {p.name for p in path.iterdir()}==set(marker)|{'completed.json'}
            for name,digest in marker.items():assert sha(path/name)==digest
            r=read(path/'report.json');env=read(path/'environment.json');timing=read(path/'q_timing.json');profile=read(path/'forward_profile.json')
            assert r['threads']==n and r['target']==target and r['scientific_episodes']==30 and r['technical_episodes']==3
            assert r['technical'] and not r['scientific_complete'] and not r['test_labels_accessed']
            assert r['source_commit']==config['benchmark_source_commit'] and r['scientific_source_commit']==protocol['source_commit']
            assert r['protocol_sha256']==sha(raw/'protocol.json')
            for k,expected in [('split_sha256',manifest['split_sha256']),('graph_tensor_sha256',manifest['training_tensor_sha256']),('dense_checkpoint_sha256',sha(dense_path)),('dense_tensor_sha256',tensors_hash(dense))]:assert r[k]==expected
            assert r['actual_host']==env['topology']['host']==config['host'] and env['topology']['lsf']['LSB_JOBID']=='1612046'
            assert int(env['topology']['lsf']['LSB_DJOB_NUMPROC'])==20
            assert r['affinity']==sorted(placement) and env['intraop']==n and env['interop']==r['interop']==1
            assert all(p['num_threads']<=n for p in r['threadpools_after'])
            assert all(t['cpus']==sorted(placement) for t in r['thread_affinity'])
            for label in ['dense','pruned']:
                f=profile[label];v=f['samples_seconds'];assert len(v)==f['repeats']==7
                assert f['warmup']==2 and f['actual_forward_calls']==9 and f['cache_used'] is False
                assert np.isclose(np.median(v),f['median_seconds'],rtol=0,atol=1e-15)
                assert np.isclose(np.std(v,ddof=1),f['stdev_seconds'],rtol=0,atol=1e-15)
            assert profile['dense']['widths']==[256]*3 and profile['dense']['parameters']==P0
            assert profile['pruned']['widths']==[111,230,219] and profile['pruned']['parameters']==266433
            assert profile['pruned']['tensor_sha256']==config['representative_pruned']['tensor_sha256']
            trace=verify_trace(path,target,dense);audits.append(dict(threads=n,target=target,**trace))
            assert trace['requests']==r['validation_evaluation_requests'] and trace['forwards']==r['actual_forward_calls'] and trace['cache_hits']==r['cache_hits']
            for k in ['q_entries','unique_states','legal_state_action_slots','deployment_unseen_choices','deployment_tied_maxima']:assert trace[k]==r['statistics'][k]
            assert r['statistics']['q_table_coverage']==trace['q_entries']/trace['legal_state_action_slots']
            assert r['statistics']['episodes']==3 and not r['statistics']['complete'] and r['statistics']['terminal_reason']=='technical_prefix'
            cp=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=build_model(cp['model_name'],**cp['model_config']).eval();model.load_state_dict(cp['model_state_dict'],strict=True)
            ids=r['retained_original_ids'];projected=projected_state(dense,ids)
            assert cp['original_kept_ids']==ids and model.hidden_channel_dims==r['final_widths']
            assert tensors_hash(projected)==tensors_hash(model.state_dict())==r['tensor_sha256'] and sha(path/'checkpoint.pt')==r['checkpoint_sha256']
            p1=sum(p.numel() for p in model.parameters());assert r['parameters_before']==P0 and p1==r['parameters_after']==count(r['final_widths'])
            assert r['achieved_compression']==1-p1/P0
            with torch.no_grad():
                logits=model(data)[split['val']];pred=logits.argmax(-1).numpy();y=data.y[split['val']].numpy()
                loss=float(torch.nn.functional.cross_entropy(logits,data.y[split['val']]))
            accuracy=accuracy_score(y,pred);f1=f1_score(y,pred,labels=list(range(7)),average='macro',zero_division=0)
            assert accuracy==r['validation']['accuracy'] and abs(f1-r['validation']['macro_f1'])<1e-12
            loss_errors.append(abs(loss-r['validation']['loss']));assert loss_errors[-1]<5e-5
            assert timing['search_seconds']==r['search_seconds'] and len(timing['episodes'])==3
            assert timing['actual_forward_seconds']<=timing['evaluation_seconds']<=r['search_seconds']
            assert sum(e['seconds'] for e in timing['episodes'])<r['search_seconds']
            assert timing['component_calls']['q_update']==trace['training_steps'] and timing['component_calls']['surgery']==trace['training_steps']+2
            safe=r['search_seconds']*(31/3)*1.75+180;assert safe==r['forecast']['safe_seconds']
            current={name:read(path/(name+'.json')) for name in ['training','deployment','decisions','q_table','evaluation_cache']}
            if n==4:base_reports[target]=r;traces[target]=current
            reference=base_reports[target]
            for k in ['retained_original_ids','final_widths','parameters_after','achieved_compression','validation_evaluation_requests','actual_forward_calls','cache_hits','tensor_sha256']:assert r[k]==reference[k],k
            assert r['validation']['accuracy']==reference['validation']['accuracy'] and r['validation']['macro_f1']==reference['validation']['macro_f1']
            assert current==traces[target]
            components=timing['observed_components_seconds']
            rows.append(dict(threads=n,target=target,host=r['actual_host'],job_id=1612046,
                search_seconds=r['search_seconds'],speedup_vs4=reference['search_seconds']/r['search_seconds'],
                forward_dense_median=profile['dense']['median_seconds'],forward_dense_sd=profile['dense']['stdev_seconds'],
                forward_pruned_median=profile['pruned']['median_seconds'],forward_pruned_sd=profile['pruned']['stdev_seconds'],
                cpu_utilization_percent=timing['mean_cpu_utilization_percent'],peak_rss_gib=r['peak_rss_bytes']/1024**3,
                evaluation_seconds=timing['evaluation_seconds'],actual_forward_seconds=timing['actual_forward_seconds'],
                cache_overhead_seconds=timing['cache_hash_lookup_overhead_seconds'],surgery_seconds=components['surgery'],
                magnitude_seconds=components['magnitude_scores']+components['channel_ranking'],
                legal_state_seconds=components['legal_action_planning']+components['state_encoding'],q_update_seconds=components['q_update'],
                residual_seconds=timing['residual_policy_trace_and_observer_seconds'],episode0_seconds=timing['episodes'][0]['seconds'],
                episode1_seconds=timing['episodes'][1]['seconds'],episode2_seconds=timing['episodes'][2]['seconds'],
                validation_requests=trace['requests'],actual_forwards=trace['forwards'],cache_hits=trace['cache_hits'],
                validation_accuracy=r['validation']['accuracy'],validation_macro_f1=r['validation']['macro_f1'],
                achieved_compression=r['achieved_compression'],final_widths='/'.join(map(str,r['final_widths'])),parameters_after=p1,
                q_entries=trace['q_entries'],q_coverage=r['statistics']['q_table_coverage'],unseen_choices=trace['deployment_unseen_choices'],
                terminal_reason=r['statistics']['terminal_reason'],forecast_minutes=safe/60,passes170=safe<=170*60,passes180=safe<=180*60,
                required_memory_mb=math.ceil((r['peak_rss_bytes']*3+2*1024**3)/1024**2),exact_equivalence_to4=True))
    old=historical/'preflight-cached/smoke'/f'{ARCH}-tabular_q-t50-budget-v5'
    # Caller passes the historical structural artifact group for this separate identity gate.
    for name in ['training','deployment','decisions','q_table','evaluation_cache']:assert read(old/(name+'.json'))==traces[.5][name]
    old_report=read(old/'report.json');assert old_report['tensor_sha256']==base_reports[.5]['tensor_sha256']
    choice=min((r for r in rows if r['target']==.5 and r['passes170']),key=lambda r:r['forecast_minutes'])
    assert choice['threads']==12
    for n in THREADS:
        worst=max(r['forecast_minutes'] for r in rows if r['threads']==n)
        assert worst==next(r for r in rows if r['threads']==n and r['target']==.5)['forecast_minutes']
    native=read(raw/'analysis.json');assert native['recommended_configuration']['threads']==12
    assert native['historical_q50_trace_equality'] and all(native['historical_q50_trace_equality'].values())
    assert len(native['scientific_equivalence'])==8 and all(r['scientific_equivalence'] for r in native['scientific_equivalence'])
    with (output/'runs.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=dict(PASS=8,FAIL=0,SKIPPED=0,source_files_verified=len(source_files),topology=topology,verified_training_steps=sum(a['training_steps'] for a in audits),
        verified_cache_requests=sum(a['requests'] for a in audits),all_exact_cross_thread_traces=True,historical4_50_trace_match=True,
        independently_recomputed_final_validations=8,local_validation_loss_max_difference=max(loss_errors),
        best_threads=12,best_worst_target_forecast_minutes=choice['forecast_minutes'],q_resource_gate='GO',
        full56_status='NOT_SUBMITTED_FRESH_COMPLETE_MANIFEST_GATE_STILL_REQUIRED',no_test_labels=True,no_new_search_or_training=True,
        traces=audits,runs=rows)
    write(output/'AUDIT.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['traces','runs','topology']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['raw','submission','historical','dense-collection','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();audit(a.raw,a.submission,a.historical,a.dense_collection,a.output)
