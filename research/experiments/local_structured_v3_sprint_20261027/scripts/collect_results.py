"""Offline V3 evidence audit and descriptive aggregation; never launches experiments.

Uses immutable completed attempts, independently replays trace arithmetic/Bellman
updates, reloads physical models and compares paired platforms without pooling.
"""
from __future__ import annotations
import argparse, collections, csv, functools, hashlib, json, math, statistics
from pathlib import Path
import torch
from gnn_pruning.models import build_model
from gnn_pruning.experiments.v3_sprint import load_frozen, algorithm_identity

SOURCE='7e2c760e988b605e305615dea147236e8b0c59a3'
REACHABLE={'q_reachable_v3','random_reachable_v3','greedy_reachable_v3'}
ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def key(state):return (state[0],tuple(state[1]),*state[2:])
def normalize_q(rows):return {key(row['state']):row['actions'] for row in rows}
def count(kind,widths):
    a,b=widths
    return (2 if kind=='graphsage' else 1)*(1433*a+a*b+7*b)+a+b+7

@functools.lru_cache(None)
def goals(kind,target):
    p0=count(kind,(32,32));percent=round(target*100)
    return tuple((a,b) for a in range(4,33) for b in range(4,33)
                 if abs(100*(p0-count(kind,(a,b)))-percent*p0)<=p0)

COINS=[0]+[999]*28
for i in range(1,29):COINS[i]=1+min(COINS[i-k] for k in (1,2,4,8) if k<=i)
@functools.lru_cache(None)
def reachable(kind,target,widths,remaining):
    a,b=widths
    return any(x<=a and y<=b and COINS[a-x]+COINS[b-y]<=remaining for x,y in goals(kind,target))

def legal(cfg,state):
    if state[-1]:return []
    widths,remaining=state[1:3];out=[]
    for layer in (0,1):
        for k in (1,2,4,8):
            future=list(widths);future[layer]-=k
            if future[layer]<4:continue
            if cfg['variant'] in REACHABLE and not reachable(cfg['model'],cfg['target'],tuple(future),remaining-1):continue
            out.append('prune:'+str(layer)+':'+str(k))
    if remaining<20 and cfg['variant'] not in REACHABLE:out.append('stop')
    return out

def qvalue(table,state,action,cfg):
    if cfg['variant']=='q_stop_exact_v3' and action=='stop':
        compression=1-count(cfg['model'],state[1])/count(cfg['model'],(32,32))
        return -2*max(0,abs(compression-cfg['target'])-.01)
    if action in table.get(state,{}):return table[state][action]
    if cfg['variant']=='q_conservative_v3':
        return -.4*sum(.9**j for j in range(state[2]))-2*max(0,max(cfg['target'],1-cfg['target'])-.01)
    return 0.

def audit_trace(cfg,report,trace,qrows):
    """Independent equations, exact width oracle, all episodes and serialized Q updates."""
    variant=cfg['variant'];p0=count(cfg['model'],(32,32))
    dense=float(torch.tensor(report['dense_validation']['accuracy'],dtype=torch.float32))
    episodes=[];table={};transitions=0
    def sequence(events,learning=False):
        nonlocal transitions
        widths=(32,32);ids=[list(range(32)),list(range(32))];steps=0;accuracy=dense;previous_s=0.
        rewards=0.;terminal=None
        for event in events:
            assert terminal is None
            state=('local_v2',widths,20-steps,math.floor(accuracy/.01),False)
            action=event['action'];assert action in legal(cfg,state),(cfg['run_id'],state,action)
            assert tuple(event['widths_before'])==widths
            new_ids=event['original_kept_ids'];after=tuple(event['widths_after'])
            if action=='stop':
                assert steps>0 and after==widths and new_ids==ids
                reason='agent_stop';base_reward=0.
            else:
                _,layer,k=action.split(':');layer,k=int(layer),int(k);expected=list(widths);expected[layer]-=k
                assert after==tuple(expected)
                for i in (0,1):
                    assert new_ids[i]==sorted(set(new_ids[i])) and len(new_ids[i])==after[i]
                    assert set(new_ids[i])<=set(ids[i])
                    if i!=layer:assert new_ids[i]==ids[i]
                steps+=1
                compression=1-count(cfg['model'],after)/p0
                base_reward=.6*(compression-previous_s)-.4*max(0,dense-event['validation_accuracy'])
                reason=('max_steps' if steps>=20 else 'no_valid_actions' if after==(4,4) else None)
                if reason is None and variant=='q_target_stop_v3' and compression>=cfg['target']-.01:reason='target_threshold'
                if reason is None and variant in REACHABLE and after in goals(cfg['model'],cfg['target']):reason='target_band'
            compression=1-count(cfg['model'],after)/p0
            close(event['sparsity'],compression);close(event['signed_target_deviation'],compression-cfg['target'])
            close(event['accuracy_drop'],max(0,dense-event['validation_accuracy']))
            assert event['successful_deletions']==steps and event['terminal_reason']==reason
            penalty=-2*max(0,abs(compression-cfg['target'])-.01) if reason else 0.
            close(event['terminal_penalty'],penalty);close(event['reward'],base_reward+penalty)
            if variant in REACHABLE:assert reachable(cfg['model'],cfg['target'],after,20-steps)
            next_state=('local_v2',after,20-steps,math.floor(event['validation_accuracy']/.01),bool(reason))
            if learning:
                future=0. if reason else max(qvalue(table,next_state,a,cfg) for a in legal(cfg,next_state))
                previous=qvalue(table,state,action,cfg)
                updated=(event['reward'] if variant=='q_stop_exact_v3' and action=='stop'
                         else previous+.1*(event['reward']+.9*future-previous))
                table.setdefault(state,{})[action]=updated
            widths,ids,accuracy,previous_s,terminal=after,new_ids,event['validation_accuracy'],compression,reason
            rewards+=event['reward'];transitions+=1
        assert terminal is not None
        return dict(steps=steps,return_value=rewards,terminal_reason=terminal,achieved_sparsity=previous_s,
                    validation_accuracy=accuracy,target_hit=widths in goals(cfg['model'],cfg['target']),ids=ids)
    if variant=='fixed_probe_v3':
        assert [(e['layer'],e['k']) for e in trace['deployment']]==[(0,1),(1,2),(0,4)]
        assert trace['deployment'][-1]['original_kept_ids']==[list(range(27)),list(range(30))]
        assert not trace['training'] and not qrows
        return {'transitions':3,'episodes':0,'deployment_steps':3},[]
    training=trace['training'];grouped=collections.defaultdict(list);resets=[]
    for event in training:
        if event.get('event')=='reset':
            assert event['widths']==[32,32];resets.append(event['episode'])
        else:grouped[event['episode']].append(event)
    if variant in REACHABLE-{'q_reachable_v3'}:
        assert not training and not qrows
    else:
        assert resets==list(range(100)) and set(grouped)==set(range(100))
        for episode in range(100):episodes.append(dict(episode=episode,**sequence(grouped[episode],True)))
    deployed=sequence(trace['deployment'])
    assert deployed['ids']==report['original_kept_ids']
    assert deployed['terminal_reason']==report['statistics']['terminal_reason']
    close(deployed['validation_accuracy'],float(torch.tensor(report['final_validation']['accuracy'],dtype=torch.float32)))
    serialized=normalize_q(qrows)
    assert sum(map(len,serialized.values()))==report['statistics'].get('q_entries',0)
    for state,row in table.items():
        for action,value in row.items():close(serialized[state][action],value)
    for state,row in serialized.items():
        for action,value in row.items():
            assert math.isfinite(value)
            if action not in table.get(state,{}):assert variant=='q_stop_exact_v3' and action=='stop'
            if variant=='q_stop_exact_v3' and action=='stop':close(value,qvalue({},state,action,cfg))
    if episodes:
        assert report['statistics']['validation_evaluations']==101+sum(e['steps'] for e in episodes)+deployed['steps']
    elif variant=='random_reachable_v3':assert report['statistics']['validation_evaluations']==1+deployed['steps']
    elif variant=='greedy_reachable_v3':assert report['statistics']['validation_evaluations']==1+report['statistics']['candidate_evaluations']
    return dict(transitions=transitions,episodes=len(episodes),deployment_steps=deployed['steps'],
                q_entries_verified=sum(map(len,serialized.values()))),episodes

def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)

def aggregate(rows):
    groups=collections.defaultdict(list)
    for row in rows:
        if row['variant']!='fixed_probe_v3':groups[(row['platform'],row['model'],row['target'],row['variant'])].append(row)
    result=[]
    for group,items in sorted(groups.items()):
        row=dict(zip(('platform','model','target','variant'),group));row.update(n=len(items),policy_seeds=','.join(str(x['policy_seed']) for x in sorted(items,key=lambda x:x['policy_seed'])),
                    target_hits=sum(x['target_hit'] for x in items),terminal_reasons=json.dumps(dict(collections.Counter(x['terminal_reason'] for x in items)),sort_keys=True))
        for metric in ('achieved_sparsity','target_gap','validation_accuracy','validation_macro_f1','validation_accuracy_drop','search_seconds','deployment_steps'):
            values=[x[metric] for x in items];row[metric+'_mean']=statistics.mean(values)
            row[metric+'_sample_std']=statistics.stdev(values) if len(values)>1 else None
            row[metric+'_min']=min(values);row[metric+'_max']=max(values)
        result.append(row)
    return result

def learning_blocks(episodes):
    groups=collections.defaultdict(list)
    for row in episodes:groups[row['run_id'],row['episode']//20].append(row)
    result=[]
    for (_,block),items in sorted(groups.items()):
        assert len(items)==20
        first=items[0];row={k:first[k] for k in ('run_id','platform','model','target','variant','policy_seed')}
        row.update(episode_start=block*20,episode_end=block*20+19,n=20,target_hits=sum(e['target_hit'] for e in items),
                   terminal_reasons=json.dumps(dict(collections.Counter(e['terminal_reason'] for e in items)),sort_keys=True))
        for metric in ('return_value','validation_accuracy','achieved_sparsity','steps'):
            row[metric+'_mean']=statistics.mean(e[metric] for e in items)
            row[metric+'_sample_std']=statistics.stdev(e[metric] for e in items)
        result.append(row)
    return result

def run(local,remote,collection,out):
    out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    configs=read(ROOT/'configuration_index.json');receipt=read(ROOT/'launch_receipt.json')
    inputs=read(local.parent/'inputs/manifest.json');rows=[];audits=[];episode_rows=[];records={};inventory=[];decisions=[]
    local_before={str(p):sha(p) for p in local.rglob('*') if p.is_file()}
    for p,digest in local_before.items():inventory.append({'platform':'local','path':str(Path(p).relative_to(local)),'sha256':digest,'bytes':Path(p).stat().st_size})
    assert sha(local/'manifest-all.json')==receipt['local']['manifest_sha256']
    for job in receipt['polus']['jobs']:
        assert sha(remote/('manifest-'+job['condition']+'.json'))==job['manifest_sha256']
        assert 'Successfully completed.' in (remote.parent/'logs/lsf'/('v3-cpu.'+str(job['job_id'])+'.out')).read_text()
    dense={kind:load_frozen(local.parent/'inputs/manifest.json',kind) for kind in ('gcn','graphsage')}
    for platform,base,expected in [('local',local,configs['local_configs']),('polus',remote,configs['polus_configs'])]:
        assert len(list((base/'outputs').glob('*/attempt-*/completed.json')))==len(expected)
        batches=list((base/'execution').glob('*/batch.finished.json'))
        assert sum(read(p)['completed_tasks'] for p in batches)==len(expected)
        assert all(read(p)['all_completed'] and read(p)['failed_tasks']==0 for p in batches)
        for declared in expected:
            runid=declared['run_id'];attempts=list((base/'outputs'/runid).glob('attempt-*'));assert len(attempts)==1
            attempt=attempts[0];done=read(attempt/'completed.json');cfg=read(attempt/'config.json');report=read(attempt/'metrics.json')
            for name,digest in done['hashes'].items():assert sha(attempt/name)==digest,(runid,name)
            assert done['status']=='completed' and done['source_commit']==SOURCE and report['run_id']==cfg['run_id']==runid
            assert canonical(cfg)==done['config_sha256']
            expected_cfg=dict(cfg)
            if platform=='polus':
                for name in ('input_manifest','output_dir'):
                    expected_cfg[name]='${POLUS_TASK_ROOT}'+cfg[name].split('/gnn-polus-20261023',1)[1]
            assert expected_cfg==declared
            assert done['algorithm_identity_sha256']==algorithm_identity(cfg,inputs)==report['provenance']['algorithm_identity_sha256']
            assert report['provenance']['label_access']=='validation_only' and cfg['training_seed']==101
            assert report['provenance']['graph_tensor_sha256']==inputs['graph_tensor_sha256']
            assert report['provenance']['split_sha256']==inputs['split']['sha256']
            assert report['provenance']['checkpoint_sha256']==inputs['models'][cfg['model']]['sha256']
            _,data,idx,original,_=dense[cfg['model']]
            assert (data.y[~torch.isin(torch.arange(2708),idx)]==-1).all()
            c=torch.load(attempt/'pruned.pt',weights_only=True,map_location='cpu');ids=c['original_kept_ids']
            assert c['provenance']==report['provenance'] and c['variant']==cfg['variant']
            assert ids==report['original_kept_ids'] and list(map(len,ids))==report['hidden_widths']
            model=build_model(c['model_name'],**c['model_config']).eval();model.load_state_dict(c['model_state_dict'],strict=True)
            parameters=count(cfg['model'],list(map(len,ids)));p0=count(cfg['model'],(32,32))
            assert sum(p.numel() for p in model.parameters())==parameters==report['parameters_after']<p0==report['parameters_before']
            for name,value in model.state_dict().items():
                layer=int(name.split('.')[1]);original_value=original['model_state_dict'][name]
                if layer<2:original_value=original_value[ids[layer]]
                if layer>0 and original_value.ndim==2:original_value=original_value[:,ids[layer-1]]
                assert torch.equal(value,original_value),(runid,name)
            with torch.no_grad():logits=model(data)
            assert logits.shape==(2708,7) and torch.isfinite(logits).all()
            observed_accuracy=float((logits.argmax(1)[idx]==data.y[idx]).double().mean())
            predictions=logits.argmax(1)[idx];labels=data.y[idx]
            confusion=torch.bincount(labels*7+predictions,minlength=49).reshape(7,7).double()
            f1=(2*confusion.diag()/(confusion.sum(0)+confusion.sum(1)).clamp_min(1)).mean().item()
            if platform=='local':
                close(observed_accuracy,report['final_validation']['accuracy']);close(f1,report['final_validation']['macro_f1'])
            hit=tuple(report['hidden_widths']) in goals(cfg['model'],cfg['target'])
            close(report['achieved_sparsity'],1-parameters/p0);assert hit==(not report['target_violation'])
            close(report['target_gap'],abs(report['achieved_sparsity']-cfg['target']))
            trace=read(attempt/'trace.json');q=read(attempt/'q_table.json');audit,episodes=audit_trace(cfg,report,trace,q)
            decision_start=len(decisions)
            if q:
                table=normalize_q(q)
                previous_accuracy=float(torch.tensor(report['dense_validation']['accuracy'],dtype=torch.float32));steps=0
                for i,event in enumerate(trace['deployment']):
                    state=('local_v2',tuple(event['widths_before']),20-steps,math.floor(previous_accuracy/.01),False)
                    actions=legal(cfg,state);values={a:qvalue(table,state,a,cfg) for a in actions}
                    best=max(values.values());ties=[a for a,v in values.items() if v==best]
                    assert event['action'] in ties
                    if cfg['variant']!='q_random_ties_v3':assert event['action']==ties[0]
                    decisions.append(dict(run_id=runid,platform=platform,variant=cfg['variant'],step=i,
                        action=event['action'],selected_q=values[event['action']],max_tie_count=len(ties),
                        selected_action_unseen=event['action'] not in table.get(state,{}),
                        state_unseen=state not in table,unseen_legal_actions=sum(a not in table.get(state,{}) for a in actions)))
                    previous_accuracy=event['validation_accuracy'];steps=event['successful_deletions']
            run_decisions=decisions[decision_start:]
            for e in episodes:
                e.pop('ids');episode_rows.append(dict(run_id=runid,platform=platform,model=cfg['model'],target=cfg['target'],variant=cfg['variant'],policy_seed=cfg['policy_seed'],**e))
            inference=report['inference'];stats=report['statistics']
            if inference:
                assert inference['warmup']==5 and inference['passes']==20 and inference['mean_seconds']>0
            row=dict(run_id=runid,platform=platform,model=cfg['model'],target=cfg['target'],variant=cfg['variant'],training_seed=101,policy_seed=cfg['policy_seed'],
                source_commit=SOURCE,algorithm_identity_sha256=done['algorithm_identity_sha256'],width0=len(ids[0]),width1=len(ids[1]),
                parameters_before=p0,parameters_after=parameters,achieved_sparsity=report['achieved_sparsity'],target_hit=hit,target_gap=report['target_gap'],
                signed_target_deviation=report['signed_target_deviation'],dense_validation_accuracy=report['dense_validation']['accuracy'],
                validation_accuracy=report['final_validation']['accuracy'],validation_macro_f1=report['final_validation']['macro_f1'],
                validation_accuracy_drop=report['dense_validation']['accuracy']-report['final_validation']['accuracy'],
                terminal_reason=stats['terminal_reason'],deployment_steps=audit['deployment_steps'],search_seconds=report['search_seconds'],
                total_seconds=report['total_seconds'],inference_mean_seconds=inference['mean_seconds'] if inference else None,
                inference_std_seconds=inference['std_seconds'] if inference else None,validation_evaluations=stats.get('validation_evaluations'),
                q_entries=stats.get('q_entries'),learned_q_entries=stats.get('learned_q_entries'),analytic_stop_entries=stats.get('analytic_stop_entries'),
                q_table_sparsity=stats.get('q_table_sparsity'),threads=report['runtime']['threads'],machine=report['runtime']['machine'],
                torch=report['runtime']['torch'],pyg=report['runtime']['pyg'],checkpoint_bytes=(attempt/'pruned.pt').stat().st_size,
                deployment_unseen_selected=sum(d['selected_action_unseen'] for d in run_decisions) if run_decisions else None,
                deployment_tied_maxima=sum(d['max_tie_count']>1 for d in run_decisions) if run_decisions else None)
            rows.append(row);audits.append(dict(run_id=runid,status='PASS',**audit,reload_local_validation_accuracy=observed_accuracy,
                          native_report_validation_accuracy=report['final_validation']['accuracy'],file_hashes_verified=len(done['hashes']),
                          checkpoint_sha256=sha(attempt/'pruned.pt'),completed_sha256=sha(attempt/'completed.json')))
            records[platform,done['algorithm_identity_sha256']]=(row,report,trace,q,c,attempt,logits)
    pairs=[];logit_comparisons=[]
    for (platform,identity),right in records.items():
        if platform!='polus':continue
        left=records['local',identity];l,lr,lt,lq,lc,lp,ll=left;r,rr,rt,rq,rc,rp,rl=right
        la=[e.get('action',(e.get('layer'),e.get('k'))) for e in lt['deployment']]
        ra=[e.get('action',(e.get('layer'),e.get('k'))) for e in rt['deployment']]
        same_weights=all(torch.equal(v,rc['model_state_dict'][k]) for k,v in lc['model_state_dict'].items()) if lr['hidden_widths']==rr['hidden_widths'] else False
        lqt,rqt=normalize_q(lq),normalize_q(rq)
        qkeys_equal=set(lqt)==set(rqt) and all(set(lqt[s])==set(rqt[s]) for s in lqt)
        qdelta=max((abs(value-rqt[s][action]) for s,row in lqt.items() for action,value in row.items()),default=0.) if qkeys_equal else None
        pairs.append(dict(algorithm_identity_sha256=identity,local_run_id=l['run_id'],polus_run_id=r['run_id'],model=l['model'],target=l['target'],variant=l['variant'],policy_seed=101,
            dense_validation_accuracy_delta=rr['dense_validation']['accuracy']-lr['dense_validation']['accuracy'],
            final_validation_accuracy_delta=r['validation_accuracy']-l['validation_accuracy'],achieved_sparsity_delta=r['achieved_sparsity']-l['achieved_sparsity'],
            final_validation_macro_f1_delta=r['validation_macro_f1']-l['validation_macro_f1'],
            same_actions=la==ra,same_retained_ids=lr['original_kept_ids']==rr['original_kept_ids'],same_checkpoint_tensors=same_weights,
            same_training_actions=[e.get('action') for e in lt['training']]==[e.get('action') for e in rt['training']],
            same_q_values=lqt==rqt,same_q_keys=qkeys_equal,max_abs_q_value_difference=qdelta,
            local_search_seconds=l['search_seconds'],polus_search_seconds=r['search_seconds'],
            polus_over_local_search_ratio=r['search_seconds']/l['search_seconds'],local_inference_seconds=l['inference_mean_seconds'],
            polus_inference_seconds=r['inference_mean_seconds'],polus_over_local_inference_ratio=r['inference_mean_seconds']/l['inference_mean_seconds']))
        if l['variant']=='fixed_probe_v3':
            for name in ('dense_logits.pt','fixed_pruned_logits.pt'):
                a=torch.load(lp/name,weights_only=True,map_location='cpu');b=torch.load(rp/name,weights_only=True,map_location='cpu')
                difference=(a-b).abs();idx=dense[l['model']][2]
                logit_comparisons.append(dict(model=l['model'],tensor=name,shape=list(a.shape),exact_equal=torch.equal(a,b),
                    max_abs_difference=float(difference.max()),mean_abs_difference=float(difference.mean()),
                    all_nodes_argmax_disagreements=int((a.argmax(1)!=b.argmax(1)).sum()),
                    validation_argmax_disagreements=int((a.argmax(1)[idx]!=b.argmax(1)[idx]).sum()),
                    allclose_rtol_1e5_atol_1e6=torch.allclose(a,b,rtol=1e-5,atol=1e-6)))
    assert len(rows)==132 and len(pairs)==34 and len(audits)==132
    for p,digest in local_before.items():assert sha(p)==digest,'Original local evidence mutated'
    assert len(local_before)==sum(p.is_file() for p in local.rglob('*'))
    csv_write(out/'runs.csv',rows);save(out/'runs.json',rows)
    summary=aggregate(rows);csv_write(out/'summary.csv',summary);save(out/'summary.json',summary)
    csv_write(out/'platform_pairs.csv',pairs);save(out/'platform_pairs.json',pairs)
    csv_write(out/'episodes.csv',episode_rows)
    csv_write(out/'learning_blocks.csv',learning_blocks(episode_rows))
    csv_write(out/'deployment_decisions.csv',decisions)
    save(out/'verification.json',dict(status='PASS',source_commit=SOURCE,verified_runs=132,local_runs=98,polus_runs=34,
        verified_training_episodes=sum(a['episodes'] for a in audits),verified_transitions=sum(a['transitions'] for a in audits),
        verified_q_entries=sum(a.get('q_entries_verified',0) for a in audits),original_local_files_unchanged=len(local_before),
        checks=['receipt/config/source/input identity','all recorded file hashes','strict physical reload','projection from frozen dense tensors',
                'exact parameter counts and target band','finite forwards','validation labels only','all training/deployment trace arithmetic',
                'legal actions and remaining-horizon reachability','once-only terminal penalty','independent Bellman replay and H1 STOP values'],runs=audits))
    save(out/'logit_comparison.json',logit_comparisons)
    save(collection/'local-inventory.json',inventory)
    save(out/'provenance.json',dict(source_commit=SOURCE,local_artifact_root=str(local),remote_snapshot_root=str(remote),
          input_manifest=inputs,local_inventory_sha256=sha(collection/'local-inventory.json'),
          polus_inventory_sha256=sha(remote.parent/'collection/inventory.json'),archive_sha256=sha(collection/'polus-results.tar.gz'),
          policy_seed_interpretation='Three local policy seeds share training checkpoint seed101; POLUS seed101 is a paired platform repeat, not an independent fourth seed.',
          inference_interpretation='Existing per-run 5-warmup/20-pass timings only; local tasks ran concurrently, library/thread/host load differ; no isolated hardware benchmark.',
          label_boundary='Validation only; no test-set evaluation or selection.',preflight_excluded=True))
    print(json.dumps({'verified_runs':len(rows),'verified_training_episodes':len(episode_rows),'paired_platform_runs':len(pairs),'output':str(out)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--local',type=Path,required=True);parser.add_argument('--remote',type=Path,required=True)
    parser.add_argument('--collection',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    run(a.local,a.remote,a.collection,a.output)
