"""Validation-only diagnostics calling unchanged production LOCAL-STRUCTURED-V2 routines."""
import argparse, collections, contextlib, hashlib, json, math, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch, yaml
from sklearn.metrics import f1_score
from gnn_pruning.data import load_dataset
from gnn_pruning.models import build_model
from gnn_pruning.pruning import adaptive_v2 as adaptive
from gnn_pruning.pruning.static_v2 import plan_static, apply_static_plan
from gnn_pruning.pruning.workflow_v2 import tensor_hash, _json_hash

ROOT=Path.cwd().parent/'prunable_gnn_artifacts/local-structured-v2-q-random-pilot-20261027'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def pcount(kind,widths):
    a,b=widths;return (2 if kind=='graphsage' else 1)*(1433*a+a*b+7*b)+a+b+7

def project(source, ids):
    result={}
    for name,value in source['model_state_dict'].items():
        layer=int(name.split('.')[1]);v=value
        if layer<2:v=v[ids[layer]]
        if layer>0 and v.ndim==2:v=v[:,ids[layer-1]]
        result[name]=v.clone()
    return result

def load_inputs(kind):
    prov=json.loads((ROOT/'provenance.json').read_text());entry=prov['selected'][kind]
    assert sha(entry['checkpoint'])==entry['checkpoint_sha256'] and sha(entry['split'])==entry['split_sha256']
    source=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
    split=yaml.safe_load(Path(entry['split']).read_text());assert _json_hash(split)==source['run_signature']['split_hash']
    assert source['run_signature']['seed']==101 and source['run_signature']['model_name']==kind
    data=load_dataset('cora',root=prov['data_root'])[0].clone()
    assert tensor_hash({k:getattr(data,k) for k in ('x','y','edge_index')})==prov['graph_tensor_sha256']
    idx=torch.tensor(split['val']);val_y=data.y[idx].clone()
    # Standalone pilot receives only validation labels, including for all reporting.
    data.y=torch.full_like(data.y,-1);data.y[idx]=val_y
    for name in ('train_mask','val_mask','test_mask'):
        if name in data:del data[name]
    assert int((data.y>=0).sum())==len(idx)
    model=build_model(kind,**source['model_config']).eval();model.load_state_dict(source['model_state_dict'],strict=True)
    return model,data,idx,source,entry,prov

def logits(model,data):
    model.eval()
    with torch.no_grad():result=model(data)
    assert result.shape==(2708,7) and torch.isfinite(result).all()
    return result

def val_metrics(model,data,idx):
    pred=logits(model,data).argmax(1)[idx].numpy();true=data.y[idx].numpy()
    return dict(accuracy=float((pred==true).mean()),macro_f1=float(f1_score(true,pred,average='macro',zero_division=0)))

@contextlib.contextmanager
def observe_q(episodes):
    """Observe original calls; clone RNG only for labeling exploration, never advance it."""
    original_reset=adaptive.LocalV2Env.reset;original_select=adaptive.select_action;original_update=adaptive.update_q
    audit={'resets':[],'decisions':[]};active={}
    def reset(env):
        result=original_reset(env);active['env']=env;active['episode']=len(audit['resets'])
        audit['resets'].append({'episode':active['episode'] if active['episode']<episodes else None,'phase':'training' if active['episode']<episodes else 'deployment','widths':list(env.state()[1]),'dense_tensor_sha256':tensor_hash(env.model.state_dict())})
        return result
    def select(table,state,actions,rng,epsilon=0.):
        row=table.get(state,{});probe=random.Random();probe.setstate(rng.getstate());exploration=epsilon>0 and probe.random()<epsilon
        choice=original_select(table,state,actions,rng,epsilon)
        env=active['env'];keys=[adaptive.action_key(a) for a in actions]
        audit['decisions'].append(dict(phase='training' if active['episode']<episodes else 'deployment',episode=active['episode'] if active['episode']<episodes else None,state=state,exact_validation_accuracy=env.accuracy,original_kept_ids=[list(x) for x in env.original_kept_ids],legal_q={k:row.get(k,0.) for k in keys},visited_actions=[k for k in keys if k in row],selected_action=adaptive.action_key(choice),exploration=exploration,selected_was_unvisited=adaptive.action_key(choice) not in row))
        return choice
    def update(table,state,action,reward,next_state,next_actions,terminal,alpha=.1,gamma=.9):
        old=table.get(state,{}).get(action,0.);future=0. if terminal else max((table.get(next_state,{}).get(k,0.) for k in next_actions),default=0.)
        original_update(table,state,action,reward,next_state,next_actions,terminal,alpha,gamma)
        env=active['env'];audit['decisions'][-1]['update']=dict(q_before=old,bootstrap=future,reward=reward,q_after=table[state][action],terminal=terminal,next_state=next_state,next_actions=next_actions,original_kept_ids_after=[list(x) for x in env.original_kept_ids])
    adaptive.LocalV2Env.reset,adaptive.select_action,adaptive.update_q=reset,select,update
    try:yield audit
    finally:adaptive.LocalV2Env.reset,adaptive.select_action,adaptive.update_q=original_reset,original_select,original_update

def verify_physical(kind,model,source,ids,data,idx):
    expected=project(source,ids);actual=model.state_dict();assert actual.keys()==expected.keys()
    for k,v in expected.items():assert torch.equal(v,actual[k]),k
    count=sum(p.numel() for p in model.parameters());assert count==pcount(kind,list(map(len,ids)))
    assert all(x==sorted(set(x)) and len(x)>=4 for x in ids)
    return count,val_metrics(model,data,idx)

def checkpoint(d,kind,model,source,ids,entry,prov,config):
    architecture=model.export_architecture_config();architecture.pop('name')
    path=d/'pruned.pt';torch.save(dict(model_name=kind,model_config=architecture,model_state_dict=model.state_dict(),original_kept_ids=ids,source_checkpoint=entry['checkpoint'],checkpoint_training_seed=101,graph_tensor_sha256=prov['graph_tensor_sha256'],checkpoint_sha256=entry['checkpoint_sha256'],split_sha256=entry['split_sha256'],diagnostic_config=config),path)
    c=torch.load(path,map_location='cpu',weights_only=True);rebuilt=build_model(kind,**c['model_config']).eval();rebuilt.load_state_dict(c['model_state_dict'],strict=True)
    for k,v in rebuilt.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    return path

def run_q(kind,target,episodes,penalty):
    start=time.perf_counter();model,data,idx,source,entry,prov=load_inputs(kind)
    run_id=f'{kind}_target{int(target*100)}_episodes{episodes}_lambda{penalty:g}'
    d=ROOT/'q_runs'/run_id;d.mkdir(parents=True,exist_ok=False)
    config=dict(kind=kind,target=target,episodes=episodes,penalty_lambda=penalty,max_steps=20,min_channels=4,tolerance=.01,gamma=.9,alpha=.1,epsilon=.2,policy_seed=101,checkpoint_training_seed=101,device='cpu',threads=torch.get_num_threads(),label_access='validation_only',input_provenance=str(ROOT/'provenance.json'))
    write(d/'config.json',config);dense=val_metrics(model,data,idx);dense_hash=tensor_hash(model.state_dict());evaluations=0
    def evaluate(candidate):
        nonlocal evaluations
        evaluations+=1;pred=logits(candidate,data).argmax(1)[idx]
        return float((pred==data.y[idx]).float().mean())
    search_start=time.perf_counter()
    with observe_q(episodes) as audit:
        result=adaptive.run_q_learning(model,evaluate,target,penalty,episodes=episodes,epsilon=.2,alpha=.1,gamma=.9,seed=101,max_steps=20,min_channels=4,tolerance=.01)
    search_seconds=time.perf_counter()-search_start
    assert evaluations==result.statistics['validation_evaluations']
    assert all(r['dense_tensor_sha256']==dense_hash for r in audit['resets']) and len(audit['resets'])==episodes+1
    count,metrics=verify_physical(kind,result.model,source,result.kept_indices,data,idx)
    path=checkpoint(d,kind,result.model,source,result.kept_indices,entry,prov,config)
    assert tensor_hash(model.state_dict())==dense_hash and sha(entry['checkpoint'])==entry['checkpoint_sha256']
    write(d/'trace.json',dict(training=result.training_trace,deployment=result.trace));write(d/'q_table.json',[{'state':s,'actions':row} for s,row in result.q_table.items()]);write(d/'audit.json',audit)
    achieved=1-count/sum(p.numel() for p in model.parameters())
    report=dict(run_id=run_id,model=kind,checkpoint_training_seed=101,policy_seed=101,target_sparsity=target,episodes=episodes,penalty_lambda=penalty,max_steps=20,hidden_widths=result.model.hidden_channel_dims,original_kept_ids=result.kept_indices,parameters_after=count,achieved_sparsity=achieved,signed_target_deviation=achieved-target,target_gap=abs(achieved-target),target_violation=abs(achieved-target)>.01,validation_accuracy=metrics['accuracy'],validation_macro_f1=metrics['macro_f1'],dense_validation=dense,terminal_reason=result.statistics['terminal_reason'],successful_deletions=result.trace[-1]['successful_deletions'],statistics=result.statistics,candidate_evaluations=0,search_seconds=search_seconds,scoring_seconds=result.statistics['scoring_seconds'],surgery_seconds=result.statistics['surgery_seconds'],policy_other_seconds=search_seconds-result.statistics['scoring_seconds']-result.statistics['surgery_seconds'],runner_seconds=time.perf_counter()-start,checkpoint=str(path),config=config,source_checkpoint_sha256=entry['checkpoint_sha256'],split_sha256=entry['split_sha256'],graph_sha256=prov['graph_tensor_sha256'],status='PASS')
    write(d/'result.json',report);print(json.dumps({k:report[k] for k in ('run_id','hidden_widths','achieved_sparsity','validation_accuracy','terminal_reason','search_seconds','status')}),flush=True)

def run_random():
    for kind in ('gcn','graphsage'):
        model,data,idx,source,entry,prov=load_inputs(kind);before=tensor_hash(model.state_dict())
        for target in (.3,.5):
            for seed in range(1000,1020):
                start=time.perf_counter();d=ROOT/'random_runs'/f'{kind}_target{int(target*100)}_mask{seed}';d.mkdir(parents=True,exist_ok=False)
                plan=plan_static(model,{},'random',target,tolerance=.01,min_channels=4,seed=seed)
                final=apply_static_plan(model,plan);count,metrics=verify_physical(kind,final,source,plan.kept_indices,data,idx)
                config=dict(model=kind,target_sparsity=target,mask_seed=seed,checkpoint_training_seed=101,plan_function='static_v2.plan_static',criterion='random',tolerance=.01,min_channels=4,label_access='validation_only')
                path=checkpoint(d,kind,final,source,plan.kept_indices,entry,prov,config)
                report=dict(model=kind,target_sparsity=target,mask_seed=seed,checkpoint_training_seed=101,hidden_widths=plan.widths,original_kept_ids=plan.kept_indices,parameters_after=count,achieved_sparsity=1-count/sum(p.numel() for p in model.parameters()),validation_accuracy=metrics['accuracy'],validation_macro_f1=metrics['macro_f1'],seconds=time.perf_counter()-start,checkpoint=str(path),config=config,static_plan=asdict(plan),source_checkpoint_sha256=entry['checkpoint_sha256'],split_sha256=entry['split_sha256'],graph_sha256=prov['graph_tensor_sha256'],status='PASS')
                write(d/'result.json',report)
            print(f'Random {kind} target={target}:20/20 PASS',flush=True)
        assert tensor_hash(model.state_dict())==before

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['q','random']);parser.add_argument('--model',choices=['gcn','graphsage']);parser.add_argument('--target',type=float);parser.add_argument('--episodes',type=int);parser.add_argument('--penalty-lambda',type=float,default=2.)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.mode=='q':run_q(args.model,args.target,args.episodes,args.penalty_lambda)
    else:run_random()
