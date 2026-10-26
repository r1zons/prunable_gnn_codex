"""Independent read-only validation of Cora GCN/GraphSAGE physical smoke artifacts."""
import csv, hashlib, json, math
from pathlib import Path
import torch, yaml
from sklearn.metrics import f1_score
from gnn_pruning.data import load_dataset
from gnn_pruning.models import build_model
from gnn_pruning.config import load_yaml, resolve_config
from gnn_pruning.pruning.workflow_v2 import tensor_hash, config_hash, _json_hash, _validate_dense_architecture

REPO=Path.cwd();OUT=REPO.parent/'prunable_gnn_artifacts/local-structured-v2-cora-matrix-20261027';OLD=REPO.parent/'prunable_gnn_artifacts/local-structured-v2-smoke-20261027'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def close(a,b,tol=1e-12):
    assert math.isclose(a,b,abs_tol=tol,rel_tol=tol),(a,b)

def pcount(kind,a,b):
    return (2 if kind=='graphsage' else 1)*(1433*a+a*b+7*b)+a+b+7

def legal(widths,step,greedy=False):
    actions=[f'prune:{layer}:{k}' for layer,w in enumerate(widths) for k in (1,2,4,8) if w-k>=4]
    return actions+(['stop'] if step>0 and not greedy else [])

def state(widths,step,accuracy,terminal):
    return ('local_v2',tuple(widths),3-step,math.floor(accuracy/.01),terminal)

def saved_state(s):
    return (s[0],tuple(s[1]),s[2],s[3],s[4])

class Oracle:
    """Project original dense tensors directly; use no V2 surgery or search helpers."""
    def __init__(self, kind, source, data, split):
        self.kind,self.source,self.data,self.split=kind,source,data,split
        self.model_config=source['model_config'];self.dense_state=source['model_state_dict']
        self.p0=pcount(kind,32,32)
        self.dense_model=self.rebuild([list(range(32)),list(range(32))]);self.dense_logits=self.forward(self.dense_model)
        self.a0=self.validation_float32(self.dense_logits)
        self.cache={}

    def projected_state(self,ids):
        result={}
        for name,v in self.dense_state.items():
            layer=int(name.split('.')[1]);value=v
            if layer<2:value=value[ids[layer]]
            if layer>0 and value.ndim==2:value=value[:,ids[layer-1]]
            result[name]=value.clone()
        return result

    def rebuild(self,ids):
        cfg=dict(self.model_config,hidden_channels=[len(i) for i in ids]);model=build_model(self.kind,**cfg).eval()
        model.load_state_dict(self.projected_state(ids),strict=True)
        assert sum(p.numel() for p in model.parameters())==pcount(self.kind,*map(len,ids))
        return model

    def forward(self,model):
        with torch.no_grad():logits=model(self.data)
        assert logits.shape==(2708,7) and torch.isfinite(logits).all()
        return logits

    def validation_float32(self,logits):
        idx=torch.tensor(self.split['val']);return float((logits.argmax(1)[idx]==self.data.y[idx]).float().mean())

    def metric(self,logits,key):
        idx=torch.tensor(self.split[key]);true=self.data.y[idx].numpy();pred=logits.argmax(1)[idx].numpy()
        return dict(accuracy=float((true==pred).mean()),macro_f1=float(f1_score(true,pred,average='macro',zero_division=0)))

    def trajectory(self,entries,target,greedy):
        ids=[list(range(32)),list(range(32))];widths=[32,32];step=0;previous_s=0.;evaluations=1;acc=self.a0
        assert entries
        for index,e in enumerate(entries):
            assert e['widths_before']==widths
            action=e['action'];assert action in legal(widths,step,greedy)
            keep=e['original_kept_ids'];assert len(keep)==2
            for layer,values in enumerate(keep):assert values==sorted(set(values)) and set(values)<=set(ids[layer]) and len(values)>=4
            if action=='stop':
                assert keep==ids;reward=0.;reason='agent_stop'
            else:
                _,layer,k=action.split(':');layer,k=int(layer),int(k)
                assert keep[1-layer]==ids[1-layer] and len(keep[layer])==len(ids[layer])-k
                ids=keep;widths=list(map(len,ids));step+=1;evaluations+=1
                key=tuple(map(tuple,ids))
                if key not in self.cache:self.cache[key]=self.validation_float32(self.forward(self.rebuild(ids)))
                acc=self.cache[key];close(e['validation_accuracy'],acc)
                s=1-pcount(self.kind,*widths)/self.p0;drop=max(0.,self.a0-acc)
                reward=.6*(s-previous_s)-.4*drop
                reason='max_steps' if step==3 else ('no_valid_actions' if not legal(widths,0,True) else ('target_reached' if greedy and s>=target-.01 else None))
                previous_s=s
            penalty=-2.*max(0.,abs(previous_s-target)-.01) if reason else 0.
            close(e['reward'],reward+penalty);close(e['terminal_penalty'],penalty);close(e['accuracy_drop'],max(0.,self.a0-acc))
            close(e['sparsity'],previous_s);close(e['signed_target_deviation'],previous_s-target)
            assert e['widths_after']==widths and e['successful_deletions']==step and e['terminal_reason']==reason
            assert (reason is not None)==(index==len(entries)-1)
            close(e['validation_accuracy'],acc)
        return evaluations

    def qcheck(self,trace,q,stats,target):
        training=trace['training'];deployment=trace['deployment'];assert stats['episodes']==3
        resets=[e for e in training if e.get('event')=='reset'];assert len(resets)==3
        assert [e['episode'] for e in resets]==[0,1,2] and all(e['widths']==[32,32] for e in resets)
        table={};visited=set();slots={};evaluations=self.trajectory(deployment,target,False)
        for episode in range(3):
            entries=[e for e in training if e.get('episode')==episode and e.get('event')!='reset']
            evaluations+=self.trajectory(entries,target,False)
            widths=[32,32];step=0;acc=self.a0
            for e in entries:
                s=state(widths,step,acc,False);actions=legal(widths,step);visited.add(s);slots[s]=len(actions)
                widths=e['widths_after'];step=e['successful_deletions'];acc=e['validation_accuracy'];terminal=e['terminal_reason'] is not None
                ns=state(widths,step,acc,terminal);nextactions=[] if terminal else legal(widths,step);visited.add(ns);slots[ns]=len(nextactions)
                future=0. if terminal else max(table.get(ns,{}).get(a,0.) for a in nextactions)
                row=table.setdefault(s,{});old=row.get(e['action'],0.);row[e['action']]=old+.1*(e['reward']+.9*future-old)
        serialized={saved_state(row['state']):row['actions'] for row in q};assert table.keys()==serialized.keys()
        for s,row in table.items():
            assert row.keys()==serialized[s].keys()
            for a,v in row.items():close(v,serialized[s][a])
        widths=[32,32];step=0;acc=self.a0
        for e in deployment:
            s=state(widths,step,acc,False);actions=legal(widths,step);visited.add(s);slots[s]=len(actions)
            assert e['action']==max(actions,key=lambda a:table.get(s,{}).get(a,0.))
            widths=e['widths_after'];step=e['successful_deletions'];acc=e['validation_accuracy']
        s=state(widths,step,acc,True);visited.add(s);slots[s]=0
        assert stats['validation_evaluations']==evaluations and stats['unique_states']==len(visited)
        assert stats['q_entries']==sum(len(row) for row in table.values()) and stats['legal_state_action_slots']==sum(slots.values())
        close(stats['q_table_sparsity'],1-stats['q_entries']/sum(slots.values()))


torch.set_num_threads(2);prov=json.loads((OUT/'provenance.json').read_text());data=load_dataset('cora',root=prov['data_root'])[0]
assert tensor_hash({k:getattr(data,k) for k in ('x','y','edge_index')})==prov['graph_tensor_sha256']
oracles={}
for kind,entry in prov['selected'].items():
    c=torch.load(entry['checkpoint'],weights_only=True,map_location='cpu');split=yaml.safe_load(Path(entry['split']).read_text())
    assert _json_hash(split)==entry['run_signature']['split_hash']
    oracles[kind]=Oracle(kind,c,data,split)

runs=json.loads((OUT/'cli_runs.json').read_text());assert len(runs)==18
assert {(r['model'],r['target_sparsity']) for r in runs}=={('gcn',.5),('graphsage',.3),('graphsage',.5)}
rows=[]
for run in runs:
    assert run['exit_code']==0,run
    kind,method,target=run['model'],run['method'],run['target_sparsity'];o=oracles[kind];frozen=prov['selected'][kind];d=Path(run['artifact_dir'])
    for name in ('pruned.pt','metrics.json','config_snapshot.json','trace.json','q_table.json'):assert (d/name).is_file()
    c=torch.load(d/'pruned.pt',weights_only=True,map_location='cpu');r=json.loads((d/'metrics.json').read_text());trace=json.loads((d/'trace.json').read_text());q=json.loads((d/'q_table.json').read_text())
    ids=c['original_kept_ids'];assert ids==r['original_kept_ids']
    for kept in ids:assert kept==sorted(set(kept)) and len(kept)>=4 and min(kept)>=0 and max(kept)<32
    expected=o.projected_state(ids);assert expected.keys()==c['model_state_dict'].keys()
    for name,value in expected.items():assert torch.equal(value,c['model_state_dict'][name]),name
    model=build_model(kind,**c['model_config']).eval();model.load_state_dict(c['model_state_dict'],strict=True);logits=o.forward(model)
    after=sum(p.numel() for p in model.parameters());widths=list(map(len,ids))
    assert model.hidden_channel_dims==r['hidden_widths']==widths
    assert after==r['parameters_after']==pcount(kind,*widths)<o.p0==r['parameters_before']
    close(r['requested_sparsity'],target);close(r['achieved_sparsity'],1-after/o.p0);close(r['target_gap'],abs(1-after/o.p0-target))
    assert r['target_violation']==(r['target_gap']>.01) and not r['unpruned']
    assert r['parameter_bytes']==sum(p.numel()*p.element_size() for p in model.parameters()) and r['checkpoint_bytes']==(d/'pruned.pt').stat().st_size
    for phase,ls in [('dense_metrics',o.dense_logits),('final_metrics',logits)]:
        for key in ('val','test'):
            for metric,value in o.metric(ls,key).items():close(r[phase][key][metric],value)
    close(r['dense_metrics']['val']['accuracy'],frozen['dense_validation_metrics']['accuracy'])
    cfg=load_yaml(run['config']);snapshot=json.loads((d/'config_snapshot.json').read_text());assert cfg==snapshot['source'] and resolve_config(run['config']).to_dict()==snapshot['resolved']
    _validate_dense_architecture(o.source,resolve_config(run['config']).model)
    p=r['provenance'];assert p==c['provenance'] and c['model_name']==kind and c['method']==r['method']==method and c['source_checkpoint']==frozen['checkpoint']
    assert p['checkpoint_sha256']==sha(frozen['checkpoint'])==frozen['checkpoint_sha256']
    assert p['split_sha256']==sha(frozen['split'])==frozen['split_sha256'] and p['split_indices']==o.split
    assert p['graph_tensor_sha256']==prov['graph_tensor_sha256'] and p['config_sha256']==config_hash(run['config'])
    assert p['dense_tensor_sha256']==tensor_hash(o.dense_model.state_dict())==frozen['dense_tensor_sha256'] and p['dense_model_config']==o.model_config
    assert p['git_sha']=='11ab841b4960d9591dd13feb240b7b7e5639c6a3' and not p['tracked_worktree_dirty']
    assert p['dataset']=='cora' and p['seed']==101 and p['device']=='cpu' and p['protocol']=='LOCAL-STRUCTURED-V2-R2'
    assert r['inference_benchmark'] is None and r['reporting_evaluations']==4
    for field in ('scoring_seconds','search_seconds','materialization_seconds','physical_surgery_seconds','total_scoring_seconds'):
        assert math.isfinite(r[field]) and r[field]>=0
    stats=r['statistics'];reason='static_plan_complete';deletions=None;validations=0
    if method in ('adaptive_greedy_local_v2','q_learning_local_v2'):
        deployed=trace['deployment'];assert deployed[-1]['original_kept_ids']==ids and deployed[-1]['widths_after']==widths
        reason=stats['terminal_reason'];assert reason==deployed[-1]['terminal_reason'];deletions=deployed[-1]['successful_deletions'];validations=stats['validation_evaluations']
        close(deployed[-1]['validation_accuracy'],r['final_metrics']['val']['accuracy'],1e-7)
        if method=='adaptive_greedy_local_v2':
            o.trajectory(deployed,target,True);assert not trace['training'] and not q
            count=sum(len(legal(e['widths_before'],e['successful_deletions']-1,True)) for e in deployed)
            assert stats['candidate_evaluations']==count and validations==count+1
        else:o.qcheck(trace,q,stats,target)
    else:
        assert trace==dict(deployment=[],training=[]) and not q and not stats
        static=r['static_plan'];assert static['kept_indices']==ids and static['widths']==widths
        assert static['parameters_after']==after and static['parameters_before']==o.p0
        close(static['achieved_sparsity'],r['achieved_sparsity']);close(static['target_gap'],r['target_gap'])
        assert static['target_violation']==r['target_violation'] and static['candidate_count']==840
    row=dict(model=kind,method=method,seed=101,target_sparsity=target,achieved_sparsity=r['achieved_sparsity'],validation_accuracy=r['final_metrics']['val']['accuracy'],validation_macro_f1=r['final_metrics']['val']['macro_f1'],terminal_reason=reason,successful_deletions=deletions,validation_evaluations=validations,search_seconds=r['search_seconds'],cli_seconds=run['cli_seconds'],status='PASS',scoring_seconds=r['scoring_seconds'],materialization_seconds=r['materialization_seconds'],physical_surgery_seconds=r['physical_surgery_seconds'],total_scoring_seconds=r['total_scoring_seconds'],reporting_evaluations=r['reporting_evaluations'],parameters_before=o.p0,parameters_after=after,hidden_widths=widths,target_gap=r['target_gap'],target_violation=r['target_violation'],final_test_metrics_recomputed=o.metric(logits,'test'),statistics=stats,artifact_dir=str(d),config=run['config'],log=run['log'],executed_this_task=True,hashes={name:sha(d/name) for name in ('pruned.pt','metrics.json','trace.json','q_table.json','config_snapshot.json')})
    rows.append(row);print(f"Verified {run['run_id']} P={after}, widths={widths}, terminal={reason}",flush=True)

# Direct source contract and the existing passing permutation regression complement
# the unmodified real CLI. No extra real-data pruning is executed for this audit.
workflow=(REPO/'src/gnn_pruning/pruning/workflow_v2.py').read_text();static_source=(REPO/'src/gnn_pruning/pruning/static_v2.py').read_text()
assert 'search_data.y[idx["test"]] = -1' in workflow and workflow.index('search_data.y[idx["test"]] = -1')<workflow.index('scores = pruner.score')
assert '"train_idx": idx["train"]' in workflow and 'predicted == search_data.y[idx["val"]]' in workflow
assert 'loss = F.cross_entropy(working(batch)[indices], batch.y[indices], reduction="mean")' in static_source
before=json.loads((OUT/'protected_files_before.json').read_text());changed=[p for p,h in before.items() if not Path(p).is_file() or sha(p)!=h];assert not changed,changed
verification=dict(status='PASS',new_runs_verified=len(rows),runs=rows,protected_files_verified=len(before),previous_artifacts_modified=changed,graph_tensor_sha256=prov['graph_tensor_sha256'],test_label_isolation='Unmodified workflow source verified: test y=-1 before scoring/search, only train indices and validation callback enter search. Existing full V2 suite passed including GCN test-label permutation regression and both-model integration. Full test metrics recalculated only after termination; no tuning or policy decision used them.',trace_checks='All selected and Q-training transitions independently projected from frozen dense original IDs and evaluated on real Cora validation indices; every reward/penalty, STOP count and terminal priority checked. Q Bellman table/greedy deployment and statistics replayed independently.',limitations=[prov['limitation'],'Static successful_deletions is null (not an adaptive trajectory); static search validation_evaluations=0 excludes four post-search reporting calls.','No latency benchmark, CUDA verification, Q-convergence claim, or scientific hyperparameter approval.'])
(OUT/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False))
print(f"PASS: {len(rows)} runs; {len(before)} protected files unchanged.",flush=True)
