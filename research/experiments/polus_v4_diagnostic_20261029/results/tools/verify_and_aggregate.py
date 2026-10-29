"""Offline independent V4 audit. No training/search/rollouts or algorithm imports.

Only frozen input loading and model construction are reused. All menu geometry,
RNG/action selection, rewards, Bellman updates, coverage and snapshot checks below
are independent equations evaluated on archived observations.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, functools, hashlib, json, math, random, re, statistics
from pathlib import Path
import torch
from gnn_pruning.models import build_model
from gnn_pruning.experiments.v3_sprint import load_frozen

REPO=Path(__file__).resolve().parents[5]
EVIDENCE=REPO/'research/experiments/polus_v4_diagnostic_20261029'
SOURCE='61e693440b53d3cac3136600f3208a73c02492db'
V3_SOURCE='7e2c760e988b605e305615dea147236e8b0c59a3'
BASE=dict(menu='standard',episodes=100,epsilon=.2,decay=False,random_ties=False,incremental_drop=False,gamma=.9,greedy=False)
PLANS={
 'q_v4_menu12':dict(BASE,menu='one_two'), 'q_v4_percent':dict(BASE,menu='percent'),
 'q_v4_300':dict(BASE,episodes=300), 'q_v4_decay300':dict(BASE,episodes=300,epsilon=.4,decay=True),
 'q_v4_random_ties':dict(BASE,random_ties=True), 'q_v4_incremental_drop':dict(BASE,incremental_drop=True),
 'q_v4_gamma0':dict(BASE,gamma=0.), 'greedy_v4_menu12':dict(BASE,menu='one_two',greedy=True),
 'greedy_v4_percent':dict(BASE,menu='percent',greedy=True),
 'q_reachable_v3':dict(BASE), 'greedy_reachable_v3':dict(BASE,greedy=True),
}

def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
def close(a,b):assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10),(a,b)
def state_key(s):return (s[0],tuple(s[1]),*s[2:])
def normalize_q(rows):
 out={state_key(r['state']):r['actions'] for r in rows}
 assert len(out)==len(rows),'Duplicate serialized Q state'
 return out
def same_q(a,b):
 assert set(a)==set(b)
 for s,row in a.items():
  assert set(row)==set(b[s])
  for k,v in row.items():close(v,b[s][k])
def csv_write(p,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def count(kind,widths):
 a,b=widths
 return (2 if kind=='graphsage' else 1)*(1433*a+a*b+7*b)+a+b+7
def sizes(menu,w):
 if menu=='standard':return (1,2,4,8)
 if menu=='one_two':return (1,2)
 assert menu=='percent'
 return tuple(sorted({math.ceil(w/20),math.ceil(w/10)}))

@functools.lru_cache(None)
def distances(menu,start):
 # Independent breadth-first directed-width graph, recomputing sizes per vertex.
 d={start:0};queue=collections.deque([start])
 while queue:
  width=queue.popleft()
  for k in sizes(menu,width):
   after=width-k
   if after>=4 and after not in d:d[after]=d[width]+1;queue.append(after)
 return d
@functools.lru_cache(None)
def goals(kind,target):
 p0=count(kind,(32,32));percent=round(target*100)
 return tuple((a,b) for a in range(4,33) for b in range(4,33)
              if abs(100*(p0-count(kind,(a,b)))-percent*p0)<=p0)
@functools.lru_cache(None)
def reachable(kind,target,menu,widths,remaining):
 if remaining<0:return False
 d0,d1=distances(menu,widths[0]),distances(menu,widths[1])
 return any(d0.get(a,999)+d1.get(b,999)<=remaining for a,b in goals(kind,target))
@functools.lru_cache(None)
def legal(kind,target,menu,widths,remaining,terminal=False):
 if terminal:return ()
 result=[]
 for layer in (0,1):
  for k in sizes(menu,widths[layer]):
   after=list(widths);after[layer]-=k
   if after[layer]>=4 and reachable(kind,target,menu,tuple(after),remaining-1):result.append(f'prune:{layer}:{k}')
 assert result,'No continuation before terminal band'
 return tuple(result)


def audit_learning(cfg,report,trace,qrows,decisions,snapshot_dir,ranking=None):
 """Replay every archived transition/RNG/Q update; no reevaluation of trajectories."""
 plan=PLANS[cfg['variant']];kind=cfg['model'];target=cfg['target'];p0=count(kind,(32,32))
 dense=float(torch.tensor(report['dense_validation']['accuracy'],dtype=torch.float32))
 table={};rng=random.Random(cfg['policy_seed']);visited=set();slots={};decision_index=0
 episodes=[];snapshots=[];deployment_decisions=[];transitions=0;rank_checks=0
 signatures=collections.defaultdict(set);observed_rewards=collections.defaultdict(set)
 episodes_count=0 if plan['greedy'] else cfg['episodes'];resets=[];by_episode=collections.defaultdict(list)
 for e in trace['training']:
  if e.get('event')=='reset':assert e['widths']==[32,32];resets.append(e['episode'])
  else:by_episode[e['episode']].append(e)
 assert resets==list(range(episodes_count)) and set(by_episode)==set(range(episodes_count))
 if plan['greedy']:assert not qrows and not decisions
 def sequence(events,episode=None):
  nonlocal decision_index,transitions,rank_checks
  learning=episode is not None;steps=0;widths=(32,32);ids=[list(range(32)),list(range(32))]
  accuracy=dense;previous_s=0.;previous_d=0.;terminal=None;total_reward=discounted=common_reward=0.
  stats=collections.Counter();initial_signature=tuple(map(tuple,ids))
  epsilon=(.4+(.05-.4)*episode/299 if plan['decay'] else .2) if learning else 0.
  for t,event in enumerate(events):
   assert terminal is None
   state=('local_v2',widths,20-steps,math.floor(accuracy/.01),False)
   signatures[state].add(tuple(map(tuple,ids)))
   actions=legal(kind,target,plan['menu'],widths,20-steps);row=table.get(state,{})
   values={a:row.get(a,0.) for a in actions};best=max(values.values());ties=[a for a in actions if values[a]==best]
   action=event['action'];assert action in actions and tuple(event['widths_before'])==widths
   if not plan['greedy']:
    mode='explore' if epsilon>0 and rng.random()<epsilon else 'exploit'
    selected=rng.choice(actions) if mode=='explore' else rng.choice(ties) if plan['random_ties'] and len(ties)>1 else ties[0]
    assert action==selected,(cfg['run_id'],episode,t,'RNG/action mismatch')
    unseen=action not in row
    stats['unseen']+=unseen;stats['explore']+=mode=='explore';stats['ties']+=len(ties)>1
    stats['nonzero_unseen_alternatives']+=unseen and any(v<0 for v in values.values())
    if decisions is not None:
     d=decisions[decision_index];decision_index+=1
     assert d['phase']==('training' if learning else 'deployment') and d['episode']==episode
     assert state_key(d['state'])==state and d['action']==action and d['mode']==mode
     close(d['epsilon'],epsilon)
     assert d['selected_action_unseen']==unseen and d['unseen_legal_actions']==sum(a not in row for a in actions)
     assert d['equal_max_count']==len(ties) and d['original_kept_ids']==ids
     assert set(d['legal_q_values'])==set(values)
     for a,v in values.items():close(d['legal_q_values'][a],v)
    if not learning:
     deployment_decisions.append(dict(step=t,action=action,unseen=unseen,tie_count=len(ties),mode=mode,
        selected_q=values[action],state_unseen=state not in table,unseen_legal_actions=sum(a not in row for a in actions)))
   _,layer,k=action.split(':');layer,k=int(layer),int(k);after=list(widths);after[layer]-=k
   after=tuple(after);assert tuple(event['widths_after'])==after
   new_ids=event['original_kept_ids']
   for i in (0,1):
    assert new_ids[i]==sorted(set(new_ids[i])) and len(new_ids[i])==after[i] and set(new_ids[i])<=set(ids[i])
    if i!=layer:assert new_ids[i]==ids[i]
   if ranking is not None:
    expected=ranking(kind,tuple(map(tuple,ids)),layer,k)
    assert tuple(new_ids[layer])==expected,(cfg['run_id'],episode,t,'Magnitude choice mismatch')
    rank_checks+=1
   steps+=1;s=1-count(kind,after)/p0;drop=max(0,dense-event['validation_accuracy'])
   reason='max_steps' if steps>=20 else 'no_valid_actions' if after==(4,4) else 'target_band' if after in goals(kind,target) else None
   penalty=-2*max(0,abs(s-target)-.01) if reason else 0.
   expected_reward=.6*(s-previous_s)-.4*(drop-previous_d if plan['incremental_drop'] else drop)+penalty
   assert event['successful_deletions']==steps and event['terminal_reason']==reason
   close(event['sparsity'],s);close(event['signed_target_deviation'],s-target);close(event['accuracy_drop'],drop)
   close(event['terminal_penalty'],penalty);close(event['reward'],expected_reward)
   observed_rewards[state,action].add(event['reward'])
   if plan['incremental_drop']:
    close(event['previous_accuracy_drop'],previous_d);close(event['accuracy_drop_difference'],drop-previous_d)
    stats['recoveries']+=drop<previous_d
   assert reachable(kind,target,plan['menu'],after,20-steps)
   next_state=('local_v2',after,20-steps,math.floor(event['validation_accuracy']/.01),reason is not None)
   next_actions=legal(kind,target,plan['menu'],after,20-steps,reason is not None)
   visited.add(state);slots[state]=len(actions);visited.add(next_state);slots[next_state]=len(next_actions)
   if learning:
    future=0. if reason else max(table.get(next_state,{}).get(a,0.) for a in next_actions)
    # Use the archived, independently checked reward to preserve native floating
    # operation order; algebraically equivalent E expressions can differ at1ulp.
    previous=row.get(action,0.);updated=previous+.1*(event['reward']+plan['gamma']*future-previous)
    if decisions is not None:
     close(d['q_before'],previous);close(d['q_after'],updated);close(d['bootstrap'],future)
     close(d['reward'],expected_reward);assert d['terminal']==bool(reason) and state_key(d['next_state'])==next_state
    table.setdefault(state,{})[action]=updated
    stats['bootstrap_nonzero']+=plan['gamma']*future!=0
   total_reward+=expected_reward;discounted+=plan['gamma']**t*expected_reward
   common_reward+=.9**t*(.6*(s-previous_s)-.4*drop+penalty)
   widths,ids,accuracy,previous_s,previous_d,terminal=after,new_ids,event['validation_accuracy'],s,drop,reason
   transitions+=1
  assert terminal is not None and widths in goals(kind,target) and steps<=20
  if plan['incremental_drop']:close(total_reward,.6*previous_s-.4*previous_d+events[-1]['terminal_penalty'])
  return dict(episode=episode,steps=steps,validation_accuracy=accuracy,achieved_sparsity=previous_s,
      return_value=total_reward,discounted_return=discounted,common_repeated_return_gamma09=common_reward,
      terminal_reason=terminal,unseen_choices=stats['unseen'],exploratory_choices=stats['explore'],
      tied_choices=stats['ties'],bootstrap_nonzero=stats['bootstrap_nonzero'],recoveries=stats['recoveries'],
      ids=ids,signature=initial_signature)
 for episode in range(episodes_count):
  e=sequence(by_episode[episode],episode);episodes.append(e)
  checkpoint=episode+1
  if snapshot_dir is not None and checkpoint in (25,50,100,200,300):
   path=Path(snapshot_dir)/f'policy_snapshot_{checkpoint:03}.json';saved=read(path)
   assert saved['episodes_completed']==checkpoint
   same_q(table,normalize_q(saved['q_table']))
   assert saved['rng_state']==json.loads(json.dumps(rng.getstate()))
   close(saved['epsilon_last'],.4+(.05-.4)*episode/299 if plan['decay'] else .2)
   snapshots.append(dict(episodes=checkpoint,q_entries=sum(map(len,table.values())),
      visited_states=len(visited),legal_slots=sum(slots.values()),coverage=sum(map(len,table.values()))/sum(slots.values()),
      q_sha256=canonical(saved['q_table']),rng_sha256=canonical(saved['rng_state'])))
 deployed=sequence(trace['deployment'])
 assert deployed['ids']==report['original_kept_ids'] and deployed['terminal_reason']==report['statistics']['terminal_reason']
 close(deployed['validation_accuracy'],float(torch.tensor(report['final_validation']['accuracy'],dtype=torch.float32)))
 assert decisions is None or decision_index==len(decisions)
 if not plan['greedy']:
  same_q(table,normalize_q(qrows));stats=report['statistics'];entries=sum(map(len,table.values()));total=sum(slots.values())
  assert stats['unique_states']==len(visited) and stats['q_entries']==entries and stats['legal_state_action_slots']==total
  close(stats['q_table_sparsity'],1-entries/total)
  assert stats['validation_evaluations']==episodes_count+1+sum(e['steps'] for e in episodes)+deployed['steps']
  if 'deployment_unseen_choices' in stats:assert stats['deployment_unseen_choices']==deployed['unseen_choices']
 else:
  assert report['statistics']['validation_evaluations']==report['statistics']['candidate_evaluations']+1
 return dict(transitions=transitions,training_episodes=episodes_count,magnitude_checks=rank_checks,
      q_entries=sum(map(len,table.values())),visited_states=len(visited),legal_slots=sum(slots.values()),
      snapshots_verified=len(snapshots),observed_aliased_states=sum(len(v)>1 for v in signatures.values()),
      state_action_reward_conflicts=sum(len(v)>1 for v in observed_rewards.values())),episodes,deployed,deployment_decisions,snapshots


def summarize(rows):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['platform'],r['model'],r['target'],r['variant']].append(r)
 out=[]
 for group,items in sorted(groups.items()):
  row=dict(zip(('platform','model','target','variant'),group));row.update(n=len(items),policy_seeds=';'.join(str(r['policy_seed']) for r in sorted(items,key=lambda r:r['policy_seed'])),
   target_hits=sum(r['target_hit'] for r in items),width_pairs=';'.join(f"{r['width0']}/{r['width1']}" for r in sorted(items,key=lambda r:r['policy_seed'])),
   stopping_reasons=json.dumps(dict(collections.Counter(r['terminal_reason'] for r in items)),sort_keys=True))
  for metric in ('validation_accuracy','validation_macro_f1','achieved_sparsity','target_gap','parameters_after','width0','width1',
     'search_seconds','validation_evaluations','q_coverage','q_entries','deployment_unseen_fraction','deployment_unseen_choices','deployment_tied_choices',
     'deployment_steps','training_exploration_fraction','training_unseen_choices','training_unseen_fraction',
     'training_bootstrap_contribution_fraction','observed_aliased_states','state_action_reward_conflicts'):
   values=[r[metric] for r in items if r[metric] is not None]
   row[metric+'_mean']=statistics.mean(values) if values else None
   row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
   row[metric+'_min']=min(values) if values else None;row[metric+'_max']=max(values) if values else None
  out.append(row)
 return out


def analyze(collection,inputs_path,out):
 out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt=read(EVIDENCE/'collection/collection_receipt.json');launch=read(EVIDENCE/'launch_receipt.json');declared=read(EVIDENCE/'configuration_index.json')
 assert sha(collection/'v4-primary-results.tar.gz')==receipt['archive_sha256']
 extracted=collection/'polus';inventory=read(extracted/'collection/inventory.json')
 assert len(inventory)==1285 and sha(extracted/'collection/inventory.json')==receipt['inventory_sha256']
 for entry in inventory:
  path=extracted/entry['path'];assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
 originals={str(extracted/entry['path']):entry['sha256'] for entry in inventory}
 for job in launch['campaign_jobs']:
  assert sha(extracted/'campaign-primary'/('manifest-'+job['condition']+'.json'))==job['manifest_sha256']
  assert 'Successfully completed.' in (extracted/'logs/lsf'/f"v4-cpu.{job['job_id']}.out").read_text()
 dense={kind:load_frozen(inputs_path,kind) for kind in ('gcn','graphsage')};input_manifest=read(inputs_path)
 for kind,(_,data,idx,_,_) in dense.items():
  nonval=torch.ones(2708,dtype=torch.bool);nonval[idx]=False
  assert (data.y[nonval]==-1).all() and set(data.y[idx].tolist())==set(range(7))
 @functools.lru_cache(maxsize=200000)
 def ranking(kind,kept,layer,k):
  # Independent incident |weight| sums on archived dense projections, including bias.
  scores=torch.zeros(len(kept[layer]),dtype=torch.float64)
  for name,value in dense[kind][3]['model_state_dict'].items():
   i=int(name.split('.')[1])
   if i!=layer and not (i==layer+1 and value.ndim==2):continue
   projected=value
   if i<2:projected=projected[list(kept[i])]
   if i>0 and projected.ndim==2:projected=projected[:,list(kept[i-1])]
   absolute=projected.abs()
   marginal=absolute.sum(1) if i==layer and absolute.ndim==2 else absolute if i==layer else absolute.sum(0)
   scores+=marginal.double()
  order=sorted(range(len(scores)),key=lambda c:(-float(scores[c]),c))
  return tuple(kept[layer][c] for c in sorted(order[:len(order)-k]))
 rows=[];baselines=[];audits=[];episode_rows=[];snapshot_rows=[];decision_rows=[];artifacts={}
 expected={c['run_id']:c for c in declared['configs']}
 attempts=sorted((extracted/'campaign-primary/outputs').glob('*/attempt-*/completed.json'))
 assert len(attempts)==92
 def process(path,primary=True):
  folder=path.parent;done=read(path);cfg=read(folder/'config.json');report=read(folder/'metrics.json');runid=cfg['run_id']
  assert done['status']=='completed' and done['run_id']==report['run_id']==runid
  assert done['config_sha256']==canonical(cfg)
  for name,digest in done['hashes'].items():assert sha(folder/name)==digest
  assert done['source_commit']==cfg['source_commit']==report['provenance']['source_commit']==(SOURCE if primary else V3_SOURCE)
  if primary:
   normalized=dict(cfg)
   for key in ('input_manifest','output_dir'):normalized[key]='${POLUS_TASK_ROOT}'+cfg[key].split('/gnn-polus-20261023',1)[1]
   assert normalized==expected[runid] and cfg['purpose']=='campaign' and cfg['plan']==PLANS[cfg['variant']]
   identity={k:cfg[k] for k in ('model','variant','target','policy_seed','training_seed','source_commit','purpose','episodes')}
   identity.update(plan=PLANS[cfg['variant']],max_steps=20,minimum=4,tolerance=.01,state_version='local_v2',alpha=.1,penalty_lambda=2.,
      graph=input_manifest['graph_tensor_sha256'],split=input_manifest['split']['sha256'],checkpoint=input_manifest['models'][cfg['model']]['sha256'])
   assert canonical(identity)==done['algorithm_identity_sha256']==report['provenance']['algorithm_identity_sha256']
  assert report['provenance']['label_access']=='validation_only' and cfg['training_seed']==101
  assert report['provenance']['graph_tensor_sha256']==input_manifest['graph_tensor_sha256']
  assert report['provenance']['split_sha256']==input_manifest['split']['sha256']
  assert report['provenance']['checkpoint_sha256']==input_manifest['models'][cfg['model']]['sha256']
  original=dense[cfg['model']][3];data=dense[cfg['model']][1];idx=dense[cfg['model']][2]
  saved=torch.load(folder/'pruned.pt',weights_only=True,map_location='cpu');ids=saved['original_kept_ids']
  assert saved['provenance']==report['provenance'] and ids==report['original_kept_ids']
  final=build_model(saved['model_name'],**saved['model_config']).eval();final.load_state_dict(saved['model_state_dict'],strict=True)
  widths=tuple(map(len,ids));assert list(widths)==report['hidden_widths']
  parameters=count(cfg['model'],widths);p0=count(cfg['model'],(32,32))
  assert sum(p.numel() for p in final.parameters())==parameters==report['parameters_after']<p0==report['parameters_before']
  for name,value in final.state_dict().items():
   i=int(name.split('.')[1]);projected=original['model_state_dict'][name]
   if i<2:projected=projected[ids[i]]
   if i>0 and projected.ndim==2:projected=projected[:,ids[i-1]]
   assert torch.equal(value,projected)
  with torch.no_grad():logits=final(data)
  assert logits.shape==(2708,7) and torch.isfinite(logits).all()
  pred=logits.argmax(1)[idx];labels=data.y[idx]
  matrix=torch.bincount(labels*7+pred,minlength=49).reshape(7,7).double()
  acc=float(matrix.diag().sum()/matrix.sum());f1=float((2*matrix.diag()/(matrix.sum(0)+matrix.sum(1)).clamp_min(1)).mean())
  close(acc,report['final_validation']['accuracy']);close(f1,report['final_validation']['macro_f1'])
  assert widths in goals(cfg['model'],cfg['target']);close(report['achieved_sparsity'],1-parameters/p0)
  trace=read(folder/'trace.json');qrows=read(folder/'q_table.json');decisions=read(folder/'decisions.json') if primary else None
  audit,episodes,deployed,deployment,snapshots=audit_learning(cfg,report,trace,qrows,decisions,folder if primary else None,ranking)
  stats=report['statistics'];isq=not PLANS[cfg['variant']]['greedy'];training_steps=sum(e['steps'] for e in episodes)
  row=dict(run_id=runid,platform=cfg['platform'],model=cfg['model'],target=cfg['target'],variant=cfg['variant'],policy_seed=cfg['policy_seed'],training_seed=101,
   source_commit=cfg['source_commit'],validation_accuracy=report['final_validation']['accuracy'],validation_macro_f1=report['final_validation']['macro_f1'],
   graph_tensor_sha256=input_manifest['graph_tensor_sha256'],split_sha256=input_manifest['split']['sha256'],
   dense_checkpoint_sha256=input_manifest['models'][cfg['model']]['sha256'],pruned_checkpoint_sha256=sha(folder/'pruned.pt'),
   algorithm_identity_sha256=report['provenance']['algorithm_identity_sha256'],
   dense_validation_accuracy=report['dense_validation']['accuracy'],dense_validation_macro_f1=report['dense_validation']['macro_f1'],
   validation_accuracy_drop=report['dense_validation']['accuracy']-report['final_validation']['accuracy'],
   achieved_sparsity=report['achieved_sparsity'],target_gap=abs(report['achieved_sparsity']-cfg['target']),target_hit=True,
   width0=widths[0],width1=widths[1],retained_ids=json.dumps(ids,separators=(',',':')),parameters_after=parameters,parameters_before=p0,
   parameter_bytes=4*parameters,checkpoint_bytes=(folder/'pruned.pt').stat().st_size,episodes=cfg['episodes'] if isq else 0,
   search_seconds=report['search_seconds'],total_seconds=report['total_seconds'],validation_evaluations=stats['validation_evaluations'],
   seconds_per_evaluation=report['search_seconds']/stats['validation_evaluations'],deployment_steps=deployed['steps'],terminal_reason=deployed['terminal_reason'],
   deployment_unseen_choices=deployed['unseen_choices'] if isq else None,deployment_unseen_fraction=deployed['unseen_choices']/deployed['steps'] if isq else None,
   deployment_tied_choices=deployed['tied_choices'] if isq else None,q_entries=audit['q_entries'] if isq else None,
   visited_states=audit['visited_states'] if isq else None,legal_visited_slots=audit['legal_slots'] if isq else None,
   q_coverage=audit['q_entries']/audit['legal_slots'] if isq else None,
   training_steps=training_steps,training_unseen_choices=sum(e['unseen_choices'] for e in episodes) if isq else None,
   training_unseen_fraction=sum(e['unseen_choices'] for e in episodes)/training_steps if isq else None,
   training_exploration_fraction=sum(e['exploratory_choices'] for e in episodes)/training_steps if isq else None,
   training_bootstrap_contribution_fraction=sum(e['bootstrap_nonzero'] for e in episodes)/training_steps if isq else None,
   deployment_return=deployed['return_value'],deployment_discounted_return=deployed['discounted_return'],
   common_repeated_return_gamma09=deployed['common_repeated_return_gamma09'],
   observed_aliased_states=audit['observed_aliased_states'] if isq else None,state_action_reward_conflicts=audit['state_action_reward_conflicts'] if isq else None,
   threads=report['runtime']['threads'],machine=report['runtime']['machine'],torch=report['runtime']['torch'],pyg=report['runtime']['pyg'])
  if primary:
   assert report['parameter_bytes']==row['parameter_bytes'] and report['checkpoint_bytes']==row['checkpoint_bytes']
   assert len(list(folder.glob('policy_snapshot_*.json')))==(5 if cfg['episodes']==300 else 3 if isq else 0)
   for e in episodes:
    e.pop('ids');e.pop('signature');episode_rows.append(dict(run_id=runid,model=cfg['model'],target=cfg['target'],variant=cfg['variant'],policy_seed=cfg['policy_seed'],**e))
   snapshot_rows.extend(dict(run_id=runid,model=cfg['model'],target=cfg['target'],variant=cfg['variant'],policy_seed=cfg['policy_seed'],**s) for s in snapshots)
   decision_rows.extend(dict(run_id=runid,model=cfg['model'],target=cfg['target'],variant=cfg['variant'],policy_seed=cfg['policy_seed'],**d) for d in deployment)
  audits.append(dict(run_id=runid,primary=primary,status='PASS',checkpoint_sha256=sha(folder/'pruned.pt'),completed_sha256=sha(path),
                    independently_recomputed_accuracy=acc,independently_recomputed_macro_f1=f1,**audit))
  artifacts[runid]=dict(folder=folder,trace=trace,q=qrows,cfg=cfg,row=row)
  return row
 for path in attempts:rows.append(process(path))
 refs=read(EVIDENCE/'reused_baselines.json')['references']
 for ref in refs:
  path=Path(ref['completed']);assert sha(path)==ref['completed_sha256'];baselines.append(process(path,False))
 oldroot=collection.parent/'local-structured-v3-collection-20261027T054655Z/polus/polus-batch-final/outputs'
 for path in sorted(oldroot.glob('*greedy_reachable_v3-p101/attempt-*/completed.json')):baselines.append(process(path,False))
 old_verification=read(REPO/'research/experiments/local_structured_v3_sprint_20261027/results/verification.json')
 old_anchors={r['run_id']:r for r in old_verification['runs']}
 for baseline in baselines:
  folder=artifacts[baseline['run_id']]['folder'];anchor=old_anchors[baseline['run_id']]
  assert sha(folder/'completed.json')==anchor['completed_sha256']
  assert sha(folder/'pruned.pt')==anchor['checkpoint_sha256']
 assert len(rows)==92 and len(baselines)==20 and len(episode_rows)==13200 and len(snapshot_rows)==300
 # Validate the first100 training episodes of B against archived100-episode Q.
 prefixes=[]
 for base in baselines:
  if base['variant']!='q_reachable_v3':continue
  b=next(r for r in rows if r['model']==base['model'] and r['target']==base['target'] and r['policy_seed']==base['policy_seed'] and r['variant']=='q_v4_300')
  original=artifacts[base['run_id']];new=artifacts[b['run_id']]
  prefix=[e for e in new['trace']['training'] if e['episode']<100]
  # Record exact archived observations, not an invented100-episode deployment.
  same_trace=prefix==original['trace']['training']
  snapshot=read(new['folder']/'policy_snapshot_100.json');a=normalize_q(snapshot['q_table']);z=normalize_q(original['q'])
  keys_equal=set(a)==set(z) and all(set(a[s])==set(z[s]) for s in a)
  delta=max((abs(v-z[s][k]) for s,row in a.items() for k,v in row.items()),default=0.) if keys_equal else None
  prefixes.append(dict(model=base['model'],target=base['target'],policy_seed=base['policy_seed'],baseline_platform=base['platform'],
   baseline_run_id=base['run_id'],v4_run_id=b['run_id'],same_first100_training_trace=same_trace,same_q_keys=keys_equal,max_abs_q_difference=delta))
 comparisons=[]
 def compare(treatment,control,scope,contrast):
  same_platform=treatment['platform']==control['platform']
  return dict(contrast=contrast,scope=scope,model=treatment['model'],target=treatment['target'],policy_seed=treatment['policy_seed'],
    treatment=treatment['variant'],control=control['variant'],treatment_run_id=treatment['run_id'],control_run_id=control['run_id'],
    treatment_platform=treatment['platform'],control_platform=control['platform'],
    accuracy_delta_pp=100*(treatment['validation_accuracy']-control['validation_accuracy']),f1_delta_pp=100*(treatment['validation_macro_f1']-control['validation_macro_f1']),
    compression_delta_pp=100*(treatment['achieved_sparsity']-control['achieved_sparsity']),same_widths=(treatment['width0'],treatment['width1'])==(control['width0'],control['width1']),
    same_retained_ids=treatment['retained_ids']==control['retained_ids'],search_ratio=treatment['search_seconds']/control['search_seconds'] if same_platform else None,
    evaluations_ratio=treatment['validation_evaluations']/control['validation_evaluations'],
    coverage_delta=treatment['q_coverage']-control['q_coverage'] if treatment['q_coverage'] is not None and control['q_coverage'] is not None else None)
 for r in rows:
  if not r['variant'].startswith('q_'):continue
  if r['variant']=='q_v4_decay300':
   control=next(b for b in rows if b['variant']=='q_v4_300' and all(b[k]==r[k] for k in ('model','target','policy_seed')))
   comparisons.append(compare(r,control,'POLUS_matched_seed','C_vs_B'))
  else:
   platform='polus' if r['policy_seed']==101 else 'local'
   control=next(b for b in baselines if b['variant']=='q_reachable_v3' and b['platform']==platform and all(b[k]==r[k] for k in ('model','target','policy_seed')))
   comparisons.append(compare(r,control,'POLUS_matched_seed' if platform=='polus' else 'cross_platform_reference_only','single_factor_vs_V3'))
  greedy_variant='greedy_v4_menu12' if r['variant']=='q_v4_menu12' else 'greedy_v4_percent' if r['variant']=='q_v4_percent' else 'greedy_reachable_v3'
  candidates=rows if greedy_variant.startswith('greedy_v4') else baselines
  greedy=next(b for b in candidates if b['platform']=='polus' and b['model']==r['model'] and b['target']==r['target'] and b['variant']==greedy_variant)
  scope='same_menu_reward_Greedy' if r['variant']!='q_v4_incremental_drop' else 'standard_menu_Greedy_reward_mismatch_descriptive'
  comparisons.append(compare(r,greedy,scope,'Q_vs_Greedy'))
 blocks=[]
 for runid,items in __import__('itertools').groupby(sorted(episode_rows,key=lambda e:(e['run_id'],e['episode'])),key=lambda e:e['run_id']):
  items=list(items)
  for start in range(0,len(items),25):
   chunk=items[start:start+25];first=chunk[0]
   block={k:first[k] for k in ('run_id','model','target','variant','policy_seed')};block.update(episode_start=start,episode_end=start+len(chunk)-1,n=len(chunk))
   for metric in ('validation_accuracy','return_value','achieved_sparsity','steps','unseen_choices','exploratory_choices'):
    block[metric+'_mean']=statistics.mean(e[metric] for e in chunk)
   blocks.append(block)
 # Retained-ID equality implies bit-identical projected tensors, already checked above.
 bridge=[]
 for r in baselines:
  if r['platform']=='polus' and r['variant']=='q_reachable_v3':
   local=next(b for b in baselines if b['platform']=='local' and all(b[k]==r[k] for k in ('model','target','policy_seed','variant')))
   bridge.append(compare(r,local,'V3_seed101_platform_bridge','V3_platform_bridge'))
 for path,digest in originals.items():assert sha(path)==digest
 csv_write(out/'runs.csv',rows);save(out/'runs.json',rows);csv_write(out/'summary.csv',summarize(rows));save(out/'summary.json',summarize(rows))
 csv_write(out/'baseline_runs.csv',baselines);save(out/'baseline_runs.json',baselines);csv_write(out/'baseline_summary.csv',summarize(baselines))
 csv_write(out/'comparisons.csv',comparisons);save(out/'comparisons.json',comparisons)
 csv_write(out/'learning_blocks.csv',blocks);csv_write(out/'snapshots.csv',snapshot_rows);csv_write(out/'deployment_decisions.csv',decision_rows)
 csv_write(out/'prefix_checks.csv',prefixes);save(out/'prefix_checks.json',prefixes);csv_write(out/'V3_platform_bridge.csv',bridge)
 csv_write(out/'episodes.csv',episode_rows)
 verification=dict(status='PASS',source_commit=SOURCE,verified_primary_runs=92,verified_reference_runs=20,
    primary_training_episodes=13200,primary_transitions=sum(a['transitions'] for a in audits if a['primary']),
    primary_q_entries=sum(a['q_entries'] for a in audits if a['primary']),primary_snapshots_verified=300,
    primary_magnitude_checks=sum(a['magnitude_checks'] for a in audits if a['primary']),
    source_inventory_files_unchanged=len(originals),archive_sha256=receipt['archive_sha256'],runs=audits,
    verification_runtime=dict(torch=str(torch.__version__),threads=2,platform='Local CPU; archived POLUS metrics retained, final forwards independently agree.'),
    limits=['Intermediate validation observations are replayed, not independently reevaluated.','No new policy rollout or timing measurement.',
       'Greedy counterfactual candidate outputs were not recorded; chosen legality/reward/projection/metrics/evaluation counts checked, candidate optimum not rerun.',
       'Coverage denominator is legal cells in visited aggregated states, not full state/mask space.'])
 save(out/'verification.json',verification)
 save(out/'provenance.json',dict(archive=str(collection/'v4-primary-results.tar.gz'),archive_sha256=receipt['archive_sha256'],
   inventory_sha256=receipt['inventory_sha256'],source_commit=SOURCE,analysis_script_sha256=sha(__file__),
   baseline_reference_index_sha256=sha(EVIDENCE/'reused_baselines.json'),input_manifest_sha256=sha(inputs_path),
   episode_rows_external='episodes.csv retained in external processed directory; compact25-episode blocks tracked.',
   inputs=read(inputs_path),scope='Offline validation-only audit/analysis; no training, pruning search, fine-tuning, new experiment or remote execution.'))
 print(json.dumps({k:v for k,v in verification.items() if k not in ('runs','limits','verification_runtime')},indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();analyze(a.collection,a.inputs,a.output)
