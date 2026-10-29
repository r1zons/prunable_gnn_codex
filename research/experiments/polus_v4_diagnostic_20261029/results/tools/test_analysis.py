"""Offline audit tests: equations and deliberate corruption, no search or models."""
import copy,importlib.util,math
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('v4_offline',Path(__file__).with_name('verify_and_aggregate.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('menu',['standard','one_two','percent'])
def test_width_oracle_matches_exhaustive_joint_paths(menu):
 for start in ((32,32),(23,18),(16,32),(4,4)):
  frontier={start};seen=set(frontier)
  for horizon in range(5):
   expected=any(abs(100*(47175-(1433*a+a*b+7*b+a+b+7))-30*47175)<=47175 for a,b in seen)
   assert m.reachable('gcn',.3,menu,start,horizon)==expected
   following=set()
   for a,b in frontier:
    for layer,width in enumerate((a,b)):
     sizes=(1,2,4,8) if menu=='standard' else (1,2) if menu=='one_two' else sorted({math.ceil(.05*width),math.ceil(.1*width)})
     for k in sizes:
      after=[a,b];after[layer]-=k
      if min(after)>=4:following.add(tuple(after))
   frontier=following-seen;seen|=following

class NoExploration:
 def __init__(self,seed):pass
 def random(self):return 1.

def fixture():
 cfg=dict(run_id='manual',variant='q_v4_gamma0',model='gcn',target=.5,policy_seed=101,episodes=2)
 accuracy=float(torch.tensor(.8,dtype=torch.float32));events=[];p0=47175
 for step in range(1,17):
  before,after=33-step,32-step;s0=1-(1433*before+before*32+224+before+32+7)/p0
  s1=1-(1433*after+after*32+224+after+32+7)/p0
  events.append(dict(action='prune:0:1',successful_deletions=step,widths_before=[before,32],widths_after=[after,32],
     original_kept_ids=[list(range(after)),list(range(32))],sparsity=s1,signed_target_deviation=s1-.5,
     validation_accuracy=accuracy,accuracy_drop=0.,terminal_reason='target_band' if step==16 else None,
     terminal_penalty=0.,reward=.6*(s1-s0)))
 training=[]
 for i in range(2):training.extend([dict(event='reset',episode=i,widths=[32,32]),*[dict(e,episode=i) for e in events]])
 q=[];slots=0
 for i,e in enumerate(events):
  state=['local_v2',e['widths_before'],20-i,math.floor(accuracy/.01),False]
  q.append(dict(state=state,actions={'prune:0:1':.19*e['reward']}))
  slots+=len(m.legal('gcn',.5,'standard',tuple(state[1]),20-i))
 report=dict(dense_validation={'accuracy':.8},final_validation={'accuracy':.8},original_kept_ids=events[-1]['original_kept_ids'],
  statistics=dict(terminal_reason='target_band',unique_states=17,q_entries=16,legal_state_action_slots=slots,
   q_table_sparsity=1-16/slots,validation_evaluations=51,deployment_unseen_choices=0))
 return cfg,report,dict(training=training,deployment=copy.deepcopy(events)),q

def test_known_gamma0_replay(monkeypatch):
 monkeypatch.setattr(m.random,'Random',NoExploration)
 audit,episodes,deployed,decisions,snapshots=m.audit_learning(*fixture(),None,None)
 assert audit['q_entries']==16 and len(episodes)==2 and deployed['steps']==16 and not snapshots

@pytest.mark.parametrize('field',['reward','q','ids','double_terminal','illegal_action','coverage'])
def test_modified_evidence_rejected(monkeypatch,field):
 monkeypatch.setattr(m.random,'Random',NoExploration)
 cfg,report,trace,q=copy.deepcopy(fixture())
 if field=='reward':trace['training'][1]['reward']+=.1
 elif field=='q':q[0]['actions']['prune:0:1']+=.1
 elif field=='ids':trace['deployment'][0]['original_kept_ids'][0][0]=31
 elif field=='double_terminal':trace['deployment'][0]['terminal_reason']='target_band'
 elif field=='illegal_action':trace['deployment'][0]['action']='prune:0:8'
 else:report['statistics']['legal_state_action_slots']+=1
 with pytest.raises(AssertionError):m.audit_learning(cfg,report,trace,q,None,None)
