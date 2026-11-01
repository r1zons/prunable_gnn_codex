"""Resource-only adapter contracts; real tiny GNNs verify observer transparency."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gnn_pruning.experiments import flickr_thread_scaling as b, flickr_thread_scaling_lsf as l
from gnn_pruning.experiments.flickr_dense import tensors_hash, write
from test_budget_v5 import graph, network


def test_physical_cores_not_smt_and_affinity_shortfall():
    rows=[dict(cpu=core*8+sibling,socket=str(core//10),core=str(core),numa='node'+str(core//10))
          for core in range(20) for sibling in range(8)]
    cpus=b.physical_affinity(rows)
    assert len(cpus)==20 and len(set(cpus))==20
    assert all(x%8==0 for x in cpus) and cpus[:4]==[0,80,8,88]
    with pytest.raises(ValueError,match='physical cores'):b.physical_affinity(rows[:12*8])


def test_forecast_keeps_original_limits_safety_and_episode_budget():
    f=b.forecast(630.4851435758173)
    assert f['safe_minutes']==pytest.approx(193.0212169)
    assert not f['passes_acceptance'] and not f['passes_absolute_limit']
    assert b.forecast(540)['passes_acceptance']
    with pytest.raises(ValueError):b.forecast(float('nan'))


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('target',[.3,.5])
def test_real_gnn_instrumentation_preserves_every_decision_reward_q_mask_and_rng(kind,target):
    torch.set_num_threads(1)
    m=network(kind,4);data=graph();data.y[8:]=-1
    split=dict(train=list(range(4,8)),val=list(range(4)),test=list(range(8,12)))
    p=dict(q_learning=dict(episodes=30,epsilon=.2,alpha=.1,gamma=.9),technical_q_episodes=3,
           policy_seed=101,budget=dict(tolerance=.01,minimum=4))
    manifest=dict(training_tensor_sha256='masked-graph',split_sha256='val-only')
    rng=torch.get_rng_state().clone();before=tensors_hash(m.state_dict())
    plain,pc,pt=b.q_profile(m,data,split,manifest,p,target,instrument=False)
    observed,oc,ot=b.q_profile(m,data,split,manifest,p,target)
    assert plain.training_trace==observed.training_trace and plain.trace==observed.trace
    assert plain.decisions==observed.decisions and plain.q_table==observed.q_table
    assert tensors_hash(plain.model.state_dict())==tensors_hash(observed.model.state_dict())
    assert pc.events==oc.events and oc.requests==oc.forward_calls+sum(e['reused'] for e in oc.events)
    assert len(ot['episodes'])==3 and [x['episode'] for x in ot['episodes']]==[0,1,2]
    assert ot['observed_components_seconds']['q_update']>=0
    assert ot['component_calls']['q_update']==len(observed.training_trace)
    assert ot['evaluation_seconds']>=ot['actual_forward_seconds']
    assert p['q_learning']['episodes']==30 and not (data.y[8:]!=-1).any()
    assert torch.equal(rng,torch.get_rng_state()) and tensors_hash(m.state_dict())==before


def test_forward_profiles_are_real_uncached_warmup_and_repeats():
    model=network('graphsage',4);data=graph();calls=[]
    hook=model.register_forward_hook(lambda *args:calls.append(1))
    result=b.forward_profile(model,data,warmup=2,repeats=3);hook.remove()
    assert len(calls)==result['actual_forward_calls']==5
    assert not result['cache_used'] and len(result['samples_seconds'])==3
    assert result['median_seconds']>0 and result['stdev_seconds']>=0


def config():
    return dict(thread_counts=[4,8,12,20],targets=[.3,.5],hosts=dict(c2='c2',c3='c3',c4='c4'),
                host='c4',benchmark_source_commit='a'*40)


def test_lsf_all_twenty_slots_single_host_exclusive_no_research_matrix():
    jobs=l.jobs(config());assert len(jobs)==3
    job=jobs[-1];cmd=l.command(job,Path('/new'))
    assert cmd[cmd.index('-n')+1]=='20' and '-x' in cmd and 'span[hosts=1]' in cmd
    assert cmd[cmd.index('-m')+1]=='c4' and cmd[cmd.index('-W')+1]=='175'
    with pytest.raises(ValueError):l.command(dict(job,slots=4),Path('/new'))
    env=b.thread_environment(12,list(range(20)))
    assert env['OMP_NUM_THREADS']==env['OPENBLAS_NUM_THREADS']==env['MKL_NUM_THREADS']=='12'
    assert env['OMP_MAX_ACTIVE_LEVELS']=='1' and env['CUDA_VISIBLE_DEVICES']==''


@pytest.mark.parametrize('response,code,expected',[
 ('Job <123> is submitted to queue <normal>.\n',0,'ACCEPTED'),
 ('Job <123> is submitted to queue <normal>.\n',1,'UNRESOLVED'),('lost response',0,'UNRESOLVED')])
def test_single_attempt_receipts_and_no_silent_duplicate_submission(tmp_path,response,code,expected):
    source=tmp_path/'source';source.mkdir();task=tmp_path/'task';task.mkdir()
    (task/'activate-polus.sh').write_text('verified')
    write(tmp_path/'source-files.json',{})
    write(tmp_path/'config.json',config());write(tmp_path/'protocol.json',{})
    calls=[]
    def runner(*args,**kwargs):
        calls.append(args)
        text=response.replace('123',str(123+len(calls)))
        return SimpleNamespace(stdout=text,stderr='',returncode=code)
    receipts=l.submit(tmp_path,source,task,tmp_path/'config.json',tmp_path/'protocol.json',tmp_path,
                      l.sha(task/'activate-polus.sh'),runner)
    assert receipts[0]['status']==expected and len(calls)==(3 if expected=='ACCEPTED' else 1)
    with pytest.raises(FileExistsError):l.submit(tmp_path,source,task,tmp_path/'config.json',
        tmp_path/'protocol.json',tmp_path,l.sha(task/'activate-polus.sh'),runner)


def test_serial_fresh_process_matrix_and_correct_module(tmp_path,monkeypatch):
    cfg=config();write(tmp_path/'config.json',cfg)
    monkeypatch.setenv('LSB_DJOB_NUMPROC','20');monkeypatch.setattr(b.socket,'gethostname',lambda:'c4')
    monkeypatch.setattr(b,'capture_topology',lambda p:None)
    monkeypatch.setattr(b,'topology',lambda:dict(cpu_topology=[dict(cpu=i,socket='0',core=str(i),numa='node0') for i in range(20)]))
    calls=[]
    def runner(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(b.subprocess,'run',runner);monkeypatch.setattr(b,'analyze',lambda *args:None)
    b.orchestrate(tmp_path,tmp_path,tmp_path/'protocol.json',tmp_path/'config.json')
    assert len(calls)==8
    assert [(a[a.index('--threads')+1],a[a.index('--target')+1]) for a,k in calls]==[(str(t),str(x)) for t in b.THREADS for x in b.TARGETS]
    assert all(a[a.index('-m')+1]==b.MODULE and k['timeout']==1200 for a,k in calls)


def test_numeric_differences_do_not_compare_unobserved_continuations():
    first=[dict(state=['a'],selected_action='x',reward=.1,q_after=.2),
           dict(state=['b'],selected_action='x',reward=99.)]
    second=[dict(state=['a'],selected_action='x',reward=.11,q_after=.2),
            dict(state=['other-mask'],selected_action='y',reward=-99.)]
    diff=b.decision_differences(first,second)
    assert diff['matched_state_action_prefix']==1 and diff['first_divergence']['index']==1
    assert diff['max_absolute_difference_on_matched_prefix']['reward']==pytest.approx(.01)
    q=b.q_differences([dict(state=['a'],actions={'x':.1}),dict(state=['b'],actions={'y':3.})],
                      [dict(state=['a'],actions={'x':.3}),dict(state=['c'],actions={'y':9.})])
    assert q['matched_entries']==q['reference_only']==q['candidate_only']==1
    assert q['max_absolute_difference']==pytest.approx(.2)
