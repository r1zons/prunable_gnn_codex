"""Autonomous release remains fail closed; no scientific workload in these tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from gnn_pruning.experiments import v5_campaign_control as c, v5_campaign as engine
from gnn_pruning.experiments.flickr_dense import digest, sha, write


def fixture(tmp_path):
    root=tmp_path/'root';root.mkdir();source=tmp_path/'source';source.mkdir();task=tmp_path/'task';task.mkdir()
    (task/'activate-polus.sh').write_text('verified');analysis=tmp_path/'analysis.py';analysis.write_text('# reporting only')
    p=engine.protocol();p['source_commit']='a'*40;p['dense']={r['dense_id']:{} for r in engine.matrix(p)}
    write(root/'protocol.json',p)
    write(root/'acceptance-preflight.json',[dict(accepted=True,job_id=j) for j in (11,12)])
    tests=dict(status='PASS',failed=0,passed=846,source_commit=p['source_commit'],
               controller_sha256=sha(Path(c.__file__)),analysis_sha256=sha(analysis))
    write(root/'tests.json',tests)
    return root,source,task,analysis,p


def test_scheduler_success_requires_actual_done(tmp_path):
    def ok(*a,**k):return SimpleNamespace(returncode=0,stdout='Done successfully.',stderr='')
    c.assert_scheduler_done([11,12],tmp_path,ok)
    def bad(*a,**k):return SimpleNamespace(returncode=0,stdout='Job is submitted',stderr='')
    with pytest.raises(RuntimeError):c.assert_scheduler_done([13],tmp_path,bad)


@pytest.mark.parametrize('stdout,code,accepted',[
    ('Job <42> is submitted to queue <normal>.\n',0,True),
    ('Job <42> is submitted to queue <normal>.\n',1,False),
    ('connection lost',0,False)])
def test_control_receipt_dependency_single_attempt(tmp_path,stdout,code,accepted):
    root,source,task,analysis,p=fixture(tmp_path);calls=[]
    def runner(args,**kwargs):
        assert (root/'submissions/control-release/intent.json').exists()
        assert args[args.index('-w')+1]=='ended(11) && ended(12)'
        script=kwargs['input'];assert 'sha256sum' in script and '--release' in script
        assert 'OMP_NUM_THREADS=1' in script
        calls.append(args);return SimpleNamespace(stdout=stdout,stderr='',returncode=code)
    if accepted:
        result=c.submit_control(root,source,task,'pin',Path(c.__file__),analysis,'release',[11,12],runner)
        assert result['accepted'] and result['job_id']==42
    else:
        with pytest.raises(RuntimeError):c.submit_control(root,source,task,'pin',Path(c.__file__),analysis,'release',[11,12],runner)
    with pytest.raises(FileExistsError):c.submit_control(root,source,task,'pin',Path(c.__file__),analysis,'release',[11,12],runner)
    assert len(calls)==1


def mocks(monkeypatch):
    monkeypatch.setattr(c,'assert_scheduler_done',lambda *a:None)
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='Open:Active 180.0 min',stderr=''))


def test_failed_native_gate_never_submits_production(tmp_path,monkeypatch):
    from gnn_pruning.experiments import v5_campaign_audit as audit
    root,source,task,analysis,p=fixture(tmp_path);mocks(monkeypatch)
    def fail(*a):raise AssertionError('native correctness failure')
    monkeypatch.setattr(audit,'capacity',fail)
    monkeypatch.setattr(c,'submit',lambda *a:pytest.fail('production bsub must remain inaccessible'))
    with pytest.raises(AssertionError):c.release(root,source,task,'pin',analysis)
    assert json.loads((root/'release/blocked.json').read_text())['status']=='BLOCKED'
    assert not (root/'capacity.json').exists() and not (root/'acceptance-research.json').exists()


def test_successful_gate_releases_all252_before_dependent_audit(tmp_path,monkeypatch):
    from gnn_pruning.experiments import v5_campaign_audit as audit
    root,source,task,analysis,p=fixture(tmp_path);mocks(monkeypatch)
    gate=dict(status='PASS',source_commit=p['source_commit'],protocol_sha256=digest(p),checked_unix=time.time(),
              dense_verified=18,technical_verified=168,threads=1,
              conditions={r['id']:dict(safe_worker_seconds=10,memory_mb=1024) for r in engine.matrix(p)})
    monkeypatch.setattr(audit,'capacity',lambda *a:gate)
    calls=[]
    def submit(jobs,*a):
        assert (root/'capacity.json').exists()
        assert len(json.loads((root/'resource-manifest.json').read_text()))==252
        assert len(jobs)==18 and sum(len(j['conditions']) for j in jobs)==252
        calls.append('production');return [dict(accepted=True,job_id=i+100) for i in range(18)]
    def control(*args):
        assert calls==['production'] and args[-2]=='verify' and args[-1]==list(range(100,118))
        calls.append('verify');return dict(accepted=True,job_id=200)
    monkeypatch.setattr(c,'submit',submit);monkeypatch.setattr(c,'submit_control',control)
    c.release(root,source,task,'pin',analysis)
    assert calls==['production','verify']
    assert json.loads((root/'release/completed.json').read_text())['conditions']==252


def test_partial_matrix_not_analyzable(tmp_path):
    from gnn_pruning.experiments.v5_campaign_analysis import analysis
    root=tmp_path/'root';root.mkdir();write(root/'protocol.json',engine.protocol())
    evidence=tmp_path/'AUDIT.json';write(evidence,dict(status='PASS',verified=251,expected=252,runs=[]))
    with pytest.raises(AssertionError):analysis(root,evidence,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_collection_rejects_unconfirmed_completion_and_unsafe_inventory(tmp_path):
    from gnn_pruning.experiments.v5_campaign_collect import collect
    access=dict(login='authorized@host',root='/new-v5-root')
    def pending(*args,**kwargs):return SimpleNamespace(returncode=1,stdout='',stderr='Required LSF job not DONE')
    with pytest.raises(RuntimeError):collect(access,tmp_path/'pending',runner=pending)
    def unsafe(*args,**kwargs):
        return SimpleNamespace(returncode=0,stdout=json.dumps([dict(path='../historical',bytes=0,sha256='x')]),stderr='')
    with pytest.raises(ValueError):collect(access,tmp_path/'unsafe',runner=unsafe)


def test_collection_verifies_transferred_bytes(tmp_path):
    import hashlib
    from gnn_pruning.experiments.v5_campaign_collect import collect
    access=dict(login='authorized@host',root='/new-v5-root');calls=[]
    def invoke(args,**kwargs):
        calls.append(args)
        if len(calls)==1:
            rows=[dict(path='report.json',bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())]
            return SimpleNamespace(returncode=0,stdout=json.dumps(rows),stderr='')
        Path(args[-1]).joinpath('report.json').write_bytes(b'{}')
        return SimpleNamespace(returncode=0)
    result=collect(access,tmp_path/'verified',runner=invoke)
    assert result['status']=='PASS' and result['files']==1
    assert '--checksum' in calls[1] and not any('--delete' in a for a in calls[1])
