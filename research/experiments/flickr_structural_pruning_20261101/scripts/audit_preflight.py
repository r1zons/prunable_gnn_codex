"""Independent native artifact audit; strict reload/projection, no new forwards."""
import argparse
import json
from pathlib import Path

import torch

from gnn_pruning.experiments.flickr_dense import digest, sha, tensors_hash, write
from gnn_pruning.experiments.flickr_pruning import frozen, matrix, verify_projection
from gnn_pruning.models import build_model
from gnn_pruning.pruning.budget_v5 import Budget


def audit(evidence, inputs, output):
    protocol=json.loads((evidence/'protocol.json').read_text())
    rows=[];models={};torch.set_num_threads(4)
    expected=[r for r in matrix(protocol) if r['target']==.5]
    assert len(expected)==28
    for c in expected:
        path=evidence/'smoke'/c['id'];marker=json.loads((path/'completed.json').read_text())
        assert {p.name for p in path.iterdir()}==set(marker)|{'completed.json'}
        assert all(sha(path/name)==value for name,value in marker.items())
        r=json.loads((path/'report.json').read_text());env=json.loads((path/'environment.json').read_text())
        assert r['condition']==c and r['technical'] and r['source_commit']==protocol['source_commit']
        assert r['protocol_sha256']==digest(protocol) and not r['test_labels_accessed']
        receipt=json.loads((evidence/'submissions'/('smoke-'+c['architecture'])/'receipt.json').read_text())
        assert receipt['accepted'] and str(receipt['job_id'])==str(env['lsf']['LSB_JOBID'])
        assert env['torch_threads']==r['threads']==4 and env['host']==r['actual_host']
        if c['architecture'] not in models:
            models[c['architecture']]=frozen(inputs,protocol,c['architecture'])[0]
        dense=models[c['architecture']]
        cp=torch.load(path/'checkpoint.pt',weights_only=True,map_location='cpu')
        assert cp['condition']==c and cp['original_kept_ids']==r['retained_original_ids']
        assert cp['dense_checkpoint_sha256']==protocol['dense'][c['architecture']]['checkpoint_sha256']
        assert cp['source_commit']==protocol['source_commit'] and cp['protocol_sha256']==digest(protocol)
        rebuilt=build_model(cp['model_name'],**cp['model_config']).eval()
        rebuilt.load_state_dict(cp['model_state_dict'],strict=True)
        verify_projection(dense,rebuilt,r['retained_original_ids'])
        assert sha(path/'checkpoint.pt')==r['checkpoint_sha256']
        assert tensors_hash(rebuilt.state_dict())==r['tensor_sha256']
        p0=sum(p.numel() for p in dense.parameters());p1=sum(p.numel() for p in rebuilt.parameters())
        assert p0==r['parameters_before'] and p1==r['parameters_after']<p0
        assert r['achieved_compression']==1-p1/p0
        assert rebuilt.hidden_channel_dims==r['final_widths']
        if (path/'evaluation_cache.json').exists():
            events=json.loads((path/'evaluation_cache.json').read_text())
            assert len(events)==r['validation_evaluation_requests']
            assert sum(not e['reused'] for e in events)==r['observed_validation_evaluations']
            assert sum(e['reused'] for e in events)==r['statistics']['evaluation_cache_hits']
            values={}
            for event in events:
                key=event['tensor_sha256'];value=event['validation_accuracy']
                if event['reused']:assert key in values and values[key]==value
                else:assert key not in values;values[key]=value
        if c['method']=='tabular_q':
            training=json.loads((path/'training.json').read_text());episodes=r['statistics']['episodes']
            assert set(e['episode'] for e in training)==set(range(episodes))
            budget=Budget(dense,.5,**protocol['budget'])
            for i in range(episodes):
                last=[e for e in training if e['episode']==i][-1]
                assert last['terminal_reason']=='target_band' and budget.goal(last['widths_after'])
            assert protocol['q_learning']['episodes']==30
        rows.append(dict(condition=c['id'],job_id=receipt['job_id'],PASS=True,
                 actual_host=r['actual_host'],threads=4,search_seconds=r['search_seconds'],
                 requests=r.get('validation_evaluation_requests',r['observed_validation_evaluations']),
                 actual_validation_forwards=r['observed_validation_evaluations'],
                 cache_hits=r['statistics'].get('evaluation_cache_hits',0),
                 candidate_evaluations=r['statistics']['candidate_evaluations'],
                 peak_process_rss_bytes=r['peak_process_rss_bytes'],
                 complete=r['statistics']['complete'],terminal_reason=r['statistics']['terminal_reason'],
                 technical=True,checkpoint_sha256=r['checkpoint_sha256'],tensor_sha256=r['tensor_sha256']))
    write(output,dict(PASS=28,FAIL=0,SKIPPED=0,source_commit=protocol['source_commit'],
                     protocol_digest=digest(protocol),new_forward_executed=False,runs=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.evidence,a.inputs,a.output)
