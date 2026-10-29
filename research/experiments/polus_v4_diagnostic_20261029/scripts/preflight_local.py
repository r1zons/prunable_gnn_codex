"""Bounded real-Cora gate; no full scientific condition is rerun."""
from pathlib import Path
import argparse, importlib.util, json, math, hashlib
import torch
from gnn_pruning.experiments.v4_campaign import run_task, compatibility, check_source


def gate(root,inputs,source):
    """36 two-episode/control tasks, independent count/trace/Bellman arithmetic."""
    check_source(source);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('prepare_v4',Path(__file__).with_name('prepare_campaign.py'))
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    prepared=prepare.prepare(root,inputs,source,stage='smoke',threads=2)
    reference=compatibility(inputs);checks=[]
    manifest=json.loads(Path(prepared['manifests']['all']).read_text())
    for item in manifest['tasks']:
        cfg=json.loads(Path(item['config']).read_text());done=run_task(cfg);out=Path(done['output'])
        report=json.loads((out/'metrics.json').read_text());trace=json.loads((out/'trace.json').read_text());decisions=json.loads((out/'decisions.json').read_text())
        a,b=report['hidden_widths'];factor=2 if cfg['model']=='graphsage' else 1
        independent=factor*(1433*a+a*b+7*b)+a+b+7
        assert report['parameters_after']==independent and abs(report['achieved_sparsity']-cfg['target'])<=.01
        for file,digest in done['hashes'].items():assert hashlib.sha256((out/file).read_bytes()).hexdigest()==digest
        for event in [*trace['training'],*trace['deployment']]:
            if event.get('event')=='reset':assert event['widths']==[32,32];continue
            assert min(event['widths_after'])>=4
            action,layer,k=event['action'].split(':');layer,k=int(layer),int(k)
            assert action=='prune' and event['widths_before'][layer]-event['widths_after'][layer]==k
            width=event['widths_before'][layer]
            menu=cfg['plan']['menu']
            allowed=(1,2,4,8) if menu=='standard' else (1,2) if menu=='one_two' else {math.ceil(.05*width),math.ceil(.1*width)}
            assert k in allowed and event['successful_deletions']<=20
            penalty=-2*max(0,abs(event['sparsity']-cfg['target'])-.01) if event['terminal_reason'] else 0.
            assert math.isclose(event['terminal_penalty'],penalty,abs_tol=1e-12)
        for d in decisions:
            if d['phase']=='training':
                expected=d['q_before']+.1*(d['reward']+cfg['plan']['gamma']*d['bootstrap']-d['q_before'])
                assert math.isclose(d['q_after'],expected,rel_tol=1e-12,abs_tol=1e-12)
                if d['terminal']:assert d['bootstrap']==0
        for sequence in [trace['deployment'],*[ [e for e in trace['training'] if e.get('episode')==i and 'action' in e] for i in range(cfg['episodes'])]]:
            if sequence:assert sum(e['terminal_reason'] is not None for e in sequence)==1
        checks.append(dict(run_id=cfg['run_id'],status='PASS',model=cfg['model'],target=cfg['target'],variant=cfg['variant'],
            search_seconds=report['search_seconds'],source_commit=source,physical_reload=True,
            recorded_file_hashes=len(done['hashes']),validation_evaluations=report['statistics']['validation_evaluations']))
    result=dict(status='PASS',source_commit=source,real_smoke_tasks=len(checks),reference=reference,checks=checks)
    (Path(root)/'gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='PASS',real_smoke_tasks=len(checks),reference_comparisons=len(reference['checks']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();gate(a.root,a.inputs,a.source)
