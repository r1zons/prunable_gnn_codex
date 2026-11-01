"""Native technical correctness/resource gate; no accuracy-based acceptance."""
import argparse
import json
import math
from pathlib import Path
import time

from gnn_pruning.experiments.flickr_dense import digest, sha, write
from gnn_pruning.experiments.flickr_pruning import ARCHITECTURES, matrix


def capacity(root,protocol):
    smoke={};failure=[];estimates=[]
    for arch in ARCHITECTURES:
        rows=json.loads((root/('smoke-'+arch+'.json')).read_text())
        if len(rows)!=7:raise ValueError('All7 native technical methods required for every architecture.')
        for entry in rows:
            c=entry['condition'];directory=root/'smoke'/c['id']
            for name,expected in json.loads((directory/'completed.json').read_text()).items():
                if sha(directory/name)!=expected:raise ValueError('Technical output hash mismatch.')
            r=json.loads((directory/'report.json').read_text())
            if not r['technical'] or not r['strict_reload'] or not r['retained_projection'] or r['test_labels_accessed']:
                raise ValueError('Native integrity/isolation failed.')
            if r['source_commit']!=protocol['source_commit'] or r['threads']!=4:
                raise ValueError('Native source/thread identity differs.')
            smoke[arch,c['method']]=r
    for condition in matrix(protocol):
        r=smoke[condition['architecture'],condition['method']];method=condition['method']
        observed=r['search_seconds']
        if method.startswith('static_'):ratio=1.
        elif method=='tabular_q':ratio=protocol['q_learning']['episodes']+1
        else:
            # Prefix observed progress, not an accuracy-based method/architecture choice.
            # Plan for the harder50% target for both targets, with1.75 time safety below.
            actions=len(json.loads((root/'smoke'/r['condition']['id']/'deployment.json').read_text()))
            gain=r['achieved_compression']/max(1,actions)
            expected_steps=min(r['statistics']['horizon'],math.ceil(.5/max(gain,1e-12)))
            ratio=max(1.,expected_steps/max(1,actions))
        bound_to_c4=('l4-h256' in condition['architecture'] or method in ('lookahead_depth2','tabular_q'))
        host_margin=1. if bound_to_c4 else 1.75
        safe=observed*ratio*1.75*host_margin+180.
        memory=max(8192,math.ceil((r['peak_process_rss_bytes']*3+2*1024**3)/1024**2))
        reason=[]
        if safe>170*60:reason.append('measured safe projection exceeds170min/180min queue limit')
        if memory>32768:reason.append('native RSS safety projection exceeds32768MB request')
        if reason:failure.append(dict(condition=condition['id'],reasons=reason))
        estimates.append(dict(condition=condition['id'],actual_profile_host=r['actual_host'],
            observed_search_seconds=observed,projection_multiplier=ratio,safe_search_seconds=safe,
            unbound_host_margin=host_margin,
            required_memory_mb=memory,validation_evaluations=r['observed_validation_evaluations']))
    return dict(status='FAIL' if failure else 'PASS',source_commit=protocol['source_commit'],protocol_digest=digest(protocol),
                checked_unix=time.time(),failures=failure,conditions=estimates,
                accuracy_used_for_gate=False,forecast_not_hard_runtime_guarantee=True,
                rule='harder50% prefix progress or31x one-episode Q cost;1.75x+180s;170min threshold')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();protocol=json.loads((a.root/'protocol.json').read_text())
    gate=capacity(a.root,protocol);write(a.root/'capacity.json',gate)
    print(json.dumps(gate,indent=2))
    if gate['status']!='PASS':raise SystemExit(2)
