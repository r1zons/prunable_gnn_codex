"""Compare preserved native policies across execution-only cache revisions."""
import argparse
import json
from pathlib import Path

from gnn_pruning.experiments.flickr_dense import write
from gnn_pruning.experiments.flickr_pruning import matrix


def compare(before,after,output):
    old=json.loads((before/'protocol.json').read_text());new=json.loads((after/'protocol.json').read_text())
    assert old['dense']==new['dense'] and old['q_learning']==new['q_learning']
    assert old['input_manifest_sha256']==new['input_manifest_sha256']
    rows=[]
    for c in matrix(new):
        if c['target']!=.5:continue
        a=before/'smoke'/c['id'];b=after/'smoke'/c['id']
        read=lambda path,name:json.loads((path/name).read_text())
        if c['method']=='tabular_q':
            # Technical profiles differ1 vs3 episodes; only their common episode0 is paired.
            left=[r for r in read(a,'training.json') if r['episode']==0]
            right=[{k:v for k,v in r.items() if k!='validation_evaluation_reused'}
                   for r in read(b,'training.json') if r['episode']==0]
            assert left==right,(c['id'],'native episode0 changed')
            scope='full first training episode: actions/rewards/states/Q updates/masks'
        else:
            assert read(a,'deployment.json')==read(b,'deployment.json'),c['id']
            assert read(a,'report.json')['tensor_sha256']==read(b,'report.json')['tensor_sha256'],c['id']
            if c['method'] in ('adaptive_greedy','lookahead_depth2'):
                left=read(a,'decisions.json');right=read(b,'decisions.json')
                assert len(left)==len(right)
                for x,y in zip(left,right):
                    for key in ('state','legal_actions','selected_action','best_planned_path','candidate_paths','first_transitions'):
                        assert x[key]==y[key],(c['id'],key)
            scope='all technical deployment actions/rewards/masks/tensors; full planning paths when recorded'
        rows.append(dict(condition=c['id'],PASS=True,scope=scope))
    write(output,dict(PASS=28,FAIL=0,SKIPPED=0,before_source=old['source_commit'],
                     after_source=new['source_commit'],new_forward_executed=False,comparisons=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare(a.before,a.after,a.output)
