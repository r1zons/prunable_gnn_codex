"""Future56-run completeness/audit/analysis; no test access or training."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import torch

from .flickr_dense import metrics, sha, tensors_hash, write
from .flickr_pruning import frozen, matrix, verify_projection
from gnn_pruning.models import build_model


def analyze(root, scheduler, output, figures=False):
    """Require all56 verified results/DONE before ranking or creating figures."""
    output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());conditions=matrix(protocol)
    missing=[c['id'] for c in conditions if not (root/'runs'/c['id']/'completed.json').is_file()]
    write(output/'COMPLETENESS.json',dict(expected=56,missing=missing))
    if missing:raise ValueError('Incomplete matrix: no rankings produced.')
    rows=[];reports={};dense_cache={};torch.set_num_threads(4)
    for c in conditions:
        path=root/'runs'/c['id']
        inventory=json.loads((path/'completed.json').read_text())
        if {p.name for p in path.iterdir()} != set(inventory)|{'completed.json'}:raise ValueError('Unexpected output files.')
        if any(sha(path/name)!=expected for name,expected in inventory.items()):raise ValueError('Output hash mismatch.')
        r=json.loads((path/'report.json').read_text());cp=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        if (r['condition']!=c or r['technical'] or r['source_commit']!=protocol['source_commit']
            or not r['statistics']['complete'] or r['test_labels_accessed']):raise ValueError('Run identity/terminal gate.')
        env=json.loads((path/'environment.json').read_text())
        if scheduler[c['id']]['status']!='DONE' or str(scheduler[c['id']]['job_id'])!=str(env['lsf']['LSB_JOBID']):raise ValueError('Scheduler identity.')
        if c['architecture'] not in dense_cache:dense_cache[c['architecture']]=frozen(root,protocol,c['architecture'])
        dense,data,split,_,origin=dense_cache[c['architecture']]
        if r['dense_checkpoint_sha256']!=origin['checkpoint_sha256'] or cp['original_kept_ids']!=r['retained_original_ids']:raise ValueError('Dense/mask identity.')
        model=build_model(cp['model_name'],**cp['model_config']).eval();model.load_state_dict(cp['model_state_dict'],strict=True)
        verify_projection(dense,model,r['retained_original_ids'])
        if sha(path/'checkpoint.pt')!=r['checkpoint_sha256'] or tensors_hash(model.state_dict())!=r['tensor_sha256']:raise ValueError('Checkpoint hash.')
        repeated=metrics(model,data,split)['val']
        if any(abs(repeated[k]-r['validation'][k])>(5e-5 if k=='loss' else 1e-12) for k in ('accuracy','macro_f1','loss')):raise ValueError('Final metrics mismatch.')
        rows.append(dict(**c,actual_host=r['actual_host'],threads=r['threads'],
            achieved_compression=r['achieved_compression'],outside_band=r['outside_band'],
            parameters_dense=r['parameters_before'],parameters_pruned=r['parameters_after'],
            dense_val_accuracy=r['dense_validation']['accuracy'],dense_val_macro_f1=r['dense_validation']['macro_f1'],
            val_accuracy=r['validation']['accuracy'],val_macro_f1=r['validation']['macro_f1'],
            delta_accuracy=r['delta_accuracy'],delta_macro_f1=r['delta_macro_f1'],
            per_class_f1=json.dumps(r['validation_per_class_f1']),widths=json.dumps(r['final_widths']),
            actions=r['structural_actions'],search_seconds=r['search_seconds'],
            validation_evaluations=r['observed_validation_evaluations'],
            validation_requests=r['validation_evaluation_requests'],
            exact_model_cache_hits=r['statistics']['evaluation_cache_hits'],
            candidate_requests=r['statistics']['candidate_evaluation_requests'],
            candidate_evaluations=r['statistics']['candidate_evaluations'],
            q_entries=r['statistics'].get('q_entries',0),q_coverage=r['statistics'].get('q_table_coverage',0),
            unseen_choices=r['statistics'].get('deployment_unseen_choices',0),
            terminal_reason=r['statistics']['terminal_reason'],peak_rss_bytes=r['peak_process_rss_bytes'],
            dense_checkpoint_sha256=r['dense_checkpoint_sha256'],checkpoint_sha256=r['checkpoint_sha256'],
            tensor_sha256=r['tensor_sha256'],source_commit=r['source_commit']))
        reports[c['id']]=r
    def table(name,data):
        with (output/name).open('x',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    table('RESULTS.csv',rows)
    paired=[]
    for arch in protocol['architectures']:
        for target in protocol['targets']:
            group=[r for r in reports.values() if r['condition']['architecture']==arch and r['condition']['target']==target]
            for a,b in itertools.combinations(group,2):
                aid=a['condition']['id'];bid=b['condition']['id']
                actions=lambda key:[x['action'] for x in json.loads((root/'runs'/key/'deployment.json').read_text())]
                paired.append(dict(architecture=arch,target=target,method_a=a['condition']['method'],method_b=b['condition']['method'],
                   both_in_band=not a['outside_band'] and not b['outside_band'],
                   identical_widths=a['final_widths']==b['final_widths'],
                   identical_masks=a['retained_original_ids']==b['retained_original_ids'],
                   identical_actions=actions(aid)==actions(bid),
                   compression_difference=b['achieved_compression']-a['achieved_compression'],
                   delta_accuracy=b['validation']['accuracy']-a['validation']['accuracy'],
                   delta_f1=b['validation']['macro_f1']-a['validation']['macro_f1'],
                   search_ratio=b['search_seconds']/max(a['search_seconds'],1e-12)))
    table('PAIRED_COMPARISONS.csv',paired)
    (output/'SCIENTIFIC_ANALYSIS.md').write_text('# Flickr structural pruning: verified56-run analysis\n\n'
        'One Dense seed101 and policy seed101, recorded separately. No test labels.\n'
        'All56 identities/outputs/checkpoints/metrics verified. Selection is validation-only.\n'
        'Use RESULTS.csv for accuracy/F1 retention and actual compression; outside-band results remain visible.\n'
        'A band match still permits different physical budgets: compare actual compression before interpreting quality differences.\n'
        'No seed robustness, causal depth/width effect, or universal superiority follows from this campaign.\n')
    lines=['# Method comparisons','','Mask equality, width equality and action equality are different diagnostics.',
           'Static layer application order and adaptive action sequences have different meanings.',
           'All pairs are within the same original checkpoint/target; see PAIRED_COMPARISONS.csv.','',
           '| Architecture | Target | Eligible methods sorted by macro-F1, then accuracy |','|---|---:|---|']
    for arch in protocol['architectures']:
        for target in protocol['targets']:
            group=sorted([r for r in rows if r['architecture']==arch and r['target']==target and not r['outside_band']],
                         key=lambda r:(-r['val_macro_f1'],-r['val_accuracy']))
            lines.append(f"|{arch}|{target}|{', '.join(r['method'] for r in group)}|")
    lines+=['','This descriptive order does not establish superiority at equal exact compression.',
             'Lookahead/Greeedy/Magnitude mask and layer-allocation differences require the retained IDs and widths in raw reports.',
             'Search ratios mix method cost and actual selected path; condition timing on recorded host/threads.']
    (output/'METHOD_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (output/'ARCHITECTURE_COMPARISON.md').write_text('# Architecture comparison limits\n\n'
        'Four selected architectures, one trained checkpoint each, are screening evidence.\n'
        'The matrix is not a balanced model/depth/width factorial. Dense quality, parameter count and pruning budget differ.\n'
        'Inspect per-architecture method cells, validation per-class F1, undershoot/overshoot, and search cost before comparing.\n'
        'Q episodes are policy search, not independent trained replicas; absent multi-seed results cannot support robustness claims.\n')
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        destination=output/'FIGURES';destination.mkdir()
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for ax,arch in zip(axes.flat,protocol['architectures']):
            for method in protocol['methods']:
                group=[r for r in rows if r['architecture']==arch and r['method']==method]
                ax.plot([r['achieved_compression'] for r in group],[r['val_macro_f1'] for r in group],'o-',label=method)
            ax.set_title(arch);ax.set_xlabel('Achieved physical compression');ax.set_ylabel('Validation macro-F1');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.tight_layout()
        fig.savefig(destination/'quality_achieved_compression.png',dpi=180);fig.savefig(destination/'quality_achieved_compression.pdf');plt.close(fig)
    write(output/'HASHES.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','scheduler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--figures',action='store_true');a=p.parse_args()
    analyze(a.root,json.loads(a.scheduler.read_text()),a.output,a.figures)
