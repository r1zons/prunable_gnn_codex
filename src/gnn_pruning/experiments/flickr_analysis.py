"""Post-completion Flickr validation analysis; no training or test-label loading."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from gnn_pruning.config import load_yaml
from .flickr_dense import digest, matrix, sha, verify_run, write


def analyze(root: Path, protocol: dict, scheduler: dict, output: Path, figures: bool = False) -> dict:
    """Require all32 verified Dense results and DONE evidence before comparisons."""
    output.mkdir(parents=True,exist_ok=False)
    conditions=matrix(protocol);rows=[];missing=[];invalid=[]
    for condition in conditions:
        path=root/'runs'/condition['id']
        if not (path/'completed.json').is_file():
            missing.append(condition['id']);continue
        try:
            result=verify_run(path,condition,root/'inputs')
            config=json.loads((path/'resolved_config.json').read_text())
            if result['source_commit']!=protocol['source_commit'] or result['protocol_sha256']!=digest(protocol):
                raise ValueError('Source/protocol differs from campaign.')
            if config['training']!=protocol['training']:
                raise ValueError('Research training settings changed.')
            if scheduler.get(condition['id'],{}).get('status')!='DONE':
                raise ValueError('Verified scheduler DONE evidence required.')
            if str(scheduler[condition['id']]['job_id'])!=str(json.loads((path/'environment.json').read_text())['lsf']['LSB_JOBID']):
                raise ValueError('Execution job ID differs from acceptance/history.')
            row=dict(**condition,actual_host=result['actual_host'],parameters=result['parameters'],
                     training_wall_seconds=result['training_wall_seconds'],peak_process_rss_bytes=result['peak_process_rss_bytes'],
                     epoch_seconds_median=result['epoch_seconds_median'],checkpoint_bytes=result['checkpoint_bytes'],
                     best_epoch=result['convergence']['best_epoch_1based'],epochs_ran=result['convergence']['total_epochs'],
                     stopping_reason=result['convergence']['stopped_by'],
                     best_in_last_50_epochs=result['convergence']['best_in_last_50_epochs'],
                     large_early_loss_excursion=result['convergence']['large_early_loss_excursion'],
                     source_commit=result['source_commit'],checkpoint_sha256=result['checkpoint_sha256'],
                     tensor_sha256=result['tensor_sha256'],split_sha256=result['split_sha256'],config_sha256=result['config_sha256'],
                     job_id=scheduler[condition['id']]['job_id'])
            for subset in ('train','val'):
                for metric in ('accuracy','macro_f1','loss'):row[subset+'_'+metric]=result['metrics'][subset][metric]
            rows.append(row)
        except (ValueError,KeyError,OSError) as exc:
            invalid.append(dict(condition=condition['id'],error=str(exc)))
    completeness=dict(expected=32,verified=len(rows),missing=missing,invalid=invalid,
                      source_commit=protocol['source_commit'],test_metrics_used=False)
    write(output/'COMPLETENESS.json',completeness)
    if missing or invalid:
        raise ValueError('Incomplete/invalid campaign: preserve COMPLETENESS.json; no scientific ranking produced.')
    with (output/'RESULTS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    frontier=[]
    for model in ('gcn','graphsage'):
        group=[r for r in rows if r['model']==model]
        for row in group:
            if not any(p['val_accuracy']>=row['val_accuracy'] and p['val_macro_f1']>=row['val_macro_f1'] and p['parameters']<=row['parameters']
                       and (p['val_accuracy']>row['val_accuracy'] or p['val_macro_f1']>row['val_macro_f1'] or p['parameters']<row['parameters']) for p in group):
                frontier.append(row['id'])
    finalists={m:[r['id'] for r in sorted([r for r in rows if r['model']==m],key=lambda r:(-r['val_macro_f1'],-r['val_accuracy'],r['parameters']))[:2]]
               for m in ('gcn','graphsage')}
    write(output/'FINALIST_RECOMMENDATIONS.json',dict(rule='validation macro-F1, validation accuracy, then fewer parameters; recommend at most2/model',
          candidates=finalists,quality_size_frontier=frontier,additional_seeds_launched=False,selection_finalized=False,test_evaluated=False))
    lines=['# Flickr Dense sweep: validation-only analysis','',
           '32/32 output inventories/checkpoints/splits and scheduler DONE identities verified.',
           'Single seed101; architecture exploration is not independent-seed confirmation.',
           'Test labels were absent from training inputs; no held-out test outcome is used.',
           'Recommendation is provisional: finalize architecture selection before any later test evaluation.','',
           '| Model | Layers | Width | Validation accuracy | Validation macro-F1 | Epochs | Stop | Actual host | Train seconds |',
           '|---|---:|---:|---:|---:|---:|---|---|---:|']
    for r in rows:
        lines.append(f"|{r['model']}|{r['layers']}|{r['width']}|{r['val_accuracy']:.6f}|{r['val_macro_f1']:.6f}|{r['epochs_ran']}|{r['stopping_reason']}|{r['actual_host']}|{r['training_wall_seconds']:.2f}|")
    lines += ['','Training cost comparisons must be conditioned on actual execution host/threads.',
              'No global cross-host architecture speed ranking or inference acceleration claim is made.',
              'Max-epoch stops, latest-best checkpoints, loss excursions and train/validation gaps require curve review.',
              'Finishing patience is not proof of generalization or global optimization.',
              'Depth2/4/5 finalists are not yet compatible with the unchanged three-layer V2 pruning controllers.',
              'The at-most-two/model recommendations require seeds202/303 confirmation; no such jobs are started.']
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    if figures:plot(root,rows,output/'FIGURES')
    write(output/'ANALYSIS_HASHES.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return completeness


def plot(root: Path, rows: list[dict], output: Path) -> None:
    """Draw saved curves and quality/size/cost; cost panels separate actual hosts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir()
    for model in ('gcn','graphsage'):
        fig,axes=plt.subplots(4,4,figsize=(16,11),squeeze=False)
        for row in [r for r in rows if r['model']==model]:
            history=[json.loads(line) for line in (root/'runs'/row['id']/'history.jsonl').read_text().splitlines()]
            ax=axes[[2,3,4,5].index(row['layers']),[32,64,128,256].index(row['width'])]
            for metric in ('train_loss','val_loss'):ax.plot([r['epoch'] for r in history],[r[metric] for r in history],label=metric,lw=.9)
            ax.axvline(row['best_epoch'],color='grey',ls=':',lw=.7);ax.set_title(f"L{row['layers']} H{row['width']}")
            ax.set_xlabel('Epoch');ax.set_ylabel('Cross-entropy');ax.grid(alpha=.2)
        axes[0,0].legend();fig.suptitle(model+' saved training/validation curves');fig.tight_layout()
        fig.savefig(output/(model+'_learning_curves.pdf'));fig.savefig(output/(model+'_learning_curves.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,model in enumerate(('gcn','graphsage')):
        for j,metric in enumerate(('val_accuracy','val_macro_f1')):
            ax=axes[i,j]
            for depth in (2,3,4,5):
                group=[r for r in rows if r['model']==model and r['layers']==depth]
                ax.plot([r['parameters'] for r in group],[r[metric] for r in group],'o-',label=f'{depth} layers')
            ax.set_xscale('log');ax.set_xlabel('Physical parameters');ax.set_ylabel(metric);ax.set_title(model);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'quality_model_size.pdf');fig.savefig(output/'quality_model_size.png',dpi=180);plt.close(fig)
    hosts=sorted({r['actual_host'] for r in rows});fig,axes=plt.subplots(1,len(hosts),figsize=(6*len(hosts),4),squeeze=False)
    for ax,host in zip(axes.flat,hosts):
        for model in ('gcn','graphsage'):
            group=[r for r in rows if r['actual_host']==host and r['model']==model]
            ax.scatter([r['training_wall_seconds'] for r in group],[r['val_macro_f1'] for r in group],label=model)
            for r in group:ax.annotate(f"{r['layers']}/{r['width']}",(r['training_wall_seconds'],r['val_macro_f1']),fontsize=6)
        ax.set_xlabel('Training seconds, same execution host');ax.set_ylabel('Validation macro-F1');ax.set_title(host);ax.legend()
    fig.tight_layout();fig.savefig(output/'quality_cost_by_host.pdf');fig.savefig(output/'quality_cost_by_host.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','scheduler','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--figures',action='store_true');args=parser.parse_args()
    analyze(args.root,load_yaml(args.protocol),json.loads(args.scheduler.read_text()),args.output,args.figures)
