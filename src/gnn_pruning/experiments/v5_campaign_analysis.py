"""Post-audit V5 paired validation analysis. Refuses partial or unaudited matrices."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .flickr_dense import sha, write
from .v5_campaign import matrix

PAIRS=(('static_magnitude','adaptive_greedy'),('adaptive_greedy','lookahead_depth2'),
       ('sequential_random','adaptive_greedy'),('tabular_q','adaptive_greedy'),
       ('tabular_q','sequential_random'),('static_snip','static_magnitude'),('static_snip','static_random'),
       ('static_snip','sequential_random'),('static_snip','adaptive_greedy'),
       ('static_snip','lookahead_depth2'),('static_snip','tabular_q'))


def table(path,rows):
    """Create one compact flat table, without replacing earlier analysis."""
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analysis(root,audit_path,output):
    """Summarize252 independently verified final checkpoints; no inference/search."""
    p=json.loads((root/'protocol.json').read_text());audit=json.loads(audit_path.read_text())
    expected={c['id'] for c in matrix(p)}
    assert audit['status']=='PASS' and audit['verified']==audit['expected']==252
    assert len(audit['runs'])==252 and {r['id'] for r in audit['runs']}==expected
    assert all(r['status']=='PASS' for r in audit['runs'])
    output.mkdir(parents=True,exist_ok=False)
    runs=[];by_id={};episodes=[];qdiagnostics=[];trajectory=[]
    for item in audit['runs']:
        r=item['report'];c=r['condition'];assert r['scientific_complete']
        stats=r['statistics'];dense=r['dense_validation'];v=r['validation']
        row=dict(**c,validation_accuracy=v['accuracy'],validation_macro_f1=v['macro_f1'],validation_loss=v['loss'],
            dense_accuracy=dense['accuracy'],dense_macro_f1=dense['macro_f1'],
            delta_accuracy=v['accuracy']-dense['accuracy'],delta_macro_f1=v['macro_f1']-dense['macro_f1'],
            relative_accuracy_change=(v['accuracy']-dense['accuracy'])/dense['accuracy'] if dense['accuracy'] else None,
            relative_f1_change=(v['macro_f1']-dense['macro_f1'])/dense['macro_f1'] if dense['macro_f1'] else None,
            parameters_before=r['parameters_before'],parameters_after=r['parameters_after'],
            achieved_compression=r['achieved_compression'],target_deviation=r['signed_target_deviation'],
            target_miss=r['target_miss'],fallback=r['fallback'],final_widths=json.dumps(r['final_widths']),
            retained_ids_sha256=__import__('hashlib').sha256(json.dumps(r['retained_original_ids']).encode()).hexdigest(),
            search_seconds=r['search_seconds'],search_cpu_seconds=r['search_cpu_seconds'],peak_rss_bytes=r['peak_rss_bytes'],
            evaluation_requests=stats['validation_callback_requests'],actual_forwards=stats['actual_validation_forwards'],
            cache_hits=stats['exact_cache_hits'],candidate_transition_requests=stats['candidate_evaluations'],
            scoring_gradient_forwards=stats.get('scoring_gradient_forwards',0),
            inference_median_seconds=r['inference']['median_seconds'],host=r['actual_host'],threads=r['threads'],
            terminal_reason=stats['terminal_reason'],q_entries=stats.get('q_entries',0),q_coverage=stats.get('q_table_coverage',0),
            unseen_deployment=stats.get('deployment_unseen_choices',0),tied_deployment=stats.get('deployment_tied_maxima',0))
        runs.append(row);by_id[c['id']]=r
        if c['method']=='tabular_q':
            directory=root/'runs'/c['id'];training=json.loads((directory/'training.json').read_text())
            decisions=json.loads((directory/'decisions.json').read_text());grouped=defaultdict(list)
            for entry in training:grouped[entry['episode']].append(entry)
            statekey=lambda state:json.dumps(state,separators=(',',':'))
            visits=[statekey(entry['state']) for entry in training];trainstates=set(visits)
            for episode,entries in sorted(grouped.items()):
                episodes.append(dict(id=c['id'],dataset=c['dataset'],model=c['model'],target=c['target'],seed=c['dense_seed'],
                    episode=episode,steps=len(entries),reward_sum=sum(e['reward'] for e in entries),
                    exploration_steps=sum(e['exploration'] for e in entries),terminal_accuracy=entries[-1]['validation_accuracy'],
                    terminal_compression=entries[-1]['sparsity'],terminal_reason=entries[-1]['terminal_reason']))
            learned_strict=unseen_better_negative=0
            for decision in decisions:
                i=decision['legal_actions'].index(decision['selected_action']);v=decision['q_values'];known=decision['known']
                learned_strict+=bool(known[i] and len(v)>1 and all(v[i]>x for j,x in enumerate(v) if j!=i))
                unseen_better_negative+=bool(not known[i] and any(k and q<0 for q,k in zip(v,known)))
            qdiagnostics.append(dict(id=c['id'],dataset=c['dataset'],model=c['model'],target=c['target'],seed=c['dense_seed'],
                training_visits=len(visits),unique_training_states=len(trainstates),repeated_training_visits=len(visits)-len(trainstates),
                deployment_steps=len(decisions),deployment_states_seen_training=sum(statekey(d['state']) in trainstates for d in decisions),
                known_selected=sum(d['known'][d['legal_actions'].index(d['selected_action'])] for d in decisions),
                strict_learned_preference_steps=learned_strict,unseen_zero_over_negative_known_steps=unseen_better_negative,
                deterministic_ties=stats['deployment_tied_maxima'],unseen_choices=stats['deployment_unseen_choices'],
                q_entries=stats['q_entries'],legal_state_action_slots=stats['legal_state_action_slots'],coverage=stats['q_table_coverage'],
                final_accuracy=r['validation']['accuracy'],final_macro_f1=r['validation']['macro_f1']))
    table(output/'runs.csv',runs)
    summary=[];groups=defaultdict(list)
    for row in runs:groups[(row['dataset'],row['model'],row['target'],row['method'])].append(row)
    for (d,m,t,method),rows in sorted(groups.items()):
        assert len(rows)==3 and {r['dense_seed'] for r in rows}=={101,202,303}
        row=dict(dataset=d,model=m,target=t,method=method,n_independent_dense_seeds=3,target_misses=sum(r['target_miss'] for r in rows))
        for metric in ('validation_accuracy','validation_macro_f1','delta_accuracy','delta_macro_f1','achieved_compression','search_seconds','actual_forwards','inference_median_seconds'):
            values=[r[metric] for r in rows];row[metric+'_mean']=statistics.mean(values);row[metric+'_sample_sd']=statistics.stdev(values)
        summary.append(row)
    table(output/'summary.csv',summary)
    lookup={(r['dataset'],r['model'],r['dense_seed'],r['target'],r['method']):r for r in runs};paired=[]
    for d in p['datasets']:
        for m in p['models']:
            for seed in p['dense_seeds']:
                for target in p['targets']:
                    for left,right in PAIRS:
                        a=lookup[d,m,seed,target,left];b=lookup[d,m,seed,target,right]
                        paired.append(dict(dataset=d,model=m,seed=seed,target=target,left=left,right=right,
                            right_minus_left_accuracy=b['validation_accuracy']-a['validation_accuracy'],
                            right_minus_left_macro_f1=b['validation_macro_f1']-a['validation_macro_f1'],
                            right_minus_left_compression=b['achieved_compression']-a['achieved_compression'],
                            right_over_left_search_seconds=b['search_seconds']/a['search_seconds']))
                    a=by_id[lookup[d,m,seed,target,'adaptive_greedy']['id']];b=by_id[lookup[d,m,seed,target,'lookahead_depth2']['id']]
                    ta=json.loads((root/'runs'/a['condition']['id']/'deployment.json').read_text())
                    tb=json.loads((root/'runs'/b['condition']['id']/'deployment.json').read_text())
                    divergence=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x['action']!=y['action'] or x['original_kept_ids']!=y['original_kept_ids']),None)
                    if divergence is None and len(ta)!=len(tb):divergence=min(len(ta),len(tb))
                    trajectory.append(dict(dataset=d,model=m,seed=seed,target=target,first_divergence_step_0based=divergence,
                        identical_final_widths=a['final_widths']==b['final_widths'],identical_retained_masks=a['retained_original_ids']==b['retained_original_ids'],
                        identical_final_tensors=a['tensor_sha256']==b['tensor_sha256'],greedy_steps=len(ta),lookahead_steps=len(tb),
                        lookahead_minus_greedy_accuracy=b['validation']['accuracy']-a['validation']['accuracy'],
                        lookahead_minus_greedy_f1=b['validation']['macro_f1']-a['validation']['macro_f1'],
                        greedy_id=a['condition']['id'],lookahead_id=b['condition']['id']))
    table(output/'paired_comparisons.csv',paired);table(output/'q_episodes.csv',episodes)
    table(output/'q_diagnostics.csv',qdiagnostics);table(output/'greedy_lookahead.csv',trajectory)
    paired_summary=[];paired_groups=defaultdict(list)
    for row in paired:paired_groups[row['dataset'],row['model'],row['target'],row['left'],row['right']].append(row)
    for (d,m,t,left,right),rows in sorted(paired_groups.items()):
        assert len(rows)==3 and {r['seed'] for r in rows}=={101,202,303}
        item=dict(dataset=d,model=m,target=t,left=left,right=right,n_dense_seeds=3)
        for metric in ('accuracy','macro_f1','compression'):
            values=[r['right_minus_left_'+metric] for r in rows]
            item[metric+'_mean_difference']=statistics.mean(values)
            item[metric+'_sample_sd']=statistics.stdev(values)
            item[metric+'_seed_directions']=';'.join(f"{r['seed']}:{'+' if v>0 else '-' if v<0 else '='}" for r,v in zip(rows,values))
        item['median_search_cost_ratio']=statistics.median(r['right_over_left_search_seconds'] for r in rows)
        paired_summary.append(item)
    table(output/'paired_summary.csv',paired_summary)
    figures(root,output,runs,episodes,qdiagnostics,trajectory)
    write(output/'PROVENANCE.json',dict(source_commit=p['source_commit'],audit_sha256=sha(audit_path),
        verified_conditions=252,test_evaluation=False,independent_dense_seeds_per_dataset_model=3,
        comparisons='paired within original frozen checkpoint and target; no episodes/targets treated as independent Dense fits'))
    (output/'README.md').write_text('# Verified V5 validation analysis\n\n252 verified conditions; paired within each frozen Dense checkpoint/target. Three\nindependent Dense seeds per dataset/model. Means and sample SD describe these\nthree fits; no statistical-significance or causal superiority claim.\n\nActual compression, validation accuracy/macro-F1 and measured search costs must\nbe considered together. Q table existence/coverage and episode rewards do not\nprove policy usefulness; inspect deployment known/unseen choices, ties and the\npaired baseline differences. Candidate transition requests, actual validation\nforwards and SNIP gradient forwards are separate counters. Test labels were not\nevaluated. Static intermediate accuracy was not recorded and is not inferred.\n')
    lines=['# V5 verified validation results','',
        'All252 conditions passed the independent stored-evidence and final-checkpoint audit. '
        'Each table row summarizes three paired original Dense training/split seeds. '
        'Target pairs and training episodes are related observations, not extra independently trained checkpoints.','',
        '| Dataset/model | Target | Method | Validation accuracy mean±SD | Macro-F1 mean±SD | Achieved compression mean | Search seconds mean |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['dataset']}/{row['model']} | {row['target']:.0%} | {row['method']} | "
            f"{row['validation_accuracy_mean']:.5f}±{row['validation_accuracy_sample_sd']:.5f} | "
            f"{row['validation_macro_f1_mean']:.5f}±{row['validation_macro_f1_sample_sd']:.5f} | "
            f"{row['achieved_compression_mean']:.5f} | {row['search_seconds_mean']:.3f} |")
    lines+=['','Paired effect directions, actual compression differences, sample SD and search-cost ratios '
            'are in paired_summary.csv. No significance test or aggregate superiority claim is made.',
            '',f"Target-miss outcomes retained: {sum(r['target_miss'] for r in runs)}/252. "
            f"Greedy/Lookahead identical final masks: {sum(r['identical_retained_masks'] for r in trajectory)}/36; "
            f"identical widths: {sum(r['identical_final_widths'] for r in trajectory)}/36. Width equality alone does not establish tensor equality.",
            '',f"Q deployment selected unseen actions on {sum(r['unseen_choices'] for r in qdiagnostics)} of "
            f"{sum(r['deployment_steps'] for r in qdiagnostics)} observed steps; "
            f"{sum(r['strict_learned_preference_steps'] for r in qdiagnostics)} steps had a known selected value strictly above every alternative. "
            'These describe implemented choices; they do not demonstrate that training improved validation quality. '
            'Compare Q against Greedy and Sequential Random in the paired table. Unobserved counterfactual accuracies remain unknown.',
            '', 'The inference benchmark is full-graph validation computation; parameter compression does not guarantee latency gains. '
            'Search sums are active method time; LSF allocation, import, I/O, validation reporting and queue waits are separate. '
            'No held-out test evaluation, Dense retraining or fine-tuning was performed.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(output/'HASHES.json',{f.name:sha(f) for f in output.iterdir() if f.is_file()})


def figures(root,output,runs,episodes,qdiagnostics,trajectory):
    """Article-ready paired-stratum figures, without pooling independently trained fits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    strata=[(d,m) for d in ('cora','citeseer','pubmed') for m in ('gcn','graphsage')]
    methods=list(dict.fromkeys(r['method'] for r in runs));colors=dict(zip(methods,plt.cm.tab10.colors))
    for metric,xfield,name in [('validation_accuracy','achieved_compression','accuracy_compression'),
                               ('validation_macro_f1','achieved_compression','f1_compression'),
                               ('delta_macro_f1','search_seconds','quality_search_cost')]:
        fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
        for ax,(d,m) in zip(axes.flat,strata):
            for method in methods:
                rows=[r for r in runs if r['dataset']==d and r['model']==m and r['method']==method]
                for target,marker in ((.3,'o'),(.5,'^')):
                    selected=[r for r in rows if r['target']==target]
                    ax.scatter([r[xfield] for r in selected],[r[metric] for r in selected],color=colors[method],marker=marker,s=24,label=method if target==.3 else None)
            if xfield=='search_seconds':ax.set_xscale('log')
            ax.set(title=f'{d} / {m}',xlabel=xfield,ylabel=metric);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=6)
        for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,field in enumerate(('search_seconds','actual_forwards')):
        totals=[sum(r[field] for r in runs if r['method']==method) for method in methods]
        axes[i].bar(methods,totals,color=[colors[m] for m in methods]);axes[i].tick_params(axis='x',rotation=65)
        axes[i].set(ylabel=field,title='Campaign sum; not independent replicates')
    for ext in ('png','pdf'):fig.savefig(output/('compute_breakdown.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for d,m in strata:
        rows=[r for r in qdiagnostics if r['dataset']==d and r['model']==m]
        axes[0].scatter([r['coverage'] for r in rows],[r['final_macro_f1'] for r in rows],label=f'{d}/{m}')
        axes[1].scatter([r['known_selected']/r['deployment_steps'] for r in rows],
                        [r['final_macro_f1'] for r in rows],label=f'{d}/{m}')
    axes[0].set(xlabel='Observed legal state-action coverage',ylabel='Final validation macro-F1')
    axes[1].set(xlabel='Deployment fraction selected known actions',ylabel='Final validation macro-F1')
    axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(output/('q_coverage_diagnostics.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    for ax,(d,m) in zip(axes.flat,strata):
        for target in (.3,.5):
            for seed in (101,202,303):
                rows=sorted([r for r in episodes if r['dataset']==d and r['model']==m and r['target']==target and r['seed']==seed],key=lambda r:r['episode'])
                ax.plot([r['episode']+1 for r in rows],[r['reward_sum'] for r in rows],label=f'{target:.0%}, seed{seed}',linewidth=.8)
        ax.set(title=f'{d} / {m}',xlabel='Q training episode',ylabel='Episode reward sum');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(output/('q_learning_curves.'+ext),dpi=180)
    plt.close(fig)
    case=next((r for r in trajectory if r['first_divergence_step_0based'] is not None),trajectory[0])
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for label,name in [('Greedy',case['greedy_id']),('Lookahead depth2',case['lookahead_id'])]:
        trace=json.loads((root/'runs'/name/'deployment.json').read_text())
        report=json.loads((root/'runs'/name/'report.json').read_text())
        ax.plot([0]+[r['sparsity'] for r in trace],[report['dense_validation']['accuracy']]+[r['validation_accuracy'] for r in trace],marker='o',label=label)
    ax.set(xlabel='Achieved physical compression',ylabel='Validation accuracy',title=f"{case['dataset']} / {case['model']} / seed{case['seed']} / target{case['target']:.0%}")
    ax.legend();ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(output/('paired_trajectory.'+ext),dpi=180)
    plt.close(fig)
    write(output/'ILLUSTRATED_CASE.json',dict(selection='First lexicographically ordered pair with observed divergence; otherwise first pair. Not selected by accuracy.',case=case))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analysis(a.root,a.audit,a.output)
