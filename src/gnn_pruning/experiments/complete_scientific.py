"""Scientific analysis of frozen audited JSON/CSV; no GNN execution imports.

All comparisons retain warnings and use paired independent dense seeds. Recorded
planning previews and Q values are observations, not newly executed rollouts.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics as st
import struct

METHODS=('static_random','static_magnitude','static_snip','sequential_random','adaptive_greedy','lookahead_depth2','tabular_q')
EPS=1e-10  # Percentage-point ties, far below a single-node accuracy change.
NUMERIC=('requested_compression','achieved_compression','signed_target_deviation','parameters_before','parameters_after','search_seconds',
         'validation_evaluations','candidate_evaluations','inference_mean_ms','inference_std_ms','dense_training_seconds',
         'dense_checkpoint_bytes','pruned_checkpoint_bytes','parameter_bytes','q_entries','q_table_sparsity',
         'dense_val_accuracy','pruned_val_accuracy','delta_val_accuracy','dense_val_macro_f1','pruned_val_macro_f1','delta_val_macro_f1',
         'dense_test_accuracy','pruned_test_accuracy','delta_test_accuracy','dense_test_macro_f1','pruned_test_macro_f1','delta_test_macro_f1')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(value):
    return tuple(freeze(x) for x in value) if isinstance(value,(tuple,list)) else value


def f32(value: float) -> float:
    return struct.unpack('f',struct.pack('f',value))[0]


def save(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})


def stat(values: list[float]) -> dict:
    """Sample SD of paired seed observations, never a standard error or CI."""
    return dict(n=len(values),mean=st.mean(values) if values else None,sd=st.stdev(values) if len(values)>1 else None,
                median=st.median(values) if values else None,min=min(values) if values else None,max=max(values) if values else None)


def load_rows(path: Path) -> list[dict]:
    with path.open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key in NUMERIC:row[key]=float(row[key]) if row.get(key) not in ('',None) else None
        row['dense_training_seed']=int(row['dense_training_seed'])
        for key in ('collected','verified','physical_pruning','outside_symmetric_band','dense_reused'):
            row[key]=row[key]=='True'
        for key in ('retained_original_ids','final_widths','issues','behavior_details'):
            row[key]=json.loads(row[key])
    return rows


def validate(rows: list[dict], collection: Path, raw: Path) -> dict:
    """Verify existing audit anchors, every raw file and all report identities."""
    hashes=json.loads((collection/'deliverable_hashes.json').read_text())
    for name,item in hashes.items():
        p=collection/name
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('AUDIT_DELIVERABLE_HASH: '+name)
    audit=json.loads((collection/'audit.json').read_text())
    if audit['totals']['verified_runs']!=252 or len(rows)!=252 or not all(r['verified'] for r in rows):raise ValueError('INCOMPLETE_AUDITED_POPULATION')
    expected={r['run_id'] for r in audit['runs']};actual=[r['run_id'] for r in rows]
    if set(actual)!=expected or len(set(actual))!=len(actual):raise ValueError('DUPLICATE_OR_MISSING_RUN_ID')
    original={r['run_id']:r for r in audit['runs']}
    for r in rows:
        old=original[r['run_id']]
        for key in ('retained_original_ids','final_widths','issues','physical_pruning','outside_symmetric_band','verified','source_commit','dense_checkpoint_sha256','split_sha256','pruned_tensor_sha256'):
            prior=old[key]
            if key in ('retained_original_ids','final_widths') and isinstance(prior,str):prior=json.loads(prior)
            if r[key]!=prior:raise ValueError('CSV_AUDIT_IDENTITY_MISMATCH: '+key)
        for key in NUMERIC:
            value=r.get(key);before=old.get(key)
            if (value is None)!=(before is None) or (value is not None and abs(value-float(before))>1e-12):raise ValueError('CSV_AUDIT_VALUE_MISMATCH: '+key)
    groups=defaultdict(list)
    for r in rows:groups[r['condition_id']].append(r)
    if len(groups)!=18:raise ValueError('DENSE_REPLICATION_COUNT')
    for group in groups.values():
        if len(group)!=14 or len({r['dense_checkpoint_sha256'] for r in group})!=1 or len({r['split_sha256'] for r in group})!=1:raise ValueError('UNPAIRED_CHECKPOINT_OR_SPLIT')
        if {(r['method'],r['requested_compression']) for r in group}!={(m,t) for m in METHODS for t in (.3,.5)}:raise ValueError('METHOD_TARGET_MATRIX')
    with (collection/'artifact_manifest.csv').open() as stream:inventory=list(csv.DictReader(stream))
    for item in inventory:
        path=raw/item['path']
        if item['status']!='VERIFIED' or not path.is_file() or path.stat().st_size!=int(item['expected_bytes']) or sha(path)!=item['expected_sha256']:raise ValueError('RAW_EVIDENCE_HASH: '+item['path'])
    return dict(observations=252,dense_checkpoints=18,independent_seeds_per_cell=3,raw_files_reverified=len(inventory),
        statuses=dict(Counter(r['audit_status'] for r in rows)),physically_pruned=sum(r['physical_pruning'] for r in rows),
        outside_band=sum(r['outside_symmetric_band'] for r in rows),original_population_unchanged=True)


def make_pairs(rows: list[dict]) -> list[dict]:
    """Every method pair within a checkpoint/target, B minus A, never unpaired."""
    groups=defaultdict(dict)
    for r in rows:
        key=(r['condition_id'],r['requested_compression'])
        if r['method'] in groups[key]:raise ValueError('DUPLICATE_PAIRED_OBSERVATION')
        groups[key][r['method']]=r
    output=[]
    for (condition,target),group in sorted(groups.items()):
        if set(group)!=set(METHODS):raise ValueError('INCOMPLETE_PAIRING')
        for ma,mb in combinations(METHODS,2):
            a,b=group[ma],group[mb]
            if a['dense_checkpoint_sha256']!=b['dense_checkpoint_sha256'] or a['split_sha256']!=b['split_sha256']:raise ValueError('PAIR_INPUT_MISMATCH')
            masks_equal=a['retained_original_ids']==b['retained_original_ids'];params_equal=a['parameters_after']==b['parameters_after']
            if masks_equal and a['pruned_tensor_sha256']!=b['pruned_tensor_sha256']:raise ValueError('IDENTICAL_MASK_DIFFERENT_TENSORS')
            p=dict(dataset=a['dataset'],architecture=a['architecture'],condition_id=condition,dense_training_seed=a['dense_training_seed'],
                requested_compression=target,method_a=ma,method_b=mb,run_a=a['run_id'],run_b=b['run_id'],
                accuracy_a=a['pruned_val_accuracy'],accuracy_b=b['pruned_val_accuracy'],macro_f1_a=a['pruned_val_macro_f1'],macro_f1_b=b['pruned_val_macro_f1'],
                delta_accuracy_pp=100*(b['pruned_val_accuracy']-a['pruned_val_accuracy']),delta_macro_f1_pp=100*(b['pruned_val_macro_f1']-a['pruned_val_macro_f1']),
                compression_a=a['achieved_compression'],compression_b=b['achieved_compression'],delta_compression_pp=100*(b['achieved_compression']-a['achieved_compression']),
                parameters_a=a['parameters_after'],parameters_b=b['parameters_after'],same_parameter_count=params_equal,
                widths_a=a['final_widths'],widths_b=b['final_widths'],same_widths=a['final_widths']==b['final_widths'],same_retained_mask=masks_equal,
                symmetric_channel_difference=sum(len(set(x)^set(y)) for x,y in zip(a['retained_original_ids'],b['retained_original_ids'])),
                search_a_seconds=a['search_seconds'],search_b_seconds=b['search_seconds'],search_ratio_b_over_a=b['search_seconds']/a['search_seconds'],
                validation_evaluations_a=a['validation_evaluations'],validation_evaluations_b=b['validation_evaluations'],
                validation_evaluation_ratio=b['validation_evaluations']/a['validation_evaluations'] if a['validation_evaluations'] else None,
                candidate_evaluations_a=a['candidate_evaluations'],candidate_evaluations_b=b['candidate_evaluations'],
                candidate_evaluation_ratio=b['candidate_evaluations']/a['candidate_evaluations'] if a['candidate_evaluations'] and b['candidate_evaluations'] is not None else None,
                physically_pruned_a=a['physical_pruning'],physically_pruned_b=b['physical_pruning'],both_in_band=not a['outside_symmetric_band'] and not b['outside_symmetric_band'],
                audit_a=a['audit_status'],audit_b=b['audit_status'],dense_checkpoint_sha256=a['dense_checkpoint_sha256'],split_sha256=a['split_sha256'])
            p['comparison_scope']=('infeasible_or_unpruned_outcome' if not a['physical_pruning'] or not b['physical_pruning'] else
                'identical_physical_mask' if masks_equal else 'equal_parameter_count_different_mask' if params_equal else 'unequal_achieved_compression')
            output.append(p)
    return output


def pair_summary(pairs: list[dict]) -> list[dict]:
    groups=defaultdict(list)
    for p in pairs:groups[(p['dataset'],p['architecture'],p['requested_compression'],p['method_a'],p['method_b'])].append(p)
    output=[]
    for key,group in sorted(groups.items()):
        if sorted(p['dense_training_seed'] for p in group)!=[101,202,303]:raise ValueError('NOT_THREE_INDEPENDENT_PAIRED_SEEDS')
        row=dict(zip(('dataset','architecture','requested_compression','method_a','method_b'),key));row['n_dense_pairs']=3
        for name in ('delta_accuracy_pp','delta_macro_f1_pp','delta_compression_pp','search_ratio_b_over_a','validation_evaluation_ratio','candidate_evaluation_ratio'):
            for suffix,value in stat([p[name] for p in group if p[name] is not None]).items():row[name+'_'+suffix]=value
        for metric in ('accuracy','macro_f1'):
            values=[p['delta_'+metric+'_pp'] for p in group]
            row[metric+'_wins_b']=sum(v>EPS for v in values);row[metric+'_ties']=sum(abs(v)<=EPS for v in values);row[metric+'_losses_b']=sum(v < -EPS for v in values)
        row.update(same_mask_pairs=sum(p['same_retained_mask'] for p in group),same_parameter_count_pairs=sum(p['same_parameter_count'] for p in group),
                   unequal_compression_pairs=sum(not p['same_parameter_count'] for p in group),unpruned_pairs=sum(not p['physically_pruned_a'] or not p['physically_pruned_b'] for p in group))
        output.append(row)
    return output


def method_summary(rows: list[dict],reports: dict) -> list[dict]:
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['architecture'],r['requested_compression'],r['method'])].append(r)
    output=[]
    for key,group in sorted(groups.items()):
        if sorted(r['dense_training_seed'] for r in group)!=[101,202,303]:raise ValueError('METHOD_REPLICATION')
        row=dict(zip(('dataset','architecture','requested_compression','method'),key))
        row.update(n_expected=3,n_observed=len(group),n_warning=sum(r['audit_status']=='WARNING' for r in group),
                   n_physically_pruned=sum(r['physical_pruning'] for r in group),n_in_band=sum(not r['outside_symmetric_band'] for r in group),
                   final_widths=[r['final_widths'] for r in sorted(group,key=lambda r:r['dense_training_seed'])],
                   stopping_reasons=dict(Counter(r['terminal_reason'] for r in group)))
        for name in ('pruned_val_accuracy','pruned_val_macro_f1','delta_val_accuracy','delta_val_macro_f1','achieved_compression','signed_target_deviation',
                     'parameters_after','parameter_bytes','search_seconds','validation_evaluations','candidate_evaluations','inference_mean_ms','inference_std_ms'):
            values=[r[name] for r in group if r.get(name) is not None]
            for suffix,value in stat(values).items():row[name+'_'+suffix]=value
        scoring=[reports[r['run_id']]['statistics'].get('scoring_seconds') for r in group]
        surgery=[reports[r['run_id']]['statistics'].get('surgery_seconds') for r in group]
        for name,values in [('scoring_seconds',scoring),('surgery_seconds',surgery)]:
            for suffix,value in stat([v for v in values if v is not None]).items():row[name+'_'+suffix]=value
        output.append(row)
    return output


def dominates(a: dict,b: dict,quality: str) -> bool:
    """Recorded three-objective dominance, not a significance/superiority test."""
    return (a[quality]>=b[quality]-1e-12 and a['achieved_compression']>=b['achieved_compression']-1e-12 and a['search_seconds']<=b['search_seconds'] and
            (a[quality]>b[quality]+1e-12 or a['achieved_compression']>b['achieved_compression']+1e-12 or a['search_seconds']<b['search_seconds']))


def pareto(rows: list[dict]) -> list[dict]:
    groups=defaultdict(list)
    for r in rows:groups[(r['condition_id'],r['requested_compression'])].append(r)
    output=[]
    for group in groups.values():
        feasible=[r for r in group if r['physical_pruning'] and not r['outside_symmetric_band']]
        for r in group:
            out=dict(run_id=r['run_id'],condition_id=r['condition_id'],dataset=r['dataset'],architecture=r['architecture'],
                     requested_compression=r['requested_compression'],dense_training_seed=r['dense_training_seed'],method=r['method'],
                     feasible_for_requested_band=r in feasible,n_methods_within_band=len(feasible),audit_status=r['audit_status'])
            for short,quality in [('accuracy','pruned_val_accuracy'),('macro_f1','pruned_val_macro_f1')]:
                dom=[p['method'] for p in group if p is not r and dominates(p,r,quality)]
                out[short+'_front_all_outcomes']=not dom;out[short+'_dominated_by']=dom
                out[short+'_front_within_band']=None if r not in feasible else not any(p is not r and dominates(p,r,quality) for p in feasible)
            output.append(out)
    return output


def legal_q(widths: list[int],step: int,minimum: int=4) -> list[str]:
    return [f'prune:{i}:{k}' for i in (0,1) for k in (1,2,4,8) if widths[i]-k>=minimum]+(['stop'] if step else [])


def q_diagnostics(rows: list[dict],reports: dict,protocol: dict) -> tuple[list, list, list]:
    """Read frozen Q tables and traces. No Q updates, episodes or actions executed."""
    runs=[];decisions=[];episodes=[]
    for r in rows:
        if r['method']!='tabular_q':continue
        d=reports[r['run_id']];table={freeze(x['state']):x['actions'] for x in d['q_table']}
        seen=defaultdict(set);observed_rewards=defaultdict(list);training=defaultdict(list)
        for entry in d['training']:
            if entry.get('event')!='reset':training[entry['episode']].append(entry)
        for episode,entries in training.items():
            widths=[32,32];keeps=[list(range(32)),list(range(32))];acc=f32(r['dense_val_accuracy'])
            for step,entry in enumerate(entries):
                state=('local_v2',tuple(widths),20-step,math.floor(acc/.01),False);seen[state].add(freeze(keeps));observed_rewards[(state,entry['action'])].append(entry['reward'])
                widths=entry['widths_after'];keeps=entry['original_kept_ids'];acc=entry['validation_accuracy']
            last=entries[-1]
            episodes.append(dict(run_id=r['run_id'],dataset=r['dataset'],architecture=r['architecture'],dense_training_seed=r['dense_training_seed'],
                requested_compression=r['requested_compression'],episode=episode,transitions=len(entries),successful_actions=sum(e['action'].startswith('prune:') for e in entries),
                terminal_reason=last['terminal_reason'],final_compression=last['sparsity'],final_validation_accuracy=last['validation_accuracy'],
                terminal_penalty=last['terminal_penalty'],undiscounted_return=sum(e['reward'] for e in entries),discounted_return=sum(.9**i*e['reward'] for i,e in enumerate(entries))))
        acc=f32(r['dense_val_accuracy']);widths=[32,32];keeps=[list(range(32)),list(range(32))];before_s=0.;first_crossing=None;run_decisions=[]
        for step,entry in enumerate(d['trace']):
            state=('local_v2',tuple(widths),20-step,math.floor(acc/.01),False);row=table.get(state,{})
            legal=legal_q(widths,step);values={a:row.get(a,0.) for a in legal};maximum=max(values.values());ties=[a for a in legal if values[a]==maximum]
            if entry['action']!=ties[0]:raise ValueError('RECORDED_Q_DEPLOYMENT_RULE_MISMATCH')
            known=[a for a in legal if a in row];selected=entry['action'];stop='stop' in legal
            rec=dict(run_id=r['run_id'],dataset=r['dataset'],architecture=r['architecture'],dense_training_seed=r['dense_training_seed'],requested_compression=r['requested_compression'],step=step+1,
                state=list(state),action=selected,selected_q=values[selected],selected_known=selected in row,unseen_state=not row,
                n_known_alternatives=len(known),n_legal_actions=len(legal),n_tied_maxima=len(ties),tied_maxima=ties,
                selected_unseen_over_strictly_negative_known=selected not in row and bool(known) and max(row[a] for a in known)<0,
                stop_legal=stop,stop_known=stop and 'stop' in row,stop_q=values.get('stop'),stop_tied_maximum='stop' in ties,
                analytic_immediate_stop_reward=-protocol['adaptive']['penalty_lambda']*max(0.,abs(before_s-r['requested_compression'])-.01) if stop else None,
                compression_before=before_s,compression_after=entry['sparsity'],validation_accuracy_after=entry['validation_accuracy'],
                reward=entry['reward'],terminal_penalty=entry['terminal_penalty'],terminal_reason=entry['terminal_reason'],
                deployment_mask_seen_under_aggregate_state=freeze(keeps) in seen.get(state,set()),aggregate_state_aliased=len(seen.get(state,set()))>1)
            decisions.append(rec);run_decisions.append(rec)
            if first_crossing is None and entry['sparsity']>=r['requested_compression']-.01:first_crossing=step+1
            acc=entry['validation_accuracy'];widths=entry['widths_after'];keeps=entry['original_kept_ids'];before_s=entry['sparsity']
        rewards_vary=sum(len(vals)>1 and max(vals)-min(vals)>1.2e-7 for vals in observed_rewards.values())
        output=dict(run_id=r['run_id'],condition_id=r['condition_id'],dataset=r['dataset'],architecture=r['architecture'],dense_training_seed=r['dense_training_seed'],
            requested_compression=r['requested_compression'],achieved_compression=r['achieved_compression'],final_widths=r['final_widths'],
            actions=len(d['trace']),terminal_reason=r['terminal_reason'],first_target_threshold_crossing=first_crossing,
            further_actions_after_threshold=20-first_crossing if first_crossing else None,
            ever_in_symmetric_band=any(abs(e['sparsity']-r['requested_compression'])<=protocol['adaptive']['tolerance'] for e in d['trace']),
            first_in_band_step=next((i+1 for i,e in enumerate(d['trace']) if abs(e['sparsity']-r['requested_compression'])<=protocol['adaptive']['tolerance']),None),
            search_seconds=r['search_seconds'],
            validation_evaluations=r['validation_evaluations'],q_entries=r['q_entries'],q_table_sparsity=r['q_table_sparsity'],
            training_terminal_reasons=dict(Counter(es[-1]['terminal_reason'] for es in training.values())),
            deployment_selected_unseen=sum(not v['selected_known'] for v in run_decisions),deployment_unseen_states=sum(v['unseen_state'] for v in run_decisions),
            deployment_stop_tied_maximum=sum(v['stop_tied_maximum'] for v in run_decisions),deployment_stop_legal=sum(v['stop_legal'] for v in run_decisions),
            training_aggregate_states=len(seen),training_aliased_states=sum(len(v)>1 for v in seen.values()),
            repeated_state_action_different_rewards=rewards_vary,
            deployment_training_mask_unseen=sum(not v['deployment_mask_seen_under_aggregate_state'] for v in run_decisions),
            terminal_penalty=d['trace'][-1]['terminal_penalty'])
        runs.append(output)
    return runs,decisions,episodes


def bridge_greedy(g: dict,l: dict,dense_accuracy: float) -> list[dict]:
    """Corroborate Greedy using recorded LA previews at identical concrete parents.

Identity includes retained masks, exact parent accuracy and remaining horizon.
Depth-2 nonselected parent branches are eligible evidence; aggregate keys alone
are not. No scores or model predictions are generated.
"""
    if any(g[k]!=l[k] for k in ('checkpoint_sha256','split_sha256','graph_tensor_sha256')) or g['run']['target']!=l['run']['target']:
        raise ValueError('BRIDGE_INPUT_IDENTITY')
    catalog=defaultdict(lambda:defaultdict(list));acc=f32(dense_accuracy)
    for di,d in enumerate(l['decisions']):
        roots={item['path'][0]:item for item in d['candidate_transitions'] if len(item['path'])==1}
        for item in d['candidate_transitions']:
            prefix=roots[item['path'][0]]['transition'] if len(item['path'])==2 else None
            mask=prefix['original_kept_ids'] if prefix else d['original_kept_ids'];parent_acc=prefix['validation_accuracy'] if prefix else acc
            key=(freeze(mask),item['from_state'][2],parent_acc)
            catalog[key][item['transition']['action']].append((item['transition'],di+1,item['path']))
        acc=l['trace'][di]['validation_accuracy']
    before=[list(range(32)),list(range(32))];acc=f32(dense_accuracy);results=[]
    for i,entry in enumerate(g['trace']):
        options=catalog.get((freeze(before),20-i,acc),{})
        legal=[a for a in legal_q(list(map(len,before)),0) if a!='stop']
        if set(options)!=set(legal):
            results.append(dict(run_id=g['run']['id'],step=i+1,status='NOT_VERIFIED_NO_IDENTICAL_PARENT_PREVIEWS'))
        else:
            for opts in options.values():
                if any(o[0]!=opts[0][0] for o in opts):raise ValueError('INCONSISTENT_SAME_PARENT_PREVIEWS')
            best=max(legal,key=lambda a:options[a][0][0]['reward']);observation,di,path=options[entry['action']][0]
            if best!=entry['action'] or observation!=entry:raise ValueError('GREEDY_BRIDGE_NOT_CORROBORATED')
            results.append(dict(run_id=g['run']['id'],step=i+1,status='CORROBORATED_RECORDED_PREVIEWS',action=entry['action'],
                best_immediate_reward=observation['reward'],lookahead_decision=di,lookahead_candidate_path=path,
                n_legal_previews=len(legal),evidence_parent_depth=len(path)-1))
        before=entry['original_kept_ids'];acc=entry['validation_accuracy']
    return results


def analyze(collection: Path,raw: Path,output: Path) -> dict:
    """Create reproducible tables from existing verified observations only."""
    output.mkdir(parents=True,exist_ok=True)
    if (output/'ANALYSIS_RESULTS.json').exists():raise FileExistsError('Preserve previous analysis: select a fresh output directory.')
    rows=load_rows(collection/'verified_results.csv');validation=validate(rows,collection,raw)
    reports={r['run_id']:json.loads((raw/'runs'/r['run_id']/'report.json').read_text()) for r in rows}
    for r in rows:
        if sha(raw/'runs'/r['run_id']/'report.json')!=r['output_report_sha256']:raise ValueError('RAW_TRACE_IDENTITY')
    protocol=json.loads((raw/'protocol.json').read_text())
    pairs=make_pairs(rows);paired=pair_summary(pairs);summary=method_summary(rows,reports);front=pareto(rows)
    qr,qd,qe=q_diagnostics(rows,reports,protocol);bridge=[];trajectory=[]
    rowmap={r['run_id']:r for r in rows}
    for r in rows:
        if r['method']!='adaptive_greedy':continue
        g=reports[r['run_id']];lid=r['run_id'].replace('adaptive_greedy','lookahead_depth2');l=reports[lid]
        seq_g=[x['action'] for x in g['trace']];seq_l=[x['action'] for x in l['trace']]
        prefix=0
        for a,b in zip(seq_g,seq_l):
            if a!=b:break
            prefix+=1
        trajectory.append(dict(condition_id=r['condition_id'],dataset=r['dataset'],architecture=r['architecture'],dense_training_seed=r['dense_training_seed'],
            requested_compression=r['requested_compression'],same_mask=g['retained_original_ids']==l['retained_original_ids'],same_action_sequence=seq_g==seq_l,
            greedy_actions=seq_g,lookahead_actions=seq_l,common_action_prefix=prefix,first_divergent_step=prefix+1 if seq_g!=seq_l else None,
            delta_accuracy_pp=100*(rowmap[lid]['pruned_val_accuracy']-r['pruned_val_accuracy']),delta_macro_f1_pp=100*(rowmap[lid]['pruned_val_macro_f1']-r['pruned_val_macro_f1']),
            search_ratio=rowmap[lid]['search_seconds']/r['search_seconds'],candidate_ratio=rowmap[lid]['candidate_evaluations']/r['candidate_evaluations']))
        bridge+=bridge_greedy(g,l,r['dense_val_accuracy'])
    case_g=reports['pubmed-gcn-s202-adaptive_greedy-t50'];case_l=reports['pubmed-gcn-s202-lookahead_depth2-t50'];decision=case_l['decisions'][1]
    best_by_first={key:max((p for p in decision['candidate_paths'] if p['actions'][0]==key),key=lambda p:p['score']) for key in decision['legal_actions']}
    case=dict(greedy=case_g,lookahead=case_l,comparison=next(p for p in pairs if p['run_a']==case_g['run']['id'] and p['method_b']=='lookahead_depth2'),
        shared_parent=decision['original_kept_ids'],shared_state=decision['state'],best_paths_by_first_action=best_by_first,
        first_decision_original_best_path=case_l['decisions'][0]['best_planned_path'],selection_rule='case requested explicitly by user; selected from validation/mask diagnostics, not test outcomes')
    # Compact case evidence includes only deployment and the relevant planning tree.
    save(output/'LOOKAHEAD_CASE_EVIDENCE.json',dict(comparison=case['comparison'],shared_state=case['shared_state'],shared_parent=case['shared_parent'],
        best_paths_by_first_action=best_by_first,first_decision_original_best_path=case['first_decision_original_best_path'],
        greedy_trace=case_g['trace'],lookahead_trace=case_l['trace'],case_selection=case['selection_rule']))
    qtarget=[]
    for condition in sorted({r['condition_id'] for r in rows}):
        ra=next(r for r in rows if r['condition_id']==condition and r['method']=='tabular_q' and r['requested_compression']==.3)
        rb=next(r for r in rows if r['condition_id']==condition and r['method']=='tabular_q' and r['requested_compression']==.5)
        da,db=reports[ra['run_id']],reports[rb['run_id']]
        qtarget.append(dict(condition_id=condition,same_mask=ra['retained_original_ids']==rb['retained_original_ids'],same_action_sequence=[x['action'] for x in da['trace']]==[x['action'] for x in db['trace']],
            same_final_quality=abs(ra['pruned_val_accuracy']-rb['pruned_val_accuracy'])<=1e-12 and abs(ra['pruned_val_macro_f1']-rb['pruned_val_macro_f1'])<=1e-12,
            same_q_table={freeze(x['state']):x['actions'] for x in da['q_table']}=={freeze(x['state']):x['actions'] for x in db['q_table']},terminal_penalty_difference_t50_minus_t30=db['trace'][-1]['terminal_penalty']-da['trace'][-1]['terminal_penalty']))
    results=dict(validation=validation,greedy_lookahead=dict(pairs=len(trajectory),same_masks=sum(x['same_mask'] for x in trajectory),same_sequences=sum(x['same_action_sequence'] for x in trajectory),
        search_ratio_descriptive_over_36=stat([x['search_ratio'] for x in trajectory]),candidate_ratio_descriptive_over_36=stat([x['candidate_ratio'] for x in trajectory]),
        bridges=dict(n=len(bridge),corroborated=sum(x['status']=='CORROBORATED_RECORDED_PREVIEWS' for x in bridge),depth2_parent=sum(x.get('evidence_parent_depth')==1 for x in bridge))),
        q=dict(compression=stat([r['achieved_compression'] for r in qr]),deployments=36,deployment_steps=len(qd),stop_reasons=dict(Counter(r['terminal_reason'] for r in qr)),
            training_episode_terminal=dict(Counter(e['terminal_reason'] for e in qe)),ever_in_symmetric_band=sum(r['ever_in_symmetric_band'] for r in qr),target_pairs=qtarget,
            selected_unseen=sum(not d['selected_known'] for d in qd),unseen_state=sum(d['unseen_state'] for d in qd),
            stop_legal=sum(d['stop_legal'] for d in qd),stop_tied_maximum=sum(d['stop_tied_maximum'] for d in qd),stop_known=sum(d['stop_known'] for d in qd),
            unseen_over_negative=sum(d['selected_unseen_over_strictly_negative_known'] for d in qd),selected_learned_positive=sum(d['selected_known'] and d['selected_q']>0 for d in qd)),
        analysis_units='n=3 independently trained dense checkpoints within each dataset/model/target; repeated targets and policy episodes are not independent',
        uncertainty='sample SD only; no confidence intervals or hypothesis tests',new_training=False,new_pruning=False,new_inference=False,new_timing=False)
    write_csv(output/'PAIRED_COMPARISONS.csv',pairs);write_csv(output/'PAIRED_SUMMARY.csv',paired);write_csv(output/'METHOD_SUMMARY.csv',summary)
    write_csv(output/'PARETO_DIAGNOSTICS.csv',front);write_csv(output/'GREEDY_LOOKAHEAD_TRAJECTORIES.csv',trajectory);write_csv(output/'GREEDY_PREVIEW_BRIDGE.csv',bridge)
    write_csv(output/'Q_RUN_DIAGNOSTICS.csv',qr);write_csv(output/'Q_DEPLOYMENT_DECISIONS.csv',qd);write_csv(output/'Q_EPISODE_SUMMARY.csv',qe);write_csv(output/'Q_TARGET_PAIR_CHECK.csv',qtarget)
    save(output/'ANALYSIS_RESULTS.json',results)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','raw','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(analyze(args.collection,args.raw,args.output),indent=2))
