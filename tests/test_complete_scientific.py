"""Pure JSON/CSV analysis checks: no models, training, pruning or inference."""
from copy import deepcopy
from pathlib import Path
import ast
import math

import pytest
from gnn_pruning.experiments import complete_scientific as a


def sample_rows(targets=(.3,)):
    rows=[]
    for seed in (101,202,303):
        for target in targets:
            for index,method in enumerate(a.METHODS):
                rows.append(dict(run_id=f's{seed}-{method}-{target}',condition_id=f'cora-gcn-s{seed}',dataset='cora',architecture='gcn',dense_training_seed=seed,
                    method=method,requested_compression=target,dense_checkpoint_sha256=f'dense{seed}',split_sha256=f'split{seed}',
                    pruned_val_accuracy=.70+index*.01,pruned_val_macro_f1=.6+index*.01,achieved_compression=target,parameters_after=100,
                    retained_original_ids=[[0,1,2,3],[0,1,2,3]],final_widths=[4,4],pruned_tensor_sha256='same',
                    search_seconds=1.+index,validation_evaluations=0 if index<3 else index,candidate_evaluations=0 if index<3 else index,
                    physical_pruning=True,outside_symmetric_band=False,audit_status='PASS'))
    return rows


def test_sample_sd_and_missing_uncertainty():
    assert a.stat([1,2,3])['sd']==1
    assert a.stat([1])['sd'] is None and a.stat([])['mean'] is None


def test_pair_orientation_and_correct_replicates():
    rows=sample_rows();pairs=a.make_pairs(rows)
    assert len(pairs)==63
    p=next(p for p in pairs if p['method_a']=='static_random' and p['method_b']=='static_magnitude')
    assert p['delta_accuracy_pp']==pytest.approx(1.) and p['search_ratio_b_over_a']==2.
    summary=a.pair_summary(pairs);assert len(summary)==21
    assert all(r['n_dense_pairs']==3 and r['delta_accuracy_pp_sd']==pytest.approx(0.) for r in summary)


def test_repeated_targets_are_not_extra_independent_seeds():
    pairs=a.make_pairs(sample_rows((.3,.5)));summaries=a.pair_summary(pairs)
    assert len(pairs)==126 and len(summaries)==42
    assert all(r['n_dense_pairs']==3 for r in summaries)


@pytest.mark.parametrize('kind',['duplicate','missing','checkpoint','split'])
def test_bad_pairing_is_rejected(kind):
    rows=sample_rows()
    if kind=='duplicate':rows.append(deepcopy(rows[0]))
    elif kind=='missing':rows.pop()
    elif kind=='checkpoint':rows[0]['dense_checkpoint_sha256']='different'
    elif kind=='split':rows[0]['split_sha256']='different'
    with pytest.raises(ValueError):a.make_pairs(rows)


def test_equal_widths_are_not_equal_original_masks():
    rows=sample_rows();rows[1]['retained_original_ids']=[[1,2,3,4],[0,1,2,3]];rows[1]['pruned_tensor_sha256']='different'
    pair=next(p for p in a.make_pairs(rows) if p['method_a']=='static_random' and p['method_b']=='static_magnitude' and p['dense_training_seed']==101)
    assert pair['same_widths'] and not pair['same_retained_mask'] and pair['symmetric_channel_difference']==2
    assert pair['comparison_scope']=='equal_parameter_count_different_mask'


def test_noop_warnings_remain_in_all_comparisons():
    rows=sample_rows();rows[0].update(physical_pruning=False,outside_symmetric_band=True,audit_status='WARNING',achieved_compression=0.)
    pairs=a.make_pairs(rows);assert len(pairs)==63
    assert sum(p['comparison_scope']=='infeasible_or_unpruned_outcome' for p in pairs)==6


def test_inconsistent_mask_tensor_identity_rejected():
    rows=sample_rows();rows[1]['pruned_tensor_sha256']='unexpected'
    with pytest.raises(ValueError,match='IDENTICAL_MASK'):a.make_pairs(rows)


def test_missing_candidate_ratio_not_invented():
    pair=a.make_pairs(sample_rows())[0]
    assert pair['candidate_evaluation_ratio'] is None and pair['validation_evaluation_ratio'] is None


def test_pareto_retains_extreme_compression_and_infeasible_rows():
    rows=sample_rows()[:7]
    for r in rows:r['pruned_val_accuracy']=.8;r['pruned_val_macro_f1']=.7
    rows[-1].update(pruned_val_accuracy=.2,pruned_val_macro_f1=.1,achieved_compression=.875,outside_symmetric_band=True)
    fronts=a.pareto(rows);assert len(fronts)==7
    assert fronts[-1]['accuracy_front_all_outcomes'] and fronts[-1]['accuracy_front_within_band'] is None
    assert not fronts[1]['accuracy_front_all_outcomes']


def bridge_fixture():
    before=[list(range(32)),list(range(32))];items=[]
    for action in a.legal_q([32,32],0):
        _,layer,k=action.split(':');layer,k=int(layer),int(k);mask=deepcopy(before);mask[layer]=mask[layer][:-k]
        info=dict(action=action,original_kept_ids=mask,validation_accuracy=.8,reward=1. if action=='prune:0:8' else 0.)
        items.append(dict(path=[action],from_state=['local_v2',[32,32],20,80,False],transition=info))
    selected=next(x['transition'] for x in items if x['path']==['prune:0:8'])
    common=dict(checkpoint_sha256='same',split_sha256='split',graph_tensor_sha256='graph',run=dict(id='g',target=.3))
    g=dict(common,trace=[selected]);l=dict(common,trace=[selected],decisions=[dict(original_kept_ids=before,candidate_transitions=items)])
    return g,l


def test_greedy_bridge_uses_recorded_previews_only():
    g,l=bridge_fixture();result=a.bridge_greedy(g,l,.8)
    assert result[0]['status']=='CORROBORATED_RECORDED_PREVIEWS' and result[0]['n_legal_previews']==8


@pytest.mark.parametrize('change',['mask','horizon','incomplete'])
def test_bridge_does_not_reuse_aliased_or_incomplete_parents(change):
    g,l=bridge_fixture()
    if change=='mask':l['decisions'][0]['original_kept_ids'][0]=list(range(1,33))
    elif change=='horizon':
        for item in l['decisions'][0]['candidate_transitions']:item['from_state'][2]=19
    else:l['decisions'][0]['candidate_transitions'].pop()
    assert a.bridge_greedy(g,l,.8)[0]['status']=='NOT_VERIFIED_NO_IDENTICAL_PARENT_PREVIEWS'


def test_bridge_rejects_different_frozen_inputs():
    g,l=bridge_fixture();l['split_sha256']='other'
    with pytest.raises(ValueError,match='BRIDGE_INPUT'):a.bridge_greedy(g,l,.8)


def test_q_unseen_zero_and_stop_tie_diagnostics_are_observations():
    row=dict(run_id='q',condition_id='c',dataset='cora',architecture='gcn',dense_training_seed=101,method='tabular_q',
        requested_compression=.3,dense_val_accuracy=.8,achieved_compression=.2,final_widths=[12,32],terminal_reason='max_steps',search_seconds=1.,
        validation_evaluations=21,q_entries=0,q_table_sparsity=1.)
    entries=[]
    for step in range(20):
        entries.append(dict(action='prune:0:1',widths_after=[31-step,32],original_kept_ids=[list(range(31-step)),list(range(32))],
            validation_accuracy=.8,reward=.006,sparsity=.01*(step+1),terminal_reason='max_steps' if step==19 else None,terminal_penalty=-.18 if step==19 else 0.))
    report=dict(q_table=[],trace=entries,training=[dict(episode=0,**e) for e in entries])
    runs,decisions,episodes=a.q_diagnostics([row],{'q':report},dict(adaptive=dict(penalty_lambda=2.,tolerance=.01)))
    assert runs[0]['deployment_selected_unseen']==20 and runs[0]['deployment_stop_tied_maximum']==19
    assert len(decisions)==20 and decisions[1]['stop_legal'] and not decisions[1]['stop_known']
    assert decisions[1]['analytic_immediate_stop_reward']==pytest.approx(-.56)
    assert len(episodes)==1  # Logged episodes are process diagnostics, not statistical n.


def test_analysis_module_has_no_model_or_execution_dependencies():
    tree=ast.parse(Path(a.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
    forbidden=('torch','torch_geometric','gnn_pruning.models','gnn_pruning.pruning','gnn_pruning.training','subprocess','socket')
    assert not any(name and name.startswith(forbidden) for name in imports)
