"""Extend the pinned V4 audit using recorded observations only; never evaluate a model.

Full training state/pair indexes stay external. CSV deployment rows preserve every
legal Q estimate and known flag; unknown immediate outcomes are never imputed.
"""
from __future__ import annotations
import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PRIOR = HERE.parent
spec = importlib.util.spec_from_file_location('prior_audit', PRIOR/'tools/verify_and_aggregate.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
VARIANTS = ['q_v4_menu12', 'q_v4_percent', 'q_v4_300', 'q_v4_decay300',
            'q_v4_random_ties', 'q_v4_incremental_drop', 'q_v4_gamma0']
TOL = 1e-10
read, save, sha, close = audit.read, audit.save, audit.sha, audit.close


def compact(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: compact(v) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()} for row in rows)


def mask_key(ids):
    return tuple(map(tuple, ids))


def variation(values):
    return len(set(values)) > 1, max(values)-min(values) > TOL


def train_index(decisions):
    states, pairs = {}, {}
    for d in decisions:
        if d['phase'] != 'training':
            continue
        s, mask, action = audit.state_key(d['state']), mask_key(d['original_kept_ids']), d['action']
        state = states.setdefault(s, collections.Counter())
        state[mask] += 1
        p = pairs.setdefault((s, action), {'masks': collections.Counter(), 'rewards': [],
                                        'mask_rewards': collections.defaultdict(list)})
        p['masks'][mask] += 1
        p['rewards'].append(d['reward'])
        p['mask_rewards'][mask].append(d['reward'])
    return states, pairs


def winner(d, row, random_ties=False):
    """Verify recorded deployment argmax; membership, not a fabricated score."""
    values = d['legal_q_values']
    assert d['epsilon'] == 0 and d['mode'] == 'exploit'
    for action, value in values.items():
        close(value, row.get(action, 0.))
    maxima = [a for a, v in values.items() if v == max(values.values())]
    assert d['action'] in maxima
    assert d['equal_max_count'] == len(maxima)
    known = d['action'] in row
    assert d['selected_action_unseen'] == (not known)
    assert d['unseen_legal_actions'] == sum(a not in row for a in values)
    if not random_ties:
        assert d['action'] == maxima[0]
    mechanism = 'unique_max' if len(maxima) == 1 else 'seeded_tie' if random_ties else 'ordered_tie'
    return maxima, mechanism + ('_known' if known else '_unseen_prior')


def first_divergence(qevents, gevents):
    """One-based first action/mask divergence along aligned dense-start prefixes."""
    action = mask = None
    for i in range(max(len(qevents), len(gevents))):
        q = qevents[i] if i < len(qevents) else None
        g = gevents[i] if i < len(gevents) else None
        if action is None and (q is None or g is None or q['action'] != g['action']):
            action = i+1
        if mask is None and (q is None or g is None or q['original_kept_ids'] != g['original_kept_ids']):
            mask = i+1
    return action, mask


def verify_prior():
    manifest = read(PRIOR/'analysis_manifest.json')
    for name, item in manifest['files'].items():
        assert sha(PRIOR/name) == item['sha256'], ('Prior analysis changed', name)
    return {name: sha(PRIOR/name) for name in ('REPORT.md', 'runs.csv', 'runs.json',
            'deployment_decisions.csv', 'reward_and_exploration.csv', 'analysis_manifest.json')}


def analyze(collection, external, output):
    external.mkdir(exist_ok=False, parents=True)
    output.mkdir(exist_ok=True, parents=True)
    prior_hashes = verify_prior()
    provenance = read(PRIOR/'provenance.json')
    assert sha(collection/'v4-primary-results.tar.gz') == provenance['archive_sha256']
    root = collection/'polus'
    assert sha(root/'collection/inventory.json') == provenance['inventory_sha256']
    inventory = read(root/'collection/inventory.json')
    for entry in inventory:
        p = root/entry['path']
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
    previous = read(PRIOR/'runs.json')
    references = read(PRIOR/'baseline_runs.json')
    anchors = {r['run_id']: r for r in read(PRIOR/'verification.json')['runs']}
    old_decisions = collections.defaultdict(list)
    with (PRIOR/'deployment_decisions.csv').open() as f:
        for row in csv.DictReader(f):
            old_decisions[row['run_id']].append(row)
    old_rewards = {r['run_id']: r for r in csv.DictReader((PRIOR/'reward_and_exploration.csv').open())}
    artifacts = {}
    paths = list((root/'campaign-primary/outputs').glob('*/*/completed.json'))
    oldroot = collection.parent/'local-structured-v3-collection-20261027T054655Z/polus/polus-batch-final/outputs'
    paths += list(oldroot.glob('*greedy_reachable_v3-p101/attempt-*/completed.json'))
    for marker in paths:
        folder = marker.parent
        done = read(marker)
        run_id = done['run_id']
        assert sha(marker) == anchors[run_id]['completed_sha256']
        for name, expected in done['hashes'].items():
            assert sha(folder/name) == expected
        artifacts[run_id] = dict(folder=folder, metrics=read(folder/'metrics.json'),
                                trace=read(folder/'trace.json'), cfg=read(folder/'config.json'))
    assert len(artifacts) == 96
    controls = [r for r in previous+references if r['variant'].startswith('greedy') and r['platform'] == 'polus']
    primary = [r for r in previous if r['variant'].startswith('q_')]
    assert len(primary) == 84 and len(controls) == 12
    steps, runs, state_rows, pair_rows, checks = [], [], [], [], []
    for n, r in enumerate(primary):
        run_id = r['run_id']
        artifact = artifacts[run_id]
        folder, metrics, trace, cfg = (artifact[k] for k in ('folder', 'metrics', 'trace', 'cfg'))
        decisions, qrows = read(folder/'decisions.json'), read(folder/'q_table.json')
        # Reuse the earlier independent equations for all recorded Q/RNG updates,
        # legality, rewards and snapshots. No ranking/evaluation callback is used.
        checked, _, deployed, compact_steps, snapshots = audit.audit_learning(
            cfg, metrics, trace, qrows, decisions, folder)
        assert checked['observed_aliased_states'] == r['observed_aliased_states']
        assert checked['state_action_reward_conflicts'] == r['state_action_reward_conflicts']
        assert int(old_rewards[run_id]['observed_aliased_states']) == r['observed_aliased_states']
        close(float(old_rewards[run_id]['deployment_discounted_return']), r['deployment_discounted_return'])
        table = audit.normalize_q(qrows)
        states, pairs = train_index(decisions)
        deployment = [d for d in decisions if d['phase'] == 'deployment']
        assert len(deployment) == len(trace['deployment']) == len(old_decisions[run_id])
        counts = collections.Counter()
        ident = {k: r[k] for k in ('run_id', 'model', 'target', 'variant', 'policy_seed')}
        for s, masks in states.items():
            state_rows.append(dict(**ident, state=s, observations=sum(masks.values()),
                                   masks=len(masks), mask_counts=[dict(ids=m, count=c) for m, c in masks.items()]))
        for (s, action), p in pairs.items():
            exact, material = variation(p['rewards'])
            same_mask_variation = sum(variation(v)[1] for v in p['mask_rewards'].values())
            pair_rows.append(dict(**ident, state=s, action=action, observations=len(p['rewards']),
                masks=len(p['masks']), exact_conflict=exact, material_conflict=material,
                reward_min=min(p['rewards']), reward_max=max(p['rewards']),
                within_mask_material_conflicts=same_mask_variation))
            counts['train_material_conflicts'] += material
            counts['train_exact_conflicts'] += exact
            counts['train_conflict_observations'] += len(p['rewards']) if material else 0
            counts['train_conflicts_multiple_masks'] += material and len(p['masks']) > 1
            counts['within_mask_material_conflicts'] += same_mask_variation
        counts['train_states'] = len(states)
        counts['train_aliased_states'] = sum(len(m) > 1 for m in states.values())
        counts['train_state_action_pairs'] = len(pairs)
        counts['train_multimask_pairs'] = sum(len(p['masks']) > 1 for p in pairs.values())
        counts['train_observations'] = sum(sum(m.values()) for m in states.values())
        counts['train_aliased_observations'] = sum(sum(m.values()) for m in states.values() if len(m) > 1)
        local_steps = []
        accuracy = float(audit.torch.tensor(r['dense_validation_accuracy'], dtype=audit.torch.float32))
        for i, (d, e, old) in enumerate(zip(deployment, trace['deployment'], old_decisions[run_id])):
            assert old['action'] == d['action'] and int(old['step']) == i
            close(float(old['selected_q']), d['legal_q_values'][d['action']])
            s, mask = audit.state_key(d['state']), mask_key(d['original_kept_ids'])
            row = table.get(s, {})
            legal = audit.legal(r['model'], r['target'], audit.PLANS[r['variant']]['menu'], tuple(s[1]), s[2])
            assert tuple(d['legal_q_values']) == legal
            maxima, reason = winner(d, row, r['variant'] == 'q_v4_random_ties')
            masks = states.get(s, {})
            pair = pairs.get((s, d['action']), {'masks': {}, 'rewards': []})
            state_unseen, novel_mask = not masks, bool(masks) and mask not in masks
            transfer = d['action'] in row and mask not in pair['masks']
            conflict = bool(pair['rewards']) and variation(pair['rewards'])[1]
            known = {a: a in row for a in legal}
            legal_training = {}
            for action in legal:
                ap = pairs.get((s, action), {'masks': {}, 'rewards': []})
                legal_training[action] = dict(observations=len(ap['rewards']), masks=len(ap['masks']),
                    current_mask_observations=ap['masks'].get(mask, 0),
                    reward_min=min(ap['rewards'], default=None), reward_max=max(ap['rewards'], default=None))
            layer, size = map(int, d['action'].split(':')[1:])
            removed = [sorted(set(before)-set(after)) for before, after in zip(d['original_kept_ids'], e['original_kept_ids'])]
            step = dict(**ident, step=i+1, state=d['state'], state_widths=s[1], remaining_actions=s[2],
                accuracy_bucket=s[3], legal_q_values=d['legal_q_values'], legal_action_known=known,
                legal_action_training=legal_training,
                action=d['action'], layer=layer, deletion_size=size, selected_q=d['legal_q_values'][d['action']],
                selected_known=d['action'] in row, maxima=maxima, why_won=reason,
                widths_before=e['widths_before'], widths_after=e['widths_after'],
                kept_ids_before=d['original_kept_ids'], kept_ids_after=e['original_kept_ids'], removed_ids=removed,
                reward=e['reward'], validation_accuracy_before=accuracy,
                validation_accuracy_after=e['validation_accuracy'], achieved_compression=e['sparsity'],
                parameters_after=audit.count(r['model'], e['widths_after']),
                terminal_reason=e['terminal_reason'], terminal_penalty=e['terminal_penalty'],
                state_unseen=state_unseen, training_state_masks=len(masks), training_state_observations=sum(masks.values()),
                training_current_mask_observations=masks.get(mask, 0), novel_mask_at_known_state=novel_mask,
                selected_pair_training_masks=len(pair['masks']), selected_pair_training_observations=sum(pair['masks'].values()),
                selected_pair_current_mask_observations=pair['masks'].get(mask, 0), cross_mask_value_transfer=transfer,
                selected_pair_reward_conflict=conflict,
                selected_pair_reward_min=min(pair['rewards'], default=None),
                selected_pair_reward_max=max(pair['rewards'], default=None))
            counts['deployment_training_alias_exposure'] += len(masks) > 1
            counts['deployment_novel_mask'] += novel_mask
            counts['deployment_cross_mask_value_transfer'] += transfer
            counts['deployment_conflict_exposure'] += conflict
            counts['deployment_state_unseen'] += state_unseen
            counts['deployment_selected_unseen'] += not step['selected_known']
            counts[reason] += 1
            local_steps.append(step)
            accuracy = e['validation_accuracy']
        steps.extend(local_steps)
        gv = ('greedy_v4_menu12' if r['variant'] == 'q_v4_menu12' else
              'greedy_v4_percent' if r['variant'] == 'q_v4_percent' else 'greedy_reachable_v3')
        g = next(g for g in controls if g['model'] == r['model'] and g['target'] == r['target'] and g['variant'] == gv)
        for key in ('platform', 'training_seed', 'graph_tensor_sha256', 'split_sha256', 'dense_checkpoint_sha256'):
            assert r[key] == g[key]
        assert audit.PLANS[r['variant']]['menu'] == audit.PLANS[g['variant']]['menu']
        gevents = artifacts[g['run_id']]['trace']['deployment']
        qevents = trace['deployment']
        first_action, first_mask = first_divergence(qevents, gevents)
        qi, gi = json.loads(r['retained_ids']), json.loads(g['retained_ids'])
        common = [sorted(set(a) & set(b)) for a, b in zip(qi, gi)]
        symdiff = [sorted(set(a) ^ set(b)) for a, b in zip(qi, gi)]
        pair = dict(**r, **counts, greedy_run_id=g['run_id'], greedy_variant=g['variant'],
            reward_matched=r['variant'] != 'q_v4_incremental_drop', first_action_divergence=first_action,
            first_mask_divergence=first_mask, same_final_widths=(r['width0'], r['width1']) == (g['width0'], g['width1']),
            same_final_mask=qi == gi, retained_symmetric_difference=symdiff, retained_intersection=common,
            greedy_widths=[g['width0'], g['width1']], greedy_retained_ids=gi,
            greedy_compression=g['achieved_sparsity'], greedy_accuracy=g['validation_accuracy'],
            greedy_macro_f1=g['validation_macro_f1'], greedy_search_seconds=g['search_seconds'],
            greedy_evaluations=g['validation_evaluations'], greedy_deployment_steps=len(gevents),
            accuracy_delta_pp=100*(r['validation_accuracy']-g['validation_accuracy']),
            f1_delta_pp=100*(r['validation_macro_f1']-g['validation_macro_f1']),
            compression_delta_pp=100*(r['achieved_sparsity']-g['achieved_sparsity']),
            search_ratio=r['search_seconds']/g['search_seconds'])
        for key in ('deployment_training_alias_exposure', 'deployment_novel_mask', 'deployment_cross_mask_value_transfer',
                    'deployment_conflict_exposure', 'deployment_selected_unseen'):
            pair[key+'_fraction'] = counts[key]/len(local_steps)
        if first_action is not None and first_action <= min(len(qevents), len(gevents)):
            j = first_action-1
            qb = deployment[j]['original_kept_ids']
            gb = gevents[j-1]['original_kept_ids'] if j else [list(range(32)), list(range(32))]
            assert qb == gb, 'First action divergence must have a common physical predecessor'
            ga = gevents[j]['action']
            pair['q_value_for_greedy_first_divergent_action'] = deployment[j]['legal_q_values'][ga]
            pair['q_first_divergent_selected_value'] = local_steps[j]['selected_q']
        runs.append(pair)
        checks.append(dict(run_id=run_id, status='PASS', transitions=checked['transitions'], snapshots=len(snapshots)))
        print(f'{n+1}/84 {run_id}', flush=True)
    # Structural, quality-blind illustrated-case rule declared before analysis.
    candidates = [r for r in runs if r['model'] == 'graphsage' and r['target'] == .3 and
                  r['greedy_variant'].startswith('greedy_v4') and not r['same_final_mask'] and r['reward_matched']]
    case = min(candidates, key=lambda r: ([101,5001,5002].index(r['policy_seed']), VARIANTS.index(r['variant'])))
    groups = []
    for model in ('gcn', 'graphsage'):
        for target in (.3, .5):
            for variant in VARIANTS:
                rs = [r for r in runs if (r['model'],r['target'],r['variant']) == (model,target,variant)]
                assert len(rs) == 3
                groups.append(dict(model=model,target=target,variant=variant,policy_seeds=[r['policy_seed'] for r in rs],
                    reward_matched=rs[0]['reward_matched'],
                    widths=[str(r['width0'])+'/'+str(r['width1']) for r in rs],
                    accuracy_delta_pp=[r['accuracy_delta_pp'] for r in rs], f1_delta_pp=[r['f1_delta_pp'] for r in rs],
                    same_final_masks=sum(r['same_final_mask'] for r in rs),
                    same_final_widths=sum(r['same_final_widths'] for r in rs),
                    first_action_divergences=[r['first_action_divergence'] for r in rs],
                    training_aliased_states=[r['train_aliased_states'] for r in rs],
                    training_conflicting_pairs=[r['train_material_conflicts'] for r in rs],
                    deployment_alias_fraction=[r['deployment_training_alias_exposure_fraction'] for r in rs],
                    cross_mask_value_transfer_fraction=[r['deployment_cross_mask_value_transfer_fraction'] for r in rs],
                    search_ratio=[r['search_ratio'] for r in rs]))
    write_csv(output/'deployment_steps.csv', steps)
    write_csv(output/'run_diagnostics.csv', runs)
    write_csv(output/'summary.csv', groups)
    write_csv(external/'training_states.csv', state_rows)
    write_csv(external/'training_state_actions.csv', pair_rows)
    # Deployment-relevant alias evidence is compact; complete training indexes external.
    exposed = {(s['run_id'], audit.state_key(s['state'])) for s in steps}
    relevant_states = [r for r in state_rows if (r['run_id'], audit.state_key(r['state'])) in exposed and r['masks'] > 1]
    relevant_pairs = [r for r in pair_rows if (r['run_id'], audit.state_key(r['state'])) in exposed and r['material_conflict']]
    write_csv(output/'deployment_alias_states.csv', relevant_states)
    write_csv(output/'deployment_reward_conflicts.csv', relevant_pairs)
    save(output/'case_study.json', dict(selection_rule='GraphSAGE 30%; new reward/menu-matched Greedy; different final mask; seed order 101,5001,5002 then menu12,percent; no quality criterion',
        run=case, q_steps=[s for s in steps if s['run_id'] == case['run_id']],
        greedy_steps=artifacts[case['greedy_run_id']]['trace']['deployment']))
    case_decisions = read(artifacts[case['run_id']]['folder']/'decisions.json')
    terminal_state = next(s['state'] for s in reversed(steps) if s['run_id'] == case['run_id'])
    updates = [dict(episode=d['episode'], state=d['state'], action=d['action'],
                    kept_ids=d['original_kept_ids'], reward=d['reward'], terminal=d['terminal'],
                    bootstrap=d['bootstrap'], q_before=d['q_before'], q_after=d['q_after'])
               for d in case_decisions if d['phase'] == 'training' and d['state'] == terminal_state]
    write_csv(output/'case_terminal_updates.csv', updates)
    verify_prior()
    save(output/'verification.json', dict(status='PASS', primary_q_runs=len(runs), primary_new_greedy_runs=8,
        reused_native_greedy_runs=4, deployment_steps=len(steps), training_decisions=sum(r['train_observations'] for r in runs),
        archive_sha256=provenance['archive_sha256'], inventory_files_rehashed=len(inventory),
        prior_files_sha256=prior_hashes, audit_script_sha256=sha(PRIOR/'tools/verify_and_aggregate.py'),
        original_scientific_source=provenance['source_commit'], runs=checks,
        external_indexes={name:dict(sha256=sha(external/name),bytes=(external/name).stat().st_size)
                          for name in ('training_states.csv','training_state_actions.csv')},
        external_root=str(external), reward_conflict_absolute_tolerance=TOL,
        no_model_evaluation=True, no_new_experiment=True, no_algorithm_change=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--external', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    analyze(args.collection.resolve(), args.external.resolve(), args.output.resolve())
