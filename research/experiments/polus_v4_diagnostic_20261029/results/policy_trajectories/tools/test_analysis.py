"""Synthetic data tests of diagnostic definitions, without model execution."""
import copy
import csv
import unittest
import analyze as a


class Diagnostics(unittest.TestCase):
    def decision(self, values, action, known, ties):
        return dict(legal_q_values=values, action=action, epsilon=0., mode='exploit',
                    selected_action_unseen=action not in known,
                    unseen_legal_actions=sum(k not in known for k in values), equal_max_count=ties)

    def test_known_zero_is_not_unseen(self):
        d = self.decision({'a': 0., 'b': 0.}, 'a', {'a': 0.}, 2)
        self.assertEqual(a.winner(d, {'a': 0.})[1], 'ordered_tie_known')

    def test_zero_prior_can_beat_negative_known_value(self):
        d = self.decision({'a': -.1, 'b': 0.}, 'b', {'a': -.1}, 1)
        self.assertEqual(a.winner(d, {'a': -.1})[1], 'unique_max_unseen_prior')

    def test_deterministic_tie_requires_first_legal_max(self):
        d = self.decision({'a': .1, 'b': .1}, 'b', {'a': .1, 'b': .1}, 2)
        with self.assertRaises(AssertionError):
            a.winner(d, {'a': .1, 'b': .1})
        self.assertEqual(a.winner(d, {'a': .1, 'b': .1}, True)[1], 'seeded_tie_known')

    def test_reject_nonmaximal_and_altered_q(self):
        d = self.decision({'a': .2, 'b': .1}, 'b', {'a': .2, 'b': .1}, 1)
        with self.assertRaises(AssertionError):
            a.winner(d, {'a': .2, 'b': .1})
        d['action'] = 'a'
        with self.assertRaises(AssertionError):
            a.winner(d, {'a': .3, 'b': .1})

    def test_training_alias_index_excludes_deployment(self):
        base = dict(phase='training', state=['local_v2', [2, 2], 10, 80, False],
                    original_kept_ids=[[0, 1], [0, 1]], action='prune:0:1', reward=.1)
        changed = copy.deepcopy(base)
        changed['original_kept_ids'][0] = [0, 2]
        changed['reward'] = -.2
        dep = copy.deepcopy(changed)
        dep['phase'] = 'deployment'
        dep['original_kept_ids'][0] = [0, 3]
        states, pairs = a.train_index([base, changed, dep])
        self.assertEqual([len(m) for m in states.values()], [2])
        self.assertEqual([sum(m.values()) for m in states.values()], [2])
        pair = next(iter(pairs.values()))
        self.assertEqual(a.variation(pair['rewards']), (True, True))
        self.assertNotIn(a.mask_key(dep['original_kept_ids']), pair['masks'])

    def test_tiny_float_variation_is_not_material(self):
        self.assertEqual(a.variation([.1, .1+1e-16]), (True, False))
        self.assertEqual(a.variation([.1, .1001]), (True, True))

    def test_equal_widths_can_have_different_masks(self):
        q = [dict(action='prune:0:1', original_kept_ids=[[0, 1], [0, 1]])]
        g = [dict(action='prune:0:1', original_kept_ids=[[0, 2], [0, 1]])]
        self.assertEqual(a.first_divergence(q, g), (None, 1))

    def test_prefix_end_is_a_divergence(self):
        e = dict(action='prune:0:1', original_kept_ids=[[0], [0, 1]])
        self.assertEqual(a.first_divergence([e], [e]), (None, None))
        self.assertEqual(a.first_divergence([e, e], [e]), (2, 2))

    def test_archived_matrix_and_step_joins(self):
        rows = list(csv.DictReader((a.HERE/'run_diagnostics.csv').open()))
        steps = list(csv.DictReader((a.HERE/'deployment_steps.csv').open()))
        expected = {(m,t,v,str(s)) for m in ('gcn','graphsage') for t in ('0.3','0.5')
                    for v in a.VARIANTS for s in (101,5001,5002)}
        self.assertEqual({(r['model'],r['target'],r['variant'],r['policy_seed']) for r in rows}, expected)
        self.assertEqual(len(rows), 84)
        self.assertEqual(len(steps), sum(int(r['deployment_steps']) for r in rows))
        old = {r['run_id']: r for r in a.read(a.PRIOR/'runs.json')}
        for row in rows:
            path = [s for s in steps if s['run_id'] == row['run_id']]
            self.assertEqual([int(s['step']) for s in path], list(range(1,len(path)+1)))
            self.assertEqual(a.json.loads(path[-1]['kept_ids_after']), a.json.loads(old[row['run_id']]['retained_ids']))

    def test_case_shared_state_and_terminal_update_identity(self):
        case = a.read(a.HERE/'case_study.json')
        rows = list(csv.DictReader((a.HERE/'run_diagnostics.csv').open()))
        eligible = [r['run_id'] for r in rows if r['model']=='graphsage' and r['target']=='0.3'
                    and r['greedy_variant'].startswith('greedy_v4') and r['same_final_mask']=='False'
                    and r['reward_matched']=='True']
        self.assertEqual(eligible, [case['run']['run_id']])
        q = case['q_steps'][-1]
        g = case['greedy_steps'][-1]
        self.assertEqual(q['kept_ids_before'], case['greedy_steps'][-2]['original_kept_ids'])
        updates = list(csv.DictReader((a.HERE/'case_terminal_updates.csv').open()))
        self.assertTrue(all(u['terminal']=='True' and float(u['bootstrap'])==0 for u in updates))
        for action, reward in [('prune:1:1', g['reward']), ('prune:1:2', q['reward'])]:
            us = [u for u in updates if u['action'] == action]
            self.assertTrue(all(float(u['reward'])==reward for u in us))
            self.assertTrue(all(a.json.loads(u['kept_ids'])==q['kept_ids_before'] for u in us))
            self.assertAlmostEqual(q['legal_q_values'][action], reward*(1-.9**len(us)), places=14)
        self.assertGreater(q['legal_q_values']['prune:1:2'],q['legal_q_values']['prune:1:1'])
        self.assertLess(q['reward'],g['reward'])

    def test_prior_manifest_and_alias_accounting(self):
        a.verify_prior()
        rows = list(csv.DictReader((a.HERE/'run_diagnostics.csv').open()))
        for row in rows:
            self.assertLessEqual(int(row['train_material_conflicts']),int(row['train_exact_conflicts']))
            self.assertEqual(int(row['train_material_conflicts']),int(row['train_conflicts_multiple_masks']))
            self.assertEqual(int(row['within_mask_material_conflicts']),0)


if __name__ == '__main__':
    unittest.main()
