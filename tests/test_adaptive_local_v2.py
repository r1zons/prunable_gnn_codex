"""V2 transition/reward/Bellman/reset oracles with synthetic validation only."""
from __future__ import annotations

import random

import pytest
import torch

from gnn_pruning.pruning import adaptive_v2 as v2
from test_local_surgery_v2 import make_model, assert_tensors


def accuracy(model):
    return 0.8


def make_env(kind="gcn", **kwargs):
    opts = dict(target=0.3, penalty_lambda=2.0)
    opts.update(kwargs)
    return v2.LocalV2Env(make_model(kind, torch.float32), accuracy, **opts)


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
def test_integer_legal_actions_minimum_stop0_and_nonmonotone_repeat(kind):
    env = make_env(kind)
    assert set(env.legal_actions()) == {(0, 1), (0, 2), (0, 4), (1, 1), (1, 2)}
    before = {n: p.clone() for n, p in env.model.state_dict().items()}
    with pytest.raises(ValueError):
        env.step("stop")
    with pytest.raises(ValueError):
        env.step((0, 8))
    assert env.steps == 0 and not env.trace
    assert_tensors(env.model.state_dict(), before)
    for action in [(1, 1), (0, 2), (1, 2)]:
        _, _, done, info = env.step(action)
        assert not done
    assert [c.out_channels for c in env.model.convs[:-1]] == [6, 4]
    assert len(env.original_kept_ids[1]) == 4
    assert "stop" in env.legal_actions()
    assert all(a == "stop" or a[0] == 0 for a in env.legal_actions())


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
def test_reward_stop_and_target_crossing_without_automatic_q_stop(kind):
    env = make_env(kind, target=0.01)
    p0 = sum(p.numel() for p in env.model.parameters())
    _, reward, done, info = env.step((0, 1))
    p = sum(p.numel() for p in env.model.parameters())
    s = 1-p/p0
    assert s > 0.02
    assert not done  # Q can continue beyond target.
    assert reward == pytest.approx(0.6*s)
    assert info["terminal_penalty"] == 0
    steps = env.steps
    _, reward, done, info = env.step("stop")
    expected = -2*max(0, abs(s-0.01)-0.01)
    assert done and env.steps == steps == 1
    assert reward == pytest.approx(expected)
    assert info["terminal_reason"] == "agent_stop"
    assert info["terminal_penalty"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        env.step("stop")


def test_drop_cost_is_dense_relative_and_terminal_penalty_once_on_last_removal():
    dense = make_model("gcn", torch.float32)
    def evaluate(model):
        return 0.9 if model.hidden_channel_dims == [8, 7] else 0.8
    env = v2.LocalV2Env(dense, evaluate, target=0.3, penalty_lambda=3., max_steps=2)
    _, r1, d1, i1 = env.step((0, 1))
    _, r2, d2, i2 = env.step((1, 1))
    assert not d1 and d2
    assert i1["accuracy_drop"] == pytest.approx(0.1)
    assert r1 == pytest.approx(0.6*i1["sparsity"]-0.4*0.1)
    penalty = -3*max(0, abs(i2["sparsity"]-0.3)-0.01)
    assert r2 == pytest.approx(0.6*(i2["sparsity"]-i1["sparsity"])-0.4*0.1+penalty)
    assert i2["terminal_reason"] == "max_steps"
    assert sum(t["terminal_penalty"] != 0 for t in env.trace) == 1
    assert env.legal_actions() == []


def test_no_valid_action_termination_and_greedy_boundary():
    env = make_env()
    env.step((0, 4))
    env.step((1, 1))
    _, _, done, info = env.step((1, 2))
    assert done and info["terminal_reason"] == "no_valid_actions"
    assert env.steps == 3
    greedy = make_env(target=0.01, greedy=True)
    _, _, done, info = greedy.step((0, 1))
    assert done and info["terminal_reason"] == "target_reached"


def test_horizon20_is_number_of_successful_deletions_and_state_has_widths():
    env = make_env()
    state = env.state()
    assert state[0] == "local_v2"
    assert state[1] == (8, 7) and state[2] == 20
    env.step((0, 1))
    state2 = env.state()
    assert state2[1] == (7, 7) and state2[2] == 19
    assert state != state2
    env.step("stop")
    assert env.steps == 1


def test_bellman_max_legal_only_unvisited_zero_terminal_zero():
    table = {"next": {"invalid": 100., "legal": -1.}}
    v2.update_q(table, "s", "a", 0., "next", ["legal"], terminal=False, alpha=1., gamma=0.9)
    assert table["s"]["a"] == pytest.approx(-0.9)
    v2.update_q(table, "s", "a", 2., "next", ["legal", "unvisited"], terminal=False, alpha=1., gamma=0.9)
    assert table["s"]["a"] == 2.
    v2.update_q(table, "s", "a", -3., "next", ["invalid"], terminal=True, alpha=1., gamma=0.9)
    assert table["s"]["a"] == -3.
    action = v2.select_action({"s": {"invalid": 100}}, "s", [(0, 1), "stop"], random.Random(1), epsilon=0.)
    assert action == (0, 1)


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
def test_greedy_scores_all_candidates_and_reports_actual_terminal_model(kind):
    dense = make_model(kind, torch.float32)
    def evaluate(model):
        return 0.8 - 0.02*(15-sum(model.hidden_channel_dims))
    # Independent one-step reward oracle for each legal action at max_steps1.
    p0 = sum(p.numel() for p in dense.parameters())
    multiplier = 1 if kind == "gcn" else 2
    candidates = []
    for action in [(0, 1), (0, 2), (0, 4), (1, 1), (1, 2)]:
        dims = [8, 7]
        dims[action[0]] -= action[1]
        a, b = dims
        p = multiplier*(5*a+a*b+3*b)+a+b+3
        s = 1-p/p0
        r = 0.6*s-0.4*0.02*action[1]-2*max(0, abs(s-0.3)-0.01)
        candidates.append((r, action, dims))
    expected = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]
    result = v2.run_greedy(dense, evaluate, target=0.3, penalty_lambda=2., max_steps=1)
    assert result.model.hidden_channel_dims == expected[2]
    assert result.trace[-1]["reward"] == pytest.approx(expected[0])
    assert result.statistics["candidate_evaluations"] == 5
    assert result.trace[-1]["terminal_reason"] == "max_steps"


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
def test_q_episode_deployment_resets_sparse_stats_and_source_unchanged(kind):
    dense = make_model(kind, torch.float32)
    before = {n: p.clone() for n, p in dense.state_dict().items()}
    result = v2.run_q_learning(dense, accuracy, target=0.01, penalty_lambda=2., episodes=2,
                               max_steps=2, epsilon=0., alpha=1., seed=101)
    resets = [t for t in result.training_trace if t.get("event") == "reset"]
    assert len(resets) == 2
    assert all(t["widths"] == [8, 7] for t in resets)
    assert result.trace[0]["widths_before"] == [8, 7]
    assert result.trace[-1]["terminal_reason"] in {"agent_stop", "max_steps", "no_valid_actions"}
    assert result.model.hidden_channel_dims == result.trace[-1]["widths_after"]
    assert result.statistics["unique_states"] >= 2
    assert result.statistics["q_entries"] == sum(len(row) for row in result.q_table.values())
    assert 0 <= result.statistics["q_table_sparsity"] <= 1
    assert_tensors(dense.state_dict(), before)
    other = v2.run_q_learning(dense, accuracy, target=0.01, penalty_lambda=2., episodes=2,
                              max_steps=2, epsilon=0., alpha=1., seed=101)
    assert result.q_table == other.q_table
    assert_tensors(result.model.state_dict(), other.model.state_dict())


@pytest.mark.parametrize("action", [(0, 1.0), (False, 1), (0, True), (0, 3), (2, 1)])
def test_external_noninteger_or_unsupported_actions_are_rejected_without_mutation(action):
    env = make_env()
    before = {n: p.clone() for n, p in env.model.state_dict().items()}
    with pytest.raises(ValueError):
        env.step(action)
    assert env.steps == 0 and not env.trace
    assert_tensors(env.model.state_dict(), before)


def test_initial_no_actions_pays_terminal_penalty_once_and_is_explicitly_unpruned():
    from gnn_pruning.models import GCNNodeClassifier
    model = GCNNodeClassifier(5, [4, 4], 3, num_layers=3, dropout=0.)
    env = v2.LocalV2Env(model, accuracy, target=0.3, penalty_lambda=2.)
    assert env.terminal and env.steps == 0
    assert env.trace[-1]["terminal_reason"] == "no_valid_actions"
    assert env.trace[-1]["terminal_penalty"] == pytest.approx(-2*(0.3-0.01))
    assert len(env.trace) == 1
    assert env.legal_actions() == []


def test_exactly_twenty_successful_removals_then_forced_termination():
    from gnn_pruning.models import GCNNodeClassifier
    model = GCNNodeClassifier(5, 32, 3, num_layers=3, dropout=0.)
    env = v2.LocalV2Env(model, accuracy, target=0.3, penalty_lambda=2.)
    assert len(env.legal_actions()) == 8
    for i in range(20):
        _, _, done, info = env.step((0, 1))
        assert done == (i == 19)
    assert env.steps == 20 and env.model.hidden_channel_dims == [12, 32]
    assert info["terminal_reason"] == "max_steps"
    assert env.legal_actions() == []
