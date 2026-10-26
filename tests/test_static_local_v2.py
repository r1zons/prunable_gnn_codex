"""Independent coordinate, enumeration and finite-difference static V2 oracles."""
from __future__ import annotations

import copy
import itertools
import random
from fractions import Fraction

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.pruning import static_v2 as v2
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2 as surgery
from test_local_surgery_v2 import make_model, graph, assert_tensors


KINDS = ["gcn", "graphsage"]


def coordinate_score(model, scores, keeps):
    """Enumerate physical coordinates; no production masks/rankings/counts."""
    ids = [set(range(model.in_channels)), *map(set, keeps), set(range(model.out_channels))]
    total = 0.0
    for name, param in model.named_parameters():
        layer = int(name.split(".")[1])
        for coord in itertools.product(*(range(n) for n in param.shape)):
            row = coord[0]
            removed = row not in ids[layer + 1]
            if param.ndim == 2:
                removed = removed or coord[1] not in ids[layer]
            if removed:
                total += float(scores[name][coord])
    return total


def oracle_rankings(model, scores, maximize=False):
    rankings = []
    for layer, conv in enumerate(model.convs[:-1]):
        values = []
        for channel in range(conv.out_channels):
            keeps = [list(range(c.out_channels)) for c in model.convs[:-1]]
            keeps[layer].remove(channel)
            values.append(coordinate_score(model, scores, keeps))
        rankings.append(sorted(range(len(values)), key=lambda c: ((values[c] if maximize else -values[c]), c)))
    return rankings


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bias", [False, True])
def test_all_841_width_pairs_count_matches_physical_slicing(kind, bias):
    cls = GCNNodeClassifier if kind == "gcn" else GraphSAGENodeClassifier
    model = cls(5, 32, 3, num_layers=3, dropout=0.0)
    if not bias:
        for conv in model.convs:
            if kind == "gcn":
                conv.bias = None
            else:
                conv.lin_l.bias = None
    for a, b in itertools.product(range(4, 33), repeat=2):
        expected = (1 if kind == "gcn" else 2) * (5*a + a*b + b*3) + int(bias)*(a+b+3)
        assert v2.parameter_count_for_widths(model, [a, b]) == expected
        physical = model
        for layer, width in enumerate((a, b)):
            if width < 32:
                physical = surgery(physical, layer, list(range(width)))
        assert sum(p.numel() for p in physical.parameters()) == expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("signed", [False, True])
def test_full_mask_unique_union_bias_and_both_sage_branches(kind, signed):
    model = make_model(kind)
    scores = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        values = torch.arange(param.numel(), dtype=torch.float64).reshape(param.shape) + i + 1
        scores[name] = torch.sin(values) if signed else values
    for keeps in [([0, 2, 5, 7], [0, 1, 3, 6]), ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])]:
        assert v2.removed_parameter_score(model, scores, keeps) == pytest.approx(coordinate_score(model, scores, keeps), abs=1e-10)
    assert v2.channel_rankings(model, scores, maximize=signed) == oracle_rankings(model, scores, signed)
    # One shared middle weight is removed by two channel deletions; count once.
    scores = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    key = "convs.1.lin.weight" if kind == "gcn" else "convs.1.lin_r.weight"
    scores[key][2, 1] = 7
    assert v2.removed_parameter_score(model, scores, ([0, 2, 3, 4], [0, 1, 3, 4])) == 7


@pytest.mark.parametrize("kind", KINDS)
def test_static_band_search_exact_fixed_ranking_surrogate_and_lexical_ties(kind):
    model = make_model(kind)
    scores = v2.compute_scores(model, "magnitude")
    rankings = oracle_rankings(model, scores)
    for target in (0.3, 0.5):
        p0 = sum(p.numel() for p in model.parameters())
        factor = 1 if kind == "gcn" else 2
        candidates = []
        for a, b in itertools.product(range(4, 9), range(4, 8)):
            p = factor*(5*a+a*b+b*3)+a+b+3
            if p == p0:
                continue
            s = Fraction(p0-p, p0)
            candidates.append((a, b, s))
        band = [c for c in candidates if abs(c[2]-Fraction(str(target))) <= Fraction(1, 100)]
        gap = min(abs(c[2]-Fraction(str(target))) for c in candidates)
        eligible = band or [c for c in candidates if abs(c[2]-Fraction(str(target))) == gap]
        expected = min(eligible, key=lambda c: (coordinate_score(model, scores, [rankings[0][:c[0]], rankings[1][:c[1]]]), c[0], c[1]))
        plan = v2.plan_static(model, scores, "magnitude", target, tolerance=0.01)
        assert plan.widths == [expected[0], expected[1]]
        assert plan.achieved_sparsity == pytest.approx(float(expected[2]))
        assert plan.fallback == (not bool(band))
        assert plan.target_violation == (abs(float(expected[2])-target) > 0.01)
        assert plan.kept_indices == [sorted(rankings[i][:w]) for i, w in enumerate(plan.widths)]
        physical = v2.apply_static_plan(model, plan)
        assert sum(p.numel() for p in physical.parameters()) == plan.parameters_after
    zero = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    plan = v2.plan_static(model, zero, "magnitude", 0.4, tolerance=1.0)
    assert plan.widths == [4, 4] and plan.kept_indices == [list(range(4)), list(range(4))]
    # Signed GraSP maximizes deletion score, rather than using abs/keep-largest.
    signed = {n: -torch.ones_like(p) for n, p in model.named_parameters()}
    grasp = v2.plan_static(model, signed, "grasp", 0.4, tolerance=1.0)
    expected = min((coordinate_score(model, signed, [list(range(a)), list(range(b))]) * -1, a, b)
                   for a, b in itertools.product(range(4, 9), range(4, 8)) if [a, b] != [8, 7])
    assert grasp.widths == list(expected[1:])


@pytest.mark.parametrize("kind", KINDS)
def test_citeseer_30_unattainable_fallback_is_not_target_hit(kind):
    cls = GCNNodeClassifier if kind == "gcn" else GraphSAGENodeClassifier
    model = cls(3703, 32, 6, num_layers=3, dropout=0.0)
    plan = v2.plan_static(model, {}, "random", 0.3, seed=101)
    assert plan.widths == [22, 32]
    assert plan.fallback and plan.target_violation
    assert plan.target_gap > 0.01
    p0 = (1 if kind == "gcn" else 2)*(3703*32+32*32+32*6)+32+32+6
    p = (1 if kind == "gcn" else 2)*(3703*22+22*32+32*6)+22+32+6
    assert plan.achieved_sparsity == pytest.approx(1-p/p0)


@pytest.mark.parametrize("kind", KINDS)
def test_random_is_reproducible_local_rng_without_importance_or_accuracy_oracle(kind, monkeypatch):
    model = make_model(kind)
    monkeypatch.setattr(v2, "channel_rankings", lambda *args, **kwargs: pytest.fail("Random used saliency oracle"))
    before = random.getstate(), torch.random.get_rng_state().clone()
    a = v2.plan_static(model, {}, "random", 0.3, seed=101)
    b = v2.plan_static(model, {}, "random", 0.3, seed=101)
    assert a == b
    assert random.getstate() == before[0]
    assert torch.equal(torch.random.get_rng_state(), before[1])


def test_analytic_snip_and_grasp_bias_cross_hessian_unused_and_stationary():
    class Quadratic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.bias = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
            self.unused = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    for criterion, expected in [("snip", {"weight": 10., "bias": 0., "unused": 0.}),
                                ("grasp", {"weight": -30., "bias": 5., "unused": 0.})]:
        model = Quadratic()
        w, b = model.weight, model.bias
        loss = (3*w*w + 2*w*b + 2*b*b)/2
        actual = v2.saliency_from_loss(dict(model.named_parameters()), loss, criterion)
        assert {k: float(v) for k, v in actual.items()} == expected
    model = Quadratic()
    loss = ((model.weight-2)**2+(model.bias+1)**2)/2
    scores = v2.saliency_from_loss(dict(model.named_parameters()), loss, "grasp")
    assert all(torch.count_nonzero(v) == 0 for v in scores.values())


@pytest.mark.parametrize("kind", KINDS)
def test_snip_train_only_eval_dropout_off_preserves_source_gradients_and_rng(kind):
    model = make_model(kind)
    model.train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    data = graph()
    data.y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    idx = torch.tensor([0, 2])
    before = {n: p.clone() for n, p in model.state_dict().items()}
    modes = {n: m.training for n, m in model.named_modules()}
    rng = torch.random.get_rng_state().clone()
    actual = v2.compute_scores(model, "snip", data, idx)
    changed = data.clone()
    changed.y[[1, 3, 4, 5, 6]] = 500  # Forbidden labels would make CE fail if used.
    other = v2.compute_scores(model, "snip", changed, idx)
    for name, p in model.named_parameters():
        torch.testing.assert_close(actual[name], other[name], atol=0, rtol=0)
        assert torch.equal(p.grad, torch.ones_like(p))
    assert_tensors(model.state_dict(), before)
    assert modes == {n: m.training for n, m in model.named_modules()}
    assert torch.equal(rng, torch.random.get_rng_state())
    # Independent central difference of one scalar multiplicative parameter gate.
    name = "convs.0.bias" if kind == "gcn" else "convs.0.lin_l.bias"
    eps = 1e-5
    losses = []
    for scale in [1-eps, 1+eps]:
        candidate = copy.deepcopy(model).eval()
        with torch.no_grad():
            dict(candidate.named_parameters())[name][0].mul_(scale)
            losses.append(float(F.cross_entropy(candidate(data)[idx], data.y[idx])))
    assert float(actual[name][0]) == pytest.approx(abs((losses[1]-losses[0])/(2*eps)), rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("kind", KINDS)
def test_grasp_full_graph_hg_matches_finite_difference_gradient(kind):
    model = make_model(kind)
    data = graph()
    data.y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    idx = torch.tensor([0, 2])
    params = dict(model.named_parameters())
    g = torch.autograd.grad(F.cross_entropy(model(data)[idx], data.y[idx]), tuple(params.values()))
    gradients = []
    eps = 1e-5
    for sign in [-1, 1]:
        candidate = copy.deepcopy(model).eval()
        with torch.no_grad():
            for p, direction in zip(candidate.parameters(), g):
                p.add_(sign*eps*direction)
        loss = F.cross_entropy(candidate(data)[idx], data.y[idx])
        gradients.append(torch.autograd.grad(loss, tuple(candidate.parameters())))
    actual = v2.compute_scores(model, "grasp", data, idx)
    for (name, param), lo, hi in zip(params.items(), *gradients):
        expected = -param.detach()*(hi-lo)/(2*eps)
        torch.testing.assert_close(actual[name], expected, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scores_rejected(bad):
    model = make_model("gcn")
    scores = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    next(iter(scores.values())).flatten()[0] = bad
    with pytest.raises(ValueError, match="finite"):
        v2.plan_static(model, scores, "magnitude", 0.3)
