"""Independent Local Surgery oracles: no production slicing/count/scoring helpers.

All graphs and weights are synthetic. No datasets, fit, fine-tuning or experiments.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier, build_model
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2 as surgery


MODELS = {"gcn": GCNNodeClassifier, "graphsage": GraphSAGENodeClassifier}
KEEP_CASES = [([0, 2, 5, 7], [0, 1, 3, 6]), ([0, 1, 2, 4, 6], [1, 2, 3, 4, 5, 6])]


def make_model(kind, dtype=torch.float64, bias=True):
    model = MODELS[kind](5, [8, 7], 3, num_layers=3, dropout=0.6).to(dtype=dtype)
    if not bias:
        for conv in model.convs:
            if kind == "gcn":
                conv.bias = None
            else:
                conv.lin_l.bias = None
    # Nonzero, distinct weights/biases; independent of initializer and RNG order.
    with torch.no_grad():
        for i, (name, param) in enumerate(model.named_parameters()):
            values = torch.arange(param.numel(), dtype=dtype).reshape(param.shape)
            values = torch.sin(values * 0.37 + i + 0.3) * 0.4
            if "bias" in name:
                values = values + 0.7
            param.copy_(values)
    return model.eval()


def graph(dtype=torch.float64, device="cpu"):
    # Includes an isolated node, asymmetric degrees and bidirectional entries.
    return Data(
        x=(torch.arange(35, dtype=dtype).reshape(7, 5) / 19 - 0.5).to(device),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 4, 4, 5, 0, 3], [1, 0, 2, 1, 4, 2, 5, 4, 3, 0]],
            dtype=torch.long, device=device,
        ),
    )


def full_weight_mask_logits(model, data, keeps):
    """Dense reference: zero union of removed rows, columns AND hidden bias."""
    masked = copy.deepcopy(model).eval()
    masks = [torch.ones(model.in_channels, dtype=torch.bool, device=data.x.device)]
    for width, keep in zip([8, 7], keeps):
        mask = torch.zeros(width, dtype=torch.bool, device=data.x.device)
        mask[keep] = True
        masks.append(mask)
    masks.append(torch.ones(model.out_channels, dtype=torch.bool, device=data.x.device))
    with torch.no_grad():
        for i, conv in enumerate(masked.convs):
            union = masks[i + 1][:, None] & masks[i][None, :]
            if isinstance(conv, GCNConv):
                conv.lin.weight.mul_(union)
                if conv.bias is not None:
                    conv.bias.mul_(masks[i + 1])
            else:
                conv.lin_l.weight.mul_(union)
                conv.lin_r.weight.mul_(union)
                if conv.lin_l.bias is not None:
                    conv.lin_l.bias.mul_(masks[i + 1])
        return masked(data)


def activation_mask_logits(model, data, keeps):
    """Second oracle: dense convs with hidden activation gates, no weight slicing."""
    x = data.x
    with torch.no_grad():
        for i, conv in enumerate(model.convs):
            x = conv(x, data.edge_index)
            if i < len(keeps):
                gate = torch.zeros_like(x)
                gate[:, keeps[i]] = 1
                x = torch.relu(x) * gate
    return x


def expected_tensors(model, keeps):
    """Explicit dense submatrices, including both SAGE branches and biases."""
    ids = [list(range(5)), *keeps, list(range(3))]
    expected = {}
    for i, conv in enumerate(model.convs):
        rows, cols = ids[i + 1], ids[i]
        if isinstance(conv, GCNConv):
            expected[f"convs.{i}.lin.weight"] = conv.lin.weight.detach()[rows][:, cols]
            if conv.bias is not None:
                expected[f"convs.{i}.bias"] = conv.bias.detach()[rows]
        else:
            for branch in ("lin_l", "lin_r"):
                linear = getattr(conv, branch)
                expected[f"convs.{i}.{branch}.weight"] = linear.weight.detach()[rows][:, cols]
            if conv.lin_l.bias is not None:
                expected[f"convs.{i}.lin_l.bias"] = conv.lin_l.bias.detach()[rows]
    return expected


def assert_tensors(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.parametrize("kind", MODELS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("keeps", KEEP_CASES)
def test_physical_tensors_count_and_two_independent_logit_oracles(kind, dtype, bias, order, keeps):
    dense = make_model(kind, dtype, bias)
    original = {name: value.clone() for name, value in dense.state_dict().items()}
    data = graph(dtype)
    original_x, original_edges = data.x.clone(), data.edge_index.clone()
    pruned = dense
    for layer in order:
        # Reverse input order must still canonicalize retained original IDs.
        pruned = surgery(pruned, layer, list(reversed(keeps[layer])))
    a, b = map(len, keeps)
    factor = 1 if kind == "gcn" else 2
    p0 = factor * (5 * 8 + 8 * 7 + 7 * 3) + int(bias) * (8 + 7 + 3)
    expected_count = factor * (5 * a + a * b + b * 3) + int(bias) * (a + b + 3)
    assert sum(p.numel() for p in dense.parameters()) == p0
    assert sum(p.numel() for p in pruned.parameters()) == expected_count < p0
    assert [conv.out_channels for conv in pruned.convs] == [a, b, 3]
    assert pruned.hidden_channel_dims == [a, b]
    assert pruned.hidden_channels == a
    assert pruned.in_channels == 5 and pruned.out_channels == 3
    assert all(p.dtype == dtype for p in pruned.parameters())
    assert_tensors(pruned.state_dict(), expected_tensors(dense, keeps))
    assert_tensors(dense.state_dict(), original)
    for name, param in pruned.named_parameters():
        assert param.data_ptr() != dict(dense.named_parameters())[name].data_ptr()
    tol = dict(atol=1e-12, rtol=1e-10) if dtype == torch.float64 else dict(atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        actual = pruned(data)
    assert actual.shape == (7, 3)
    torch.testing.assert_close(actual, full_weight_mask_logits(dense, data, keeps), **tol)
    torch.testing.assert_close(actual, activation_mask_logits(dense, data, keeps), **tol)
    assert torch.equal(data.x, original_x) and torch.equal(data.edge_index, original_edges)


@pytest.mark.parametrize("kind", MODELS)
def test_repeated_nonmonotone_removal_tracks_original_ids_and_order_commutes(kind):
    dense = make_model(kind, torch.float32)
    # layer 1 -> 0 -> 1, where the last call uses CURRENT positions for original IDs.
    first = surgery(dense, 1, [0, 1, 2, 3, 5, 6])
    second = surgery(first, 0, [0, 2, 5, 7])
    final = surgery(second, 1, [0, 1, 3, 5])  # original IDs: [0,1,3,6]
    other = surgery(surgery(dense, 0, [0, 2, 5, 7]), 1, [0, 1, 3, 6])
    assert_tensors(final.state_dict(), expected_tensors(dense, KEEP_CASES[0]))
    assert_tensors(final.state_dict(), other.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(final(graph(torch.float32)), other(graph(torch.float32)), rtol=0, atol=0)


@pytest.mark.parametrize("kind", MODELS)
def test_strict_disk_checkpoint_roundtrip_nonuniform_widths(kind, tmp_path):
    dense = make_model(kind, torch.float32)
    keeps = KEEP_CASES[1]
    pruned = surgery(surgery(dense, 1, keeps[1]), 0, keeps[0])
    checkpoint = tmp_path / "local_v2.pt"
    torch.save({"architecture": pruned.export_architecture_config(), "state_dict": pruned.state_dict(),
                "original_kept_ids": keeps}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(saved["architecture"])
    name = config.pop("name")
    rebuilt = build_model(name, **config).eval()
    rebuilt.load_state_dict(saved["state_dict"], strict=True)
    assert config["hidden_channels"] == [5, 6]
    assert saved["original_kept_ids"] == keeps
    assert_tensors(rebuilt.state_dict(), expected_tensors(dense, keeps))
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(graph(torch.float32)), pruned(graph(torch.float32)), rtol=0, atol=0)


@pytest.mark.parametrize("kind", MODELS)
def test_preserves_device_dtype_modes_parameter_flags_and_pyg_linear_metadata(kind):
    model = make_model(kind)
    model.train()
    model.convs[0].eval()  # Intentionally mixed modes, including PyG Linear children.
    for i, param in enumerate(model.parameters()):
        param.requires_grad_(i % 2 == 0)
    pruned = surgery(model, 0, [0, 2, 5, 7])
    assert {n: m.training for n, m in pruned.named_modules()} == {n: m.training for n, m in model.named_modules()}
    for name, param in pruned.named_parameters():
        source = dict(model.named_parameters())[name]
        assert param.dtype == source.dtype and param.device == source.device
        assert param.requires_grad == source.requires_grad
    for layer in (0, 1):
        names = ("lin",) if kind == "gcn" else ("lin_l", "lin_r")
        for name in names:
            linear = getattr(pruned.convs[layer], name)
            assert type(linear) is type(getattr(model.convs[layer], name))
            assert linear.in_channels == linear.weight.shape[1]
            assert linear.out_channels == linear.weight.shape[0]
    # The clone must remain independent even when the caller later changes it.
    before = {n: p.clone() for n, p in model.state_dict().items()}
    with torch.no_grad():
        next(pruned.parameters()).add_(100)
    assert_tensors(model.state_dict(), before)


@pytest.mark.parametrize("kind", MODELS)
@pytest.mark.parametrize("layer,keep", [
    (-1, [0, 1, 2, 3]), (2, [0, 1, 2, 3]), (0, []), (0, [0, 1, 2]),
    (0, [-1, 0, 1, 2]), (0, [0, 1, 2, 8]), (0, [0, 1, 2, 3, 3]),
    (0, [0, 1, 2, 3.5]), (0, [0, 1, 2, True]), (0, list(range(8))),
])
def test_invalid_requests_fail_without_mutating_source(kind, layer, keep):
    model = make_model(kind, torch.float32)
    before = {n: p.clone() for n, p in model.state_dict().items()}
    with pytest.raises((ValueError, TypeError)):
        surgery(model, layer, keep)
    assert_tensors(model.state_dict(), before)


@pytest.mark.parametrize("option", [dict(project=True), dict(normalize=True), dict(aggr="max"), dict(root_weight=False)])
@pytest.mark.parametrize("layer", [0, 1, 2])
def test_rejects_sage_variants_outside_equivalence_proof(option, layer):
    model = make_model("graphsage", torch.float32)
    dims = [5, 8, 7, 3]
    model.convs[layer] = SAGEConv(dims[layer], dims[layer + 1], **option)
    before = {n: p.clone() for n, p in model.state_dict().items()}
    with pytest.raises(ValueError, match="SAGE"):
        surgery(model, 0, [0, 2, 5, 7])
    assert_tensors(model.state_dict(), before)


def test_gcn_warm_graph_caches_are_cleared_without_changing_original():
    model = make_model("gcn", torch.float32)
    for conv in model.convs:
        conv.cached = True
    with torch.no_grad():
        model(graph(torch.float32))
    cached = [conv._cached_edge_index for conv in model.convs]
    pruned = surgery(model, 0, [0, 2, 5, 7])
    for layer in (0, 1):
        assert pruned.convs[layer]._cached_edge_index is None
        assert pruned.convs[layer]._cached_adj_t is None
    assert all(conv._cached_edge_index is old for conv, old in zip(model.convs, cached))
    with torch.no_grad():
        torch.testing.assert_close(pruned(graph(torch.float32)),
                                   activation_mask_logits(model, graph(torch.float32), ([0, 2, 5, 7], list(range(7)))),
                                   rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", MODELS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; device preservation unverified on GPU")
def test_cuda_device_and_logits(kind):
    model = make_model(kind, torch.float64).cuda()
    data = graph(torch.float64, "cuda")
    pruned = surgery(model, 0, [0, 2, 5, 7])
    assert all(p.device.type == "cuda" for p in pruned.parameters())
    with torch.no_grad():
        torch.testing.assert_close(pruned(data), full_weight_mask_logits(model, data, ([0, 2, 5, 7], list(range(7)))),
                                   rtol=1e-9, atol=1e-11)
