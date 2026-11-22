"""Validate the pinned GCNP forward paths and reproduce its dense Flickr run."""

import argparse
from contextlib import ExitStack
import importlib
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F
import yaml

from gnn_pruning.experiments.flickr_dense import sha, write
from gnn_pruning.models import build_model


AUTHOR_COMMIT = "3b0646450620419dc6e2de457fc8f1afcbf03754"
REFERENCE = Path("author/train_config/fullbatch/flickr_4x.yml")
CONFIG = Path("configs/experiments/flickr_zhou_dense.yaml")
ARCH = dict(in_channels=500, out_channels=7, hidden_channels=256, num_layers=2, dropout=.3)
DATA_FILES = ("adj_full.npz", "adj_train.npz", "feats.npy", "role.json", "class_map.json")


def author_model(features, labels, network, params):
    """Import the original dense model with inert shims for unused CLI and sampler modules."""
    sys.path.insert(0, str(Path("author").resolve()))
    flags = ModuleType("GNN.globals")
    flags.args_global = SimpleNamespace(gpu=0)
    sys.modules["GNN.globals"] = flags
    for name in ("minibatch_sampler", "minibatch"):
        qualified = f"GNN.pytorch_version.{name}"
        module = ModuleType(qualified)
        if name == "minibatch":
            # The adapter isolates the unused training data loader from the dense model.
            def unused(*_args, **_kwargs):
                raise RuntimeError("Native GCNP minibatch loader is outside the reference forward")
            module._coo_scipy2torch = unused
        sys.modules[qualified] = module
    cls = importlib.import_module("GNN.pytorch_version.models").GraphSAINT
    return cls(7, network, params, features, np.eye(7)[labels]).cuda()


@torch.no_grad()
def copy_weights(ours, author, to_author):
    """Copy only the tensors used by the dense reference forward."""
    pairs = [(ours.classifier.weight, author.classifier.f_lin[0].weight),
             (ours.classifier_bias, author.classifier.f_bias[0])]
    for layer, original in zip(ours.layers, author.aggregators):
        for i, branch in enumerate((layer.self_branch, layer.neighbor_branch)):
            pairs.extend([(branch.projection.weight, original.f_lin[i].weight),
                          (branch.bias, original.f_bias[i]), (branch.scale, original.scale[i]),
                          (branch.offset, original.offset[i])])
    for a, b in pairs:
        if to_author:
            b.copy_(a)
        else:
            a.copy_(b)


def author_logits(author, features, adj, intermediates=None):
    """Follow the unmodified GCNP full-graph layer methods up to its logits."""
    for layer in author.aggregators:
        features = layer.inplace_forward(features, adj)
        if intermediates is not None:
            intermediates.append(features.clone())
    F.normalize(features, p=2, dim=1, out=features)
    return author.classifier.inplace_forward(features)


def sparse(matrix, device):
    """Convert one precomputed SciPy graph to coalesced Torch COO."""
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(np.stack((coo.row, coo.col)), coo.data.astype(np.float32),
                                   coo.shape, device=device).coalesce()


def normalize(matrix):
    """Compute D^-1 A once, leaving zero-degree rows empty."""
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1., degree, out=np.zeros_like(degree, dtype=np.float64), where=degree != 0)
    return sp.diags(inverse).dot(matrix).tocsr()


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["architecture"])
    model = build_model(config.pop("name"), **config).cuda()
    model.load_state_dict(payload["state"], strict=True)
    return model.eval(), payload


def checkpoint(path, model, **metadata):
    torch.save(dict(architecture=model.export_architecture_config(),
                    state={k: v.cpu() for k, v in model.state_dict().items()}, **metadata), path)


@torch.no_grad()
def parity(cfg, network, params, out):
    """Gate both paths against the pinned model on the allocated P100."""
    rng = np.random.default_rng(cfg["seed"])
    features = rng.normal(size=(11, 500)).astype(np.float32)
    labels = np.arange(len(features)) % 7
    adjacency = rng.uniform(size=(11, 11)) < .35
    adjacency[0] = False
    np.fill_diagonal(adjacency, False)
    adj = sparse(normalize(sp.csr_matrix(adjacency)), "cuda")
    author = author_model(features, labels, network, params)
    ours = build_model("graphsage_zhou", **ARCH).cuda()
    for layer in author.aggregators:
        for bias, offset, scale in zip(layer.f_bias, layer.offset, layer.scale):
            bias.uniform_(-.2, .2)
            offset.uniform_(-.1, .1)
            scale.uniform_(.5, 1.5)
    author.classifier.f_bias[0].uniform_(-.2, .2)
    copy_weights(ours, author, to_author=False)
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in
              (("ours", ours), ("author", author))}
    assert counts == {"ours": 524807, "author": 524821}
    shapes = {k: list(p.shape) for k, p in ours.named_parameters()}
    assert [layer.self_branch.projection.weight.shape for layer in ours.layers] == [
        (256, 500), (256, 512)]
    assert ours.classifier.weight.shape == (7, 512)
    observations = {}
    for training in (True, False):
        author.train(training)
        ours.train(training)
        reference_layers, actual_layers = [], []
        with ExitStack() as hooks:
            for layer in ours.layers:
                handle = layer.register_forward_hook(lambda _m, _i, x: actual_layers.append(x.clone()))
                hooks.callback(handle.remove)
            if training:
                for layer in author.aggregators:
                    handle = layer.register_forward_hook(lambda _m, _i, x: reference_layers.append(x[1].clone()))
                    hooks.callback(handle.remove)
                torch.cuda.manual_seed(cfg["seed"])
                expected = author(np.arange(len(features)), adj)[0]
                torch.cuda.manual_seed(cfg["seed"])
            else:
                expected = author_logits(author, author.feat_full, adj, reference_layers)
            actual = ours(author.feat_full, adj)
        errors = {}
        for name, a, b in zip(("layer1", "layer2", "logits"), (*actual_layers, actual),
                              (*reference_layers, expected)):
            torch.testing.assert_close(a, b, rtol=cfg["parity_rtol"], atol=cfg["parity_atol"])
            errors[name] = float((a-b).abs().max())
        observations["train" if training else "eval"] = errors

    checkpoint(out / "parity-checkpoint.pt", ours)
    loaded, _ = restore(out / "parity-checkpoint.pt")
    torch.testing.assert_close(loaded(author.feat_full, adj), actual, rtol=0, atol=0)
    # Confirm the old registry key and strict reload still use the previous parameter layout.
    legacy = build_model("graphsage", in_channels=500, hidden_channels=32,
                         out_channels=7, num_layers=2, dropout=.5).cuda().eval()
    checkpoint(out / "legacy-checkpoint.pt", legacy)
    old, _ = restore(out / "legacy-checkpoint.pt")
    for name, tensor in legacy.state_dict().items():
        torch.testing.assert_close(old.state_dict()[name], tensor, rtol=0, atol=0)
    from torch_geometric.data import Data
    data = Data(x=author.feat_full, edge_index=adj.indices())
    legacy_logits, reloaded_logits = legacy(data), old(data)
    # CUDA scatter accumulation in the unchanged PyG SAGEConv can vary by roundoff.
    torch.testing.assert_close(legacy_logits, reloaded_logits,
                               rtol=cfg["parity_rtol"], atol=cfg["parity_atol"])
    write(out / "parity.json", dict(status="PASS", device=torch.cuda.get_device_name(0),
          precision="FP32", author_commit=AUTHOR_COMMIT, atol=cfg["parity_atol"], rtol=cfg["parity_rtol"],
          max_abs=observations, parameter_counts=counts, parameter_shapes=shapes,
          strict_reload=True, legacy_graphsage_strict_reload=True,
          legacy_graphsage_reload_logits_max_abs=float((legacy_logits - reloaded_logits).abs().max())))
    print("PASS", observations, counts, flush=True)


def load_dataset(directory, out):
    """Verify published files and construct the documented train+validation graph."""
    import json

    roles = json.loads((directory / "role.json").read_text())
    assert set(roles) == {"tr", "va", "te"}
    split = {name: np.asarray(ids, dtype=np.int64) for name, ids in roles.items()}
    assert np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(89250))
    class_map = json.loads((directory / "class_map.json").read_text())
    assert set(map(int, class_map)) == set(range(89250))
    labels = np.array([class_map[str(i)] for i in range(89250)])
    features = np.load(directory / "feats.npy", allow_pickle=False)
    assert features.shape == (89250, 500) and np.isfinite(features).all()
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
    labels -= labels.min()
    assert set(labels) == set(range(7))
    full, train = [sp.load_npz(directory / f"adj_{name}.npz").astype(bool).tocsr()
                   for name in ("full", "train")]
    for matrix in (full, train):
        assert matrix.shape == (89250, 89250) and matrix.has_canonical_format
        assert matrix.nnz == matrix.count_nonzero()
    assert full.nnz == 899756

    def induced(ids):
        row_mask = np.zeros(full.shape[0], dtype=np.float32)
        row_mask[ids] = 1
        diagonal = sp.diags(row_mask)
        return (diagonal @ full @ diagonal).tocsr()

    assert (induced(split["tr"]) != train).nnz == 0
    expected = induced(np.concatenate((split["tr"], split["va"])))
    val_source = directory / "adj_val.npz"
    if val_source.exists():
        val = sp.load_npz(val_source).astype(bool).tocsr()
        val_origin = "provided"
        assert val.shape == full.shape and (val != expected).nnz == 0
    else:
        val, val_origin = expected, "derived_train_plus_val_induced"
    val_path = out / "adj_val.npz"
    sp.save_npz(val_path, val)
    train_rows = np.array(list(set(train.nonzero()[0])))
    scaler = StandardScaler().fit(features[train_rows])
    features = scaler.transform(features).astype(np.float32)
    assert np.isfinite(features).all()
    np.savez(out / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
    write(out / "data.json", dict(hashes={name: sha(directory / name) for name in DATA_FILES},
          split_counts={name: len(ids) for name, ids in split.items()},
          nnz=dict(train=train.nnz, val=val.nnz, full=full.nnz), adj_val_origin=val_origin,
          adj_val_sha256=sha(val_path), feature_fit_nodes=len(train_rows), scaler_sha256=sha(out/"scaler.npz")))
    return features, labels, train, val, full, split


class WalkSampler:
    """GraphSAINT random-walk induced subgraphs and coverage normalization."""

    def __init__(self, train, train_nodes, phase, params, cfg):
        self.rng = np.random.default_rng(cfg["seed"])
        self.nodes = train_nodes
        self.indptr, self.neighbors = train.indptr, train.indices
        self.degree = np.diff(self.indptr)
        self.edge_ids = sp.csr_matrix((np.arange(train.nnz, dtype=np.int32) + 1,
                                      train.indices, train.indptr), shape=train.shape)
        self.roots, self.depth, self.batch = phase["num_root"], phase["depth"], cfg["sampling_batch"]
        self.cache = []
        node_seen = np.zeros(train.shape[0], dtype=np.int64)
        edge_seen = np.zeros(train.nnz, dtype=np.int64)
        coverage = 0
        while coverage <= params["sample_coverage"] * len(self.nodes):
            self.refill()
            for nodes, _, _, edges in self.cache[-self.batch:]:
                node_seen[nodes] += 1
                edge_seen[edges] += 1
                coverage += len(nodes)
        row = np.repeat(np.arange(train.shape[0]), self.degree)
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = np.clip(node_seen[row] / edge_seen, 0, 1e4)
        correction[np.isnan(correction)] = .1
        self.edge_weight = (correction.astype(np.float32) / self.degree[row]).astype(np.float32)
        self.node_weight = (len(self.cache) / np.where(node_seen, node_seen, .1) / len(self.nodes)).astype(np.float32)
        self.calibration = dict(subgraphs=len(self.cache), sampled_nodes=coverage,
                                coverage=coverage / len(self.nodes), batch_size=self.batch)

    def refill(self):
        for _ in range(self.batch):
            roots = self.rng.choice(self.nodes, size=self.roots, replace=True)
            current = roots
            visited = [roots]
            for _ in range(self.depth):
                current = current[self.degree[current] > 0]
                offset = self.rng.integers(self.degree[current])
                current = self.neighbors[self.indptr[current] + offset]
                visited.append(current)
            nodes = np.unique(np.concatenate(visited))
            coo = self.edge_ids[nodes][:, nodes].tocoo()
            self.cache.append((nodes, coo.row, coo.col, coo.data - 1))

    def next(self):
        if not self.cache:
            self.refill()
        nodes, row, col, edges = self.cache.pop()
        subgraph = sp.coo_matrix((self.edge_weight[edges], (row, col)),
                                 shape=(len(nodes), len(nodes)))
        return nodes, subgraph, torch.as_tensor(self.node_weight[nodes], device="cuda")


@torch.no_grad()
def evaluation(model, features, adjacency, labels, nodes):
    """Compute the full-graph logits and micro-F1 on only the requested split."""
    model.eval()
    logits = model(features, adjacency)
    value = logits[nodes]
    assert torch.isfinite(value).all()
    return float((value.argmax(dim=-1) == labels[nodes]).double().mean())


@torch.no_grad()
def benchmark(model, author, features, adjacency, labels, cfg):
    """Alternate implementations for identical warmed-up, synchronized timing scopes."""
    from itertools import cycle

    n = len(features)
    weight = torch.full((n,), 1 / n, device=features.device)
    nodes = np.arange(n)
    model.eval()
    author.eval()

    def ours_scope():
        logits = model(features, adjacency)
        # Mirror GCNP's full loss/probability work and Python batch setup.
        loss = (F.cross_entropy(logits, labels, reduction="none") * weight).sum()
        logits.exp_()
        logits.div_(logits.sum(dim=1, keepdim=True))
        return loss, logits

    torch.testing.assert_close(model(features, adjacency), author_logits(author, features, adjacency),
                               rtol=cfg["parity_rtol"], atol=cfg["parity_atol"])
    logits_difference = float((model(features, adjacency) - author_logits(author, features, adjacency)).abs().max())
    author_loss, author_prob, _ = author.eval_step(nodes, adjacency, weight)
    ours_loss, ours_prob = ours_scope()
    torch.testing.assert_close(ours_loss, author_loss, rtol=cfg["parity_rtol"], atol=cfg["parity_atol"])
    torch.testing.assert_close(ours_prob, author_prob, rtol=cfg["parity_rtol"], atol=cfg["parity_atol"])
    output = {"full_graph_logit_max_abs": logits_difference, "scopes": {}}

    def ours_eval():
        np.arange(n)
        return ours_scope()

    def original_eval():
        return author.eval_step(np.arange(n), adjacency, weight)

    for scope, runners in (
        ("logits_only", (lambda: model(features, adjacency), lambda: author_logits(author, features, adjacency))),
        ("logits_loss_probabilities", (ours_eval, original_eval)),
    ):
        results = {key: [] for key in ("ours", "author")}
        for i, order in zip(range(cfg["warmup"] + cfg["repeats"]), cycle(((0, 1), (1, 0)))):
            for index in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                runners[index]()
                torch.cuda.synchronize()
                if i >= cfg["warmup"]:
                    results[("ours", "author")[index]].append(time.perf_counter() - started)
        output["scopes"][scope] = {
            key: dict(samples_seconds=values, median_ms=1000 * statistics.median(values),
                      stdev_ms=1000 * statistics.stdev(values),
                      vertices_per_second=n / statistics.median(values))
            for key, values in results.items()}
    return output


def train(cfg, network, params, phase, out):
    """Train the dense configuration and select exclusively by validation Micro-F1."""
    import json

    gate = json.loads(Path("parity/gate.json").read_text())
    assert gate["status"] == "PASS" and gate["source"] == Path("source-commit.txt").read_text().strip()
    assert gate["parity_sha256"] == sha(Path("parity/parity.json"))
    x, y, train_adj, val_adj, full_adj, split = load_dataset(Path("data"), out)
    features = torch.as_tensor(x, device="cuda")
    labels = torch.as_tensor(y, dtype=torch.long, device="cuda")
    validation = torch.as_tensor(split["va"], device="cuda")
    val = sparse(normalize(val_adj), "cuda")
    sampler = WalkSampler(train_adj, split["tr"], phase, params, cfg)
    write(out / "sampler.json", sampler.calibration)
    model = build_model("graphsage_zhou", **ARCH).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    steps = math.ceil(len(split["tr"]) / (phase["num_root"] * phase["depth"]))
    best = float("-inf")
    path = out / "best.pt"
    for epoch in range(1, phase["end"] + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for _ in range(steps):
            nodes, adj, weights = sampler.next()
            nodes = torch.as_tensor(nodes, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[nodes], sparse(adj, "cuda"))
            loss = (F.cross_entropy(logits, labels[nodes], reduction="none") * weights).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        micro = evaluation(model, features, val, labels, validation)
        write(out / f"epoch-{epoch:02}.json", dict(epoch=epoch, optimizer_steps=steps,
              validation_micro_f1=micro, train_loss=statistics.mean(losses), seconds=time.perf_counter()-started))
        print("epoch", epoch, "validation_micro_f1", micro, flush=True)
        if micro > best:
            best = micro
            checkpoint(path, model, epoch=epoch, validation_micro_f1=micro)
    selected, metadata = restore(path)
    assert evaluation(selected, features, val, labels, validation) == best
    full = sparse(normalize(full_adj), "cuda")
    test = torch.as_tensor(split["te"], device="cuda")
    micro = evaluation(selected, features, full, labels, test)
    author = author_model(x, y, network, params)
    copy_weights(selected, author, to_author=True)
    measurements = benchmark(selected, author, features, full, labels, cfg)
    write(out / "timing.json", measurements)
    write(out / "result.json", dict(status="PASS", epoch=metadata["epoch"], validation_micro_f1=best,
          test_micro_f1=micro, epochs=phase["end"], steps_per_epoch=steps,
          checkpoint_sha256=sha(path), parity_receipt_sha256=sha(Path("parity/parity.json")),
          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    print("test_micro_f1", micro, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "train"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 1 and "P100" in torch.cuda.get_device_name(0)
    cfg = yaml.safe_load(CONFIG.read_text())
    reference = yaml.safe_load(REFERENCE.read_text())
    network, params, phase = (reference[k][0] for k in ("network", "params", "phase"))
    assert network == dict(dim=256, aggr="concat", loss="softmax", arch="1-1", act="relu", bias="norm")
    assert params == dict(lr=.05, dropout=.3, weight_decay=0., sample_coverage=50)
    assert phase == dict(end=15, sampler="rw", num_root=6000, depth=2)
    torch.manual_seed(cfg["seed"])
    args.out.mkdir()
    source = Path("source-commit.txt").read_text().strip()
    write(args.out / "protocol.json", dict(config=cfg, network=network, params=params, phase=phase,
          source=source, author_commit=AUTHOR_COMMIT, author_config_sha256=sha(REFERENCE),
          job=os.environ["LSB_JOBID"], host=platform.node(), torch=str(torch.__version__),
          numpy=np.__version__, python=platform.python_version(), cuda=torch.version.cuda,
          gpu=torch.cuda.get_device_name(0)))
    if args.mode == "parity":
        parity(cfg, network, params, args.out)
        # Publish the source-bound gate only after every numeric and reload check passes.
        write(args.out / "gate.json", dict(status="PASS", source=source, parity_sha256=sha(args.out/"parity.json")))
    else:
        train(cfg, network, params, phase, args.out)


if __name__ == "__main__":
    main()
