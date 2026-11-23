"""Fine-tune three frozen Flickr 4x masks with one shared GraphSAINT schedule."""

import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch
from torch.nn import functional as F
import yaml

from gnn_pruning.experiments.flickr_dense import digest, sha, tensors_hash, write
from run_flickr_zhou_dense import WalkSampler, checkpoint, load_dataset, normalize, restore, sparse
from run_flickr_zhou_pruning import metrics, timing


CONFIG = Path("configs/experiments/flickr_zhou_finetune_4x.yaml")
MASKS = Path("pruned")


def pruned_count(model):
    """Require the original fixed 4x geometry."""
    size = sum(p.numel() for p in model.parameters())
    assert model.hidden_channel_dims == (128, 128) and size == 82055
    return size


def batches_for(train_adj, train_ids, cfg):
    """Materialize one weighted random-walk schedule reused verbatim by every model."""
    phase = cfg["phase"]
    sampler = WalkSampler(train_adj, train_ids, phase, cfg, cfg)
    steps = math.ceil(len(train_ids) / (phase["num_root"] * phase["depth"]))
    assert steps == 4
    allowed = torch.zeros(train_adj.shape[0], device="cuda", dtype=torch.bool)
    allowed[torch.as_tensor(train_ids, device="cuda")] = True
    batches, batch_hashes = [], []
    for _ in range(phase["end"] * steps):
        nodes, adj, weights = sampler.next()
        nodes = torch.as_tensor(nodes, device="cuda")
        assert bool(allowed[nodes].all())
        adj = sparse(adj, "cuda")
        batch_hashes.append(tensors_hash(dict(nodes=nodes, indices=adj.indices(),
                                             edges=adj.values(), weights=weights)))
        batches.append((nodes, adj, weights))
    return batches, steps, dict(batch_sha256=batch_hashes, sha256=digest(batch_hashes),
                                calibration=sampler.calibration)


def fit(model, x, y, adj, val, batches, steps, cfg, out, provenance):
    """Train all physical parameters and select the first maximum of validation Micro-F1."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    torch.cuda.manual_seed(cfg["seed"])
    best = -math.inf
    log = []
    started = time.perf_counter()
    for epoch, offset in enumerate(range(0, len(batches), steps), start=1):
        model.train()
        losses = []
        for nodes, sampled_adj, weights in batches[offset:offset + steps]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[nodes], sampled_adj)
            loss = (F.cross_entropy(logits, y[nodes], reduction="none") * weights).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        validation = metrics(model, x, adj, y, val)
        score = validation["micro_f1"]
        log.append(dict(epoch=epoch, updates=offset + steps, train_loss=statistics.mean(losses),
                        validation=validation))
        if score > best:
            best = score
            state_hash = tensors_hash(model.state_dict())
            checkpoint(out / "best.pt", model, epoch=epoch, validation=validation,
                       state_sha256=state_hash, **provenance)
        print(out.name, epoch, score, flush=True)
    elapsed = time.perf_counter() - started
    write(out / "epochs.json", log)
    selected, saved = restore(out / "best.pt")
    assert tensors_hash(selected.state_dict()) == saved["state_sha256"]
    assert pruned_count(selected) == pruned_count(model)
    assert saved["original_kept_ids"] == provenance["original_kept_ids"]
    assert metrics(selected, x, adj, y, val) == saved["validation"]
    return selected, saved, elapsed


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 1 and "P100" in torch.cuda.get_device_name(0)
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["methods"] == ["static_random", "static_snip", "adaptive_greedy"]
    assert cfg["phase"]["end"] == 30
    protocol = dict(config=cfg, source=Path("source-commit.txt").read_text().strip(),
                    job=os.environ["LSB_JOBID"], host=platform.node(), gpu=torch.cuda.get_device_name(0),
                    torch=str(torch.__version__), cuda=torch.version.cuda)
    out = Path("finetune")
    out.mkdir()
    write(out / "protocol.json", protocol)
    parent = json.loads((MASKS / "result.json").read_text())
    assert parent["status"] == "PASS" and parent["protocol"]["job"] == "1625978"
    rows = {row["method"]: row for row in parent["conditions"]}
    x, y, train_adj, val_adj, full_adj, split = load_dataset(Path("data"), out)
    assert sha(out / "data.json") == sha(MASKS / "data.json")
    features = torch.as_tensor(x, device="cuda")
    val = torch.as_tensor(split["va"], device="cuda")
    val_adj = sparse(normalize(val_adj), "cuda")
    safe = y.copy()
    safe[split["te"]] = -1
    labels = torch.as_tensor(safe, device="cuda")
    batches, steps, schedule = batches_for(train_adj, split["tr"], cfg)
    write(out / "schedule.json", schedule)
    results = []
    for method in cfg["methods"]:
        folder = out / method
        folder.mkdir()
        origin = MASKS / method / "checkpoint.pt"
        row = rows[method]
        assert sha(origin) == row["checkpoint_sha256"]
        initial, original = restore(origin)
        assert pruned_count(initial) == row["parameters"] and row["width_divisor"] == 4
        assert metrics(initial, features, val_adj, labels, val) == row["validation"]
        assert original["dense_sha256"] == parent["dense_sha256"]
        kept = original["original_kept_ids"]
        assert [len(ids) for ids in kept] == [128, 128]
        assert all(ids == sorted(set(ids)) and 0 <= ids[0] <= ids[-1] < 512 for ids in kept)
        provenance = dict(original_kept_ids=kept, initial_sha256=row["checkpoint_sha256"],
                          schedule_sha256=schedule["sha256"], method=method)
        selected, saved, elapsed = fit(initial, features, labels, val_adj, val, batches, steps, cfg, folder, provenance)
        assert initial.export_architecture_config() == saved["architecture"] == original["architecture"]
        assert tensors_hash(original["state"]) != saved["state_sha256"]
        results.append(dict(provenance, before_validation=row["validation"],
                            after_validation=saved["validation"], best_epoch=saved["epoch"],
                            updates=len(batches), parameters=pruned_count(selected),
                            branch_widths=selected.branch_widths, fit_seconds=elapsed,
                            checkpoint_sha256=sha(folder / "best.pt")))
    write(out / "selection.json", results)

    # The test view is first used after all three choices are recorded.
    del batches
    test = torch.as_tensor(split["te"], device="cuda")
    y = torch.as_tensor(y, device="cuda")
    full_adj = sparse(normalize(full_adj), "cuda")
    for result in results:
        method = result["method"]
        initial, _ = restore(MASKS / method / "checkpoint.pt")
        path = out / method / "best.pt"
        assert sha(path) == result["checkpoint_sha256"]
        selected, _ = restore(path)
        result["before_test"] = metrics(initial, features, full_adj, y, test)
        assert result["before_test"] == rows[method]["test"]
        result["after_test"] = metrics(selected, features, full_adj, y, test)
        measured = timing(initial, selected, features, full_adj, cfg)
        result["timing"] = dict(before=measured["dense"], after=measured["pruned"])
    write(out / "result.json", dict(status="PASS", protocol=protocol, conditions=results,
          parent_sha256=sha(MASKS / "result.json"), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    main()
