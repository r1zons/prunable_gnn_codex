"""Continue the archived eight-epoch experiment and measure checkpoint rollouts."""

import argparse
import os
from pathlib import Path
from statistics import mean

import torch
import yaml

from run_cylinderflow_block_distillation import (
    CHECKPOINT, CHECKPOINT_PATH, FIRST, LAST, CylinderFlowRecords, capture,
    frozen_hash, load_mesh_checkpoint, load_windows, read, scientific, seal,
    sha, tensors_hash, train_block, variant,
)


def run(parent, previous, out):
    experiment = yaml.safe_load(Path("configs/experiments/meshgraphnet_cylinderflow_distillation_extended.yaml").read_text())
    assert sha(previous / "protocol.json") == experiment["previous_protocol_sha256"]
    assert sha(previous / "distilled-block.pt") == experiment["previous_block_sha256"]
    prior = read(previous / "protocol.json")
    cfg = dict(prior["config"], epochs=experiment["epochs"])
    milestones = experiment["rollout_epochs"]
    original_epoch = prior["config"]["epochs"]
    assert original_epoch == milestones[0] and milestones[-1] == cfg["epochs"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1 and "P100" in torch.cuda.get_device_name(0)
    checkpoint = parent / CHECKPOINT_PATH
    assert sha(checkpoint) == CHECKPOINT
    teacher = load_mesh_checkpoint(checkpoint, "cuda").requires_grad_(False)
    student = variant(teacher, LAST, LAST).requires_grad_(False)
    block = student.processors[FIRST].requires_grad_(True)
    fixed = frozen_hash(student, FIRST)
    out.mkdir()
    directory = out / "features"
    directory.mkdir()
    seal(out / "protocol.json", dict(job=os.environ["LSB_JOBID"], parent_job=prior["job"],
         config=cfg, extension=experiment, splits=prior["splits"], train_ids=prior["train_ids"],
         heldout_train_ids=prior["heldout_train_ids"], source=Path("source-commit.txt").read_text().strip()))
    records = CylinderFlowRecords(parent / "data/train.tfrecord", "train")
    train, heldout = [capture(teacher, records, prior[key], cfg["train_frames"], "cuda")
                      for key in ("train_ids", "heldout_train_ids")]
    reference = torch.load(previous / "distilled-block.pt", weights_only=True, map_location="cpu")

    def snapshot(epoch, current, optimizer, rng):
        if epoch not in milestones:
            return
        if epoch == original_epoch:
            # The old file contains weights only. Replay restores Adam moments and sample order.
            assert tensors_hash(current.state_dict()) == tensors_hash(reference), "Epoch 8 replay differs"
        torch.save(current.state_dict(), out / f"block-{epoch}.pt")
        torch.save(dict(optimizer=optimizer.state_dict(), rng_state=rng.getstate(), epoch=epoch),
                   out / f"optimizer-{epoch}.pt")

    best_state, best_epoch, history = train_block(student, train, heldout, cfg, directory, on_epoch=snapshot)
    del records, train, heldout, teacher
    torch.save(best_state, out / "best-block.pt")
    splits = prior["splits"]
    windows = load_windows(CylinderFlowRecords(parent / "data/valid.tfrecord", "valid"),
                           splits["diagnostic"], splits["starts"], cfg["horizon"], purpose="diagnostic")
    summary = []
    for epoch in sorted(set(milestones + [best_epoch])):
        state = (best_state if epoch == best_epoch else
                 torch.load(out / f"block-{epoch}.pt", weights_only=True, map_location="cpu"))
        block.load_state_dict(state)
        rows = scientific(f"rollout-{epoch}", student, windows, out)
        ok = all(r["status"] == "ok" for r in rows)
        result = dict(epoch=epoch, updates=history[epoch]["updates"], heldout=history[epoch]["heldout"],
                      windows=len(rows), successful=sum(r["status"] == "ok" for r in rows),
                      E100=mean(r["E100"] for r in rows) if ok else None,
                      mean100=mean(r["mean100"] for r in rows) if ok else None)
        seal(out / f"summary-{epoch}.json", result)
        summary.append(result)
    assert frozen_hash(student, FIRST) == fixed
    seal(out / "finish.json", dict(best_heldout_epoch=best_epoch, checkpoints=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "previous", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
