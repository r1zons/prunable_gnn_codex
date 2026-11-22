"""Recompute receipt arithmetic without running a model or reading labels."""

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


METHODS = ("static_random", "static_magnitude", "static_snip", "sequential_random",
           "adaptive_greedy", "lookahead_depth2", "tabular_q")
JOBS = {2: "1625979", 4: "1625978", 8: "1625980"}
DENSE = 524807


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def same(x, y):
    assert math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12), (x, y)


def key(value):
    return tuple(map(key, value)) if isinstance(value, list) else value


def count(widths):
    w1, w2 = widths
    return 503*w1 + (w1+10)*w2 + 7


def legal(widths, floor):
    return [f"prune:{layer}:{step}" for layer, width in enumerate(widths)
            for step in (1, 2, 4, 8) if width-step >= floor]


def transition(t, floor, dense_accuracy):
    _, layer, removal = t["action"].split(":")
    layer, removal = int(layer), int(removal)
    before = t["widths_before"]
    expected = before.copy()
    expected[layer] -= removal
    assert t["action"] in legal(before, floor)
    assert expected == t["widths_after"]
    assert [len(s) for s in t["original_kept_ids"]] == expected
    assert t["terminal_reason"] == ("target_band" if expected == [floor]*2 else None)
    s = 1-count(expected)/DENSE
    same(t["sparsity"], s)
    same(t["terminal_penalty"], 0)
    same(t["reward"], .6*(s - (1-count(before)/DENSE)) -
         .4*max(0, dense_accuracy - t["validation_accuracy"]))


def audit_search(audit, row, floor, dense_accuracy, cfg):
    assert audit["stats"]["complete"] and audit["stats"]["terminal_reason"] == "target_band"
    assert row["validation_requests"] == audit["validation_forwards"] + audit["cache_hits"]
    assert row["validation_requests"] == audit["stats"]["validation_evaluations"]
    method = row["method"]
    path, training = audit["deployment"], audit["training"]
    assert len(path) == len(audit["decisions"])
    before = [512, 512]
    old_ids = [set(range(512)), set(range(512))]
    if method.startswith("static_"):
        assert not path and not training and not row["validation_requests"]
        assert audit["stats"]["scoring_gradient_forwards"] == (method == "static_snip")
        return None
    rng = random.Random(cfg["policy_seed"])
    q_cfg = cfg["q_learning"]
    table = {}
    for step in training:
        transition(step, floor, dense_accuracy)
        state, next_state = key(step["state"]), key(step["next_state"])
        actions = legal(step["widths_before"], floor)
        assert actions == step["legal_actions"]
        values = [table.get(state, {}).get(a, 0.) for a in actions]
        assert values == step["q_values"]
        if rng.random() < q_cfg["epsilon"]:
            assert step["exploration"]
            selected = rng.choice(actions)
        else:
            assert not step["exploration"]
            selected = actions[values.index(max(values))]
        assert selected == step["action"] == step["selected_action"]
        old = table.get(state, {}).get(selected, 0.)
        bootstrap = max((table.get(next_state, {}).get(a, 0.) for a in step["next_legal_actions"]), default=0.)
        value = old + q_cfg["alpha"] * (step["reward"] + q_cfg["gamma"] * bootstrap - old)
        same(value, step["q_after"])
        table.setdefault(state, {})[selected] = step["q_after"]
    if method == "tabular_q":
        assert len({step["episode"] for step in training}) == q_cfg["episodes"]
        assert sum(t["terminal_reason"] == "target_band" for t in training) == q_cfg["episodes"]
        assert table == {key(state): scores for state, scores in audit["q_table"]}
    else:
        assert not training
    for step, decision in zip(path, audit["decisions"]):
        transition(step, floor, dense_accuracy)
        assert step["widths_before"] == before
        before = step["widths_after"]
        for kept, previous in zip(step["original_kept_ids"], old_ids):
            assert len(set(kept)) == len(kept) and set(kept) <= previous
        old_ids = list(map(set, step["original_kept_ids"]))
        actions = legal(step["widths_before"], floor)
        assert actions == decision["legal_actions"]
        assert step["action"] == decision["selected_action"]
        if method == "sequential_random":
            assert step["action"] == rng.choice(actions)
        elif method == "tabular_q":
            q_values = [table.get(key(decision["state"]), {}).get(a, 0.) for a in actions]
            assert q_values == decision["q_values"]
            assert step["action"] == actions[q_values.index(max(q_values))]
        else:
            candidates = decision["candidate_paths"]
            for candidate in candidates:
                same(candidate["score"], sum(.9**i*r for i,r in enumerate(candidate["rewards"])))
            assert step["action"] == max(candidates, key=lambda item: item["score"])["actions"][0]
    assert before == [floor, floor]
    same(path[-1]["validation_accuracy"], row["validation"]["micro_f1"])
    return len(path)


def main(root, ratios):
    output = []
    for ratio in ratios:
        job = JOBS[ratio]
        folder = root / f"{ratio}x"
        result = read(folder / "result.json")
        assert result["protocol"]["job"] == job
        eligible = ("polus-c2", "polus-c3", "polus-c4") if ratio == 8 else ("polus-c4",)
        assert result["protocol"]["host"].startswith(eligible) and "P100" in result["protocol"]["gpu"]
        assert result["status"] == "PASS" and result["dense_parameters"] == DENSE
        assert "Successfully completed." in (root / f"lsf.{job}.out").read_text()
        dense_acc = result["dense_validation"]["micro_f1"]
        assert dense_acc == 0.5092326998924346
        assert result["dense_test"]["micro_f1"] == 0.512436696096446
        assert [r["method"] for r in result["conditions"]] == list(METHODS)
        floor = 512 // ratio
        for row in result["conditions"]:
            assert row["width_divisor"] == ratio
            assert list(map(sum, row["branch_widths"])) == [floor, floor]
            assert row["parameters"] == count([floor, floor])
            same(row["compression"], 1-row["parameters"]/DENSE)
            paths = folder / row["method"]
            assert row["checkpoint_sha256"] == sha(paths / "checkpoint.pt")
            assert row["strict_reload"] and row["dense_projection_exact"]
            for timing in row["timing"].values():
                samples = timing["samples_seconds"]
                assert len(samples) == 20 and min(samples) > 0
                same(timing["median_ms"], 1000 * statistics.median(samples))
                same(timing["mean_ms"], 1000 * statistics.mean(samples))
                same(timing["stdev_ms"], 1000 * statistics.stdev(samples))
                same(timing["vertices_per_second"], 89250 / statistics.median(samples))
            same(row["speedup"], row["timing"]["dense"]["median_ms"] / row["timing"]["pruned"]["median_ms"])
            with open(paths / "search.json") as stream:
                row["deployment_actions"] = audit_search(json.load(stream), row, floor, dense_acc,
                                                         result["protocol"]["config"])
            output.append(row)
    json.dump(output, sys.stdout, indent=2)


if __name__ == "__main__":
    main(Path(sys.argv[1]), list(map(int, sys.argv[2:])) or JOBS)
