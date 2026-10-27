"""Targeted regression for the dependency backport; no GNN/model execution."""
from __future__ import annotations

import json
from pathlib import Path

import torch

checks = []
for dtype, bits in [(torch.int16, 16), (torch.int32, 32), (torch.int64, 64)]:
    for length in [1, 3, 17, 64, 257]:
        pattern = [-(1 << (bits - 2)), -13, -1, 0, 1, 7, (1 << (bits - 3)) - 1]
        values = [pattern[i % len(pattern)] for i in range(length)]
        shifts = [i % bits for i in range(length)]
        a = torch.tensor(values, dtype=dtype)
        b = torch.tensor(shifts, dtype=dtype)
        mask = (1 << bits) - 1
        expected_left = []
        for value, shift in zip(values, shifts):
            wrapped = (value << shift) & mask
            expected_left.append(wrapped - (1 << bits) if wrapped >= (1 << (bits - 1)) else wrapped)
        expected_right = [value >> shift for value, shift in zip(values, shifts)]
        if torch.bitwise_left_shift(a, b).tolist() != expected_left:
            raise RuntimeError("Left shift mismatch for {} / {}".format(dtype, length))
        if torch.bitwise_right_shift(a, b).tolist() != expected_right:
            raise RuntimeError("Right shift mismatch for {} / {}".format(dtype, length))
        checks.append({"dtype": str(dtype), "length": length, "valid_shift_counts_only": True})
payload = {"status": "PASS", "torch": torch.__version__, "cases": checks,
           "training_or_pruning_executed": False}
Path("logs/vsx-shift-regression.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
