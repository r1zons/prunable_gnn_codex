"""Print dense/post_prune/post_finetune summary table for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    diagnostics_path = Path(args.run_dir).expanduser() / "pruning_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    print("phase,test_accuracy,test_macro_f1,parameter_count,inference_time_mean_sec")
    for phase in ("dense", "post_prune", "post_finetune"):
        metrics = diagnostics["metrics"][phase]
        bench = metrics.get("benchmark", {})
        print(
            f"{phase},"
            f"{metrics['test']['accuracy']:.6f},"
            f"{metrics['test']['macro_f1']:.6f},"
            f"{bench.get('parameter_count', float('nan'))},"
            f"{bench.get('inference_time_mean_sec', float('nan'))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
