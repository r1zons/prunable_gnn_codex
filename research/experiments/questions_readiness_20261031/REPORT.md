# Questions local readiness — 2026-10-31

**Data/software readiness: GO on the verified local CPU stack. Inclusion in the
full POLUS quality campaign at the proposed baselines: NO-GO pending a declared
convergence/imbalance diagnostic and acceptance criterion.** The integration works,
but GCN predicts no minority examples and GraphSAGE recalls only 5/316 at fixed
argmax. Accuracy alone conceals this failure. No wider architecture, different
loss, threshold tuning, extra seed, POLUS job or full matrix was run.

Source base is canonical commit `2e116c46184f4b07cffd3b07db86e926fb24cfda`, retaining
the Lookahead threshold audit. Work is isolated on
`implementation/questions-readiness-20261031`. Old V1–V4 results, manuscript,
training/split implementations and Greedy/Lookahead/Q algorithms are unchanged.

## Dataset, direction and protocol

The native PyG HeterophilousGraphDataset loader is registered as `questions`.
Measured tensors: 48,921 nodes, 301 finite float32 features, int64 labels 0/1,
307,080 unique reciprocal directed entries and no self-loops. The raw NPZ contains
153,540 undirected edges stored once; installed PyG 2.6.1 performs the reverse-edge
conversion and coalescing. Our loader adds no edges/transforms. Full bounds,
shape, binary-class, uniqueness, symmetry and official-mask checks pass.
See [dataset integration documentation](../../../docs/questions.md),
[the authors' release](https://github.com/yandex-research/heterophilous-graphs) and
[the versioned processor](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/datasets/heterophilous_graph_dataset.html).

Raw input SHA256:
`757ebd772bab1475c4dd951ca9e364400c6db161656cff9d21780ee874cf3074`.
The compact [readiness evidence](readiness.json) records graph/cache/split hashes,
original PyG processor identity, observed dimensions and all source/runtime pins.
Raw NPZ, processed tensors, checkpoints and histories remain outside Git under
`${GNN_ARTIFACT_ROOT}/questions-readiness-20261031/`.

Ten official boolean mask columns are retained, each partitioning nodes as
24,460/12,230/12,231 (50/25/25). **None is selected.** Our primary study retains
`generate_exact_ratio_split`: seed 101, label-independent Python shuffle,
floor 60%/floor 20%/remainder. Generated indices are deterministic, disjoint,
exhaustive and different from all official folds. No stratification was introduced.

| Partition | Nodes | Class0 | Class1 |
|---|---:|---:|---:|
| Full graph | 48,921 | 47,461 | 1,460 |
| Generated train | 29,352 | 28,516 | 836 |
| Generated validation | 9,784 | 9,468 | 316 |
| Generated test | 9,785 | 9,477 | 308 |

Class1 is 2.9844% overall. Split-class counts are dataset inspection, not test
performance. Test labels are replaced by −1 before training and the pruning smoke;
no test metric is computed or used for selection. The framework's usual frozen
workflow still reports test metrics only after its decisions, as previously.

The [official implementation](https://github.com/yandex-research/heterophilous-graphs/blob/main/datasets.py)
uses ROC-AUC and one-logit BCE for binary tasks, and its
[logger](https://github.com/yandex-research/heterophilous-graphs/blob/main/utils.py)
selects the best validation metric on official splits. We deliberately retain
two-logit CE and validation-loss selection on generated splits. ROC-AUC from
continuous class 1 probabilities is **supplementary reporting**, never a new
selection/reward objective. These results do not reproduce or rank against the
authors' official ten-fold benchmark. Generic historical metric schemas are unchanged;
the readiness reporting helper adds AUC, class-wise metrics and confusion matrices.

## Approved bounded pilot and results

Exactly one dense run per model: three layers, widths 32/32, dropout 0.5, seed 101,
Adam lr 0.01, weight decay 0.0005, maximum 200 epochs and patience 50. The existing
DenseTrainer restores its minimum-validation-CE state. Both runs reach the 200-epoch
cap rather than exhausting patience. Best epochs below are **one-based**.

| Model | Best epoch | Val. CE loss | Val. accuracy % | Val. macro-F1 | Val. ROC-AUC | Positive recall |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 199 | 0.153227 | 96.7702 | 0.491793 | 0.569418 | 0/316 |
| GraphSAGE | 198 | 0.125288 | 96.8213 | 0.507497 | 0.751437 | 5/316 =1.5823% |
| Train-majority constant prediction | — | — | 96.7702 | 0.491793 | 0.5 for constant scores | 0/316 |

Confusion matrices (rows true0/1, columns predicted0/1):
GCN `[[9468,0],[316,0]]`; GraphSAGE `[[9468,0],[311,5]]`.
The train-prior constant-probability validation CE is 0.142893; GCN is worse than
that log-loss baseline, whereas GraphSAGE is better. GraphSAGE has useful score
ranking but very low positive recall at the unchanged argmax decision rule.
These are single-seed diagnostics, not independent-checkpoint estimates or a
statistically established model ranking.

### Diagnosis and minimal next step

There is no observed data-shape, nonfinite-feature, duplicated-edge, split or
checkpoint-reload defect. Features range −1.6609 to 3.3611, with no constant columns;
column RMS ranges 0.0266–0.3863. Training losses remain finite. The evidence instead
shows the tension between a roughly97% majority class, unweighted CE/validation-
loss selection, fixed argmax and minority-sensitive quality. GCN's discrimination
is weak; GraphSAGE's AUC shows that absence of many positive decisions does not
mean absence of ranking signal. Three-layer aggregation/limited capacity may
contribute, but this pilot does not causally isolate them.

Both best epochs lie near the imposed cap. A minimal **separately approved** next
diagnostic is a prespecified same-width 32 convergence check with a larger epoch
cap (for example 400), holding split, architecture, optimizer and selection fixed.
It would test a budget limitation, not guarantee improved minority performance.
Do not automatically widen to 64: the current evidence does not identify width as
the main cause, and added capacity alone does not resolve class imbalance or the
metric/decision-rule mismatch. Any later width 64, class-weight or calibrated-
threshold comparison needs its own declared protocol; none was performed here.

## Physical channel smoke and resources

Each selected checkpoint receives exactly one fixed magnitude-based action
`prune:0:1` through the existing LocalV2 environment. There is no adaptive-policy
search or compression grid. Both final architectures are 31/32; strict reload
produces identical validation logits, and source dense tensors remain unchanged.

| Model | Parameters before → after | Exact reduction % | Smoke search validation calls |
|---|---|---:|---:|
| GCN | 10,786 → 10,452 | 3.096607 | 2 (dense + one candidate) |
| GraphSAGE | 21,506 → 20,839 | 3.101460 | 2 (dense + one candidate) |

Validation accuracy/F1/AUC remain unchanged in these two particular smokes. This
proves the tested physical operation and checkpoint compatibility, not general
pruning robustness or an accuracy/cost benefit. Additional reporting/reload forwards
are separate from the environment's two search callbacks and included in smoke
wall time; no inference benchmark is claimed.

| Model | Training wall s | Median epoch s | Worker peak RSS MiB | Dense checkpoint bytes |
|---|---:|---:|---:|---:|
| GCN | 129.80 | 0.629 | 622.09 | 46,562 |
| GraphSAGE | 196.40 | 0.977 | 899.61 | 90,417 |

CPU runtime: Python 3.12.9, Torch 2.2.2, PyG 2.6.1; two threads per worker. Models ran
in separate concurrent workers, overlapping pytest load, so these wall times are
approximate readiness observations, not isolated performance comparisons. RSS is
each process's high-water value including imports, data, optimizer and smoke.
Core x/edge/y storage is 64,205,532 bytes (about 61.23 MiB); model weights are tiny
relative to graph/activations. Recorded validation callbacks took roughly 0.33–0.44s
locally (two observations per model, no benchmark warmup); they cannot predict
POLUS search time or multiply into an approved LSF budget. Full Lookahead/Q search
can require many such forwards and must have a separately authorized native
capacity gate. POLUS loader/version/cache compatibility was not checked remotely.

Pilot checkpoints retain frozen model/split identities and are pruning-compatible;
they are diagnostic outputs, not approved campaign baselines or resumable full
optimizer checkpoints. No fine-tuning, larger-width fit or additional training seed
was run. No existing system environment was changed.

## Tests, provenance and decision

| Gate | PASS | FAIL | SKIPPED |
|---|---:|---:|---:|
| Focused Questions + existing dataset tests | 32 | 0 | 0 |
| Full pytest suite | 527 | 0 | 6 |
| Real local pilot workers / physical smokes | 2 / 2 | 0 | 0 |

Full-suite skips: two existing CUDA device-preservation tests (no CUDA) and four
explicit opt-in archived Cora checks from the previous audit (not enabled in this
Questions task). Those four skips are not a Questions PASS; their earlier audit
receipt remains untouched. The 39 warnings are existing Torch JIT AST deprecations.
Offline tests cover native PyG NPZ processing/cache stability, invalid tensors and
official masks, factory/registry/config, generated split precedence and determinism,
both forwards, tiny training/checkpoints, static/native Greedy/Q/Lookahead frozen
structural workflows, and continuous-score AUC without modifying legacy metrics.

[Verification receipt](verification.json) binds logs/JUnit statuses, source hashes,
input/checkpoint hashes and compact pilot evidence. Full raw logs, model weights,
actual split IDs and epoch histories are external. The pilot ran from base 2e116c4
with new source files pinned by hashes; those exact bytes are preserved in this
implementation commit. No earlier GitHub branch replaced the audited local code.

**GO** for dataset/API/physical-workflow integration on the tested local stack.
**NO-GO** for committing Questions to the full POLUS quality campaign with these
frozen baselines yet: minority quality and convergence need a prespecified decision,
continuous-score metrics need inclusion in campaign reporting, and native resource
preflight remains unverified. Accuracy-only success would be misleading. Questions
can be studied as an explicitly labelled imbalance stress case, but that is a
scientific protocol choice, not a claim that this pilot qualified its baseline.
The prior threshold-versus-exact-controller campaign distinction remains in force.
No push, merge, POLUS operation, historical-result modification or full matrix
launch is part of this task.
