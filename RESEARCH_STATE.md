# RESEARCH_STATE — GNN pruning Master's project

**Document status:** `CODE SNAPSHOT FROZEN at audited commit; research protocol / empirical evidence OPEN`  
**Prepared / revised:** 2026-09-23  
**Repository:** https://github.com/r1zons/prunable_gnn_codex  
**Audited code snapshot:** `882b3c97385a43361e417ff02f8e6854899ede85` (`main`, publicly inspected 2026-09-23)  
**Pre-audit code revision:** `7b0f28089112c163d2aa19741f8ec0d4a4d5eae6`  
**Document revision:** `research-state-v2-audited-commit`  
**Authoritative code snapshot:** https://github.com/r1zons/prunable_gnn_codex/tree/882b3c97385a43361e417ff02f8e6854899ede85 . The immutable commit fixes the code baseline; it does **not** freeze the scientific hypothesis, experiment protocol, environment, or empirical results.

> **Read this first.** This document separates **FACT / VERIFIED REPOSITORY** (publicly inspected source at the pinned SHA), **REPORTED TEST RESULT** (Codex's local test report; no independent CI verification), **DECISION** (explicit working constraints), **CANDIDATE** (not selected), and **OPEN** (unresolved). Implemented code and passing unit tests do not establish a scientific contribution, validated benchmarks, or a final Master's thesis topic. Do not silently promote a candidate topic to a decision.

## 1. Purpose and research context

- The project extends earlier coursework on the **influence of structural pruning on graph neural network performance**. The central task so far is **node classification** with PyTorch Geometric.
- The current academic planning frame in this conversation is **approximately three semesters**. Earlier project discussions used a shorter horizon; validate the actual academic submission calendar before committing to a new large workstream.
- Research posture: empirical investigation with a meaningful, but manageable, mathematical/optimization component. Prefer **managed risk**: a defensible core contribution should remain achievable without a large-scale RL training campaign or uninterrupted access to powerful GPUs.
- Existing code is an experimental platform and useful prior work, **not** a finalized thesis problem, contribution, or result.
- Earlier conversations considered inference-pipeline cost decomposition, normalized speedups, and comparisons between model-side and graph-side acceleration. Preserve these as important research **interests/criteria**, not as a commitment to a joint-pruning thesis.

## 2. Version and evidence ledger

| Evidence/status | Item | Implication |
|---|---|---|
| FACT / VERIFIED REPOSITORY | `main` points to audited commit `882b3c97385a43361e417ff02f8e6854899ede85` (2026-09-23); its parent is `7b0f28089112c163d2aa19741f8ec0d4a4d5eae6`. [Audit commit](https://github.com/r1zons/prunable_gnn_codex/commit/882b3c97385a43361e417ff02f8e6854899ede85). | The post-audit source is now reproducibly identifiable; use this exact SHA for comparison rather than moving `main`. |
| FACT / VERIFIED REPOSITORY | Audit commit updates 22 paths: aggregation/reporting, structural checkpoint compatibility, documentation and metadata, validation-based summary/plot selection, and regression tests. | Code changes are confirmed in the published tree; their behavior on every configuration has not been independently exhaustively tested. |
| FACT / VERIFIED REPOSITORY | Suite aggregation groups by stable condition and includes per-run metric means, sample SD, Student-t 95% CI, run count/status and stopping-reason counts. | Multi-seed reporting code is present; before the final research study inspect raw rows and check that intended independent replicates and distinct hyperparameter conditions remain separated. |
| FACT / VERIFIED REPOSITORY | GCN/GraphSAGE accept a scalar or per-layer hidden-channel widths and export that architecture; structural checkpoint reconstruction preserves non-uniform widths. Structured `achieved_sparsity` is normalized to parameter-count reduction, with channel-removal diagnostics separate. | Stronger support for sequential local surgery/checkpoint round trips and comparable compression coordinates. Old output files are not automatically migrated. |
| FACT / VERIFIED REPOSITORY | Package metadata declares `requires-python >=3.10`; README/ROADMAP/AGENTS describe the implemented tabular Q-learning prototype and its limitations. | This is a minimum Python declaration, not a verified CUDA/PyTorch/PyG compatibility matrix for the university server. |
| REPORTED TEST RESULT | Codex reported `python -m pytest -q` using its local Conda environment and **198 passing tests**, including focused aggregation, surgery, Q-learning and checkpoint checks. | This is a report from the user-supplied Codex run; no independently observed CI run or full experimental reproduction is available for the pinned SHA. |
| OPEN | Record exact Python/Torch/PyG/CUDA/driver/hardware versions and test log/provenance; establish canonical post-audit run IDs and selection protocol. | The executable source is frozen, while environment and empirical benchmark are not. |

**Code-snapshot freeze:** complete for published commit `882b3c97385a43361e417ff02f8e6854899ede85`. **Scientific/evidence freeze:** not complete. Any subsequent code change requires a new pinned code SHA and an explicit update of this document. The document may be committed separately; its pinned SHA should continue to reference the audited *parent code revision*, not its own documentation commit.

## 3. Research objects: keep them distinct

**A. Structural model pruning — current implementation.** Remove hidden channels from the GNN itself and rebuild affected parameter tensors/layers. The input graph need not change. Structural channel reduction is different from zeroing individual weights.

**B. Input-graph pruning / sparsification — separate candidate problem.** Remove or select edges, nodes, messages, or subgraphs of the *input graph*, possibly preserving the model architecture. This affects message passing and task information in a different way.

**C. Joint model-and-graph pruning — broader candidate.** Optimize both structures. Do not equate it with A or assume the existing code already implements it.

**DECISION:** Select a precise pruning target and deployment scenario **before** locking a new algorithm, novelty claim, or benchmark. Existing implementation does not by itself determine the thesis topic.

## 4. Implemented platform (VERIFIED REPOSITORY at audited SHA)

### Models, data, and experimental workflow

- Models: **GCN** and **GraphSAGE** for node classification. GAT/GIN have been discussed but are not part of the verified implemented model registry.
- Supported dataset loaders/configurations: **Cora, CiteSeer, PubMed, Texas, Cornell, Wisconsin, Actor, Amazon Computers, Flickr, Reddit**, and **DBLP** via an explicit homogeneous author-projection adapter. Loader/configuration support does **not** mean a validated final benchmark exists for every dataset.
- `ogbn-arxiv` was discussed as a possible benchmark; it is **not a verified current dataset loader** and its official split/evaluator is not yet integrated.
- Main pipeline: dense train/evaluate → prune → post-prune evaluate → optional fine-tune → post-fine-tune evaluate. YAML configuration, checkpoint and split artifacts, CLI runs, suites, CSV/JSON reports, and a structural model-surgery subsystem exist.
- Default split helper uses seeded, exact-ratio, disjoint **random 60/20/20 splits**, not stratification and not the canonical split of every benchmark. Hold the split constant within paired comparisons; respect official splits where a future benchmark requires them.

### Pruning methods — distinguish mechanism from resulting model format

| Family | Implemented names | Scope/caveat |
|---|---|---|
| Random / magnitude | `random`, `global_magnitude`, `layerwise_magnitude` | Baseline scoring/selection; structural and unstructured modes where supported. |
| Gradient / saliency | `snip`, `grasp` | Gradient-based importance; inspect actual structural mode and computational cost when comparing. |
| Regularization-inspired | `l1_threshold`, `group_lasso` | Not interchangeable: L1-threshold is unstructured; Group Lasso supports channel-level structural pruning. |
| Learnable / gating | `movement`, `hard_concrete_l0` | Existing implementations induce **unstructured sparsity**; do not claim a physically smaller model or realized speedup without compaction and measurement. |
| Adaptive heuristic | `adaptive_layerwise` | Sequential **rule-based** channel pruning with traces. This is **not RL**. Its recorded speedup reward term is zero in the inspected audited implementation. |
| Reinforcement learning | `q_learning_tabular` | An experimental sequential structural **model-channel** pruning agent, not input-graph pruning. See the next section. |

### Tabular Q-learning prototype

- `src/gnn_pruning/rl/{environment.py,q_learning.py,state.py}` implements a discretized, graph-aware state, epsilon-greedy action selection, Q updates, and sequential channel-pruning environment.
- Candidate actions select **hidden layer × prune ratio**, with feasibility checks, local/cascade surgery options, stopping conditions, and training/deployment traces. An explicit STOP action is supported by the pruner.
- Training uses reset episodes; deployment is a separate greedy rollout from the original dense model. These control-flow properties are present in the audited code; Codex also reported checking them in its local test run.
- **Important:** in the inspected RL environment, `speed_proxy = compression_gain`. The reward therefore uses parameter reduction as a surrogate, **not measured latency**. Do not label this prototype hardware-aware or latency-optimized.
- Verified design caveat: accuracy-drop constraint is checked *after* an action; a terminal model can exceed the specified budget. Report feasibility violations instead of assuming the constraint is always met.
- Graph-derived features in the state do not on their own establish transfer/generalization across datasets. The tabular policy has not been established as superior to strong static or rule-based baselines.

## 5. Experimental status and preliminary observations

**Historical / exploratory only:** Earlier coursework and discussions reported non-trivial accuracy–sparsity trade-offs on small citation graphs; fine-tuning sometimes recovered much of the post-prune loss, while other experiments showed marked degradation at high sparsity. Optimal trade-offs varied by model/dataset. These observations are **not canonical quantitative findings** without exact checkpoints, configs, splits, seeds, and compatible metric definitions.

- Flickr integration and GraphSAGE architecture configs exist, but a converged dense Flickr baseline selected exclusively by validation and supported by clean repeated experiments has **not been established in this document**.
- Reddit is supported in the loader/configuration system but is an optional expensive study, not a required thesis dependency.
- The existing compact benchmark covers small citation datasets and static/Q-learning methods, but its published configurations are development-oriented (e.g., 30 dense epochs, no fine-tuning in selected runs). They must **not** be silently treated as a final, fairly tuned benchmark.
- No post-audit canonical numerical results have been verified or entered here. Do not import earlier CSVs into a new aggregate without schema/provenance reconciliation.

## 6. Metrics and interpretation

Core tracked/reportable outcomes: accuracy, macro-F1, requested and achieved sparsity, parameter count/bytes, checkpoint size, inference time, and method-dependent pruning/training/trace diagnostics.

Define and report distinctly:

- **Parameter reduction:** `S_params = 1 - P_pruned / P_dense`.
- **Channel removal:** fraction of applicable channels removed, with a declared denominator/scope; it need not equal `S_params`.
- **Requested target vs achieved outcome:** a static channel target and a Q-learning parameter-reduction target are not directly equivalent even when their numeric values match.
- **Inference acceleration:** normalized speedup on the **same** fixed device/software/protocol, together with absolute latency and timing variability. Parameter reduction does not guarantee latency reduction.
- **Method cost:** pruning-policy training/search overhead, structural surgery, optional fine-tuning, and inference are distinct costs. Realistic end-to-end inference may also include sampling/data movement/graph preprocessing; define its boundaries before making a pipeline-wide claim.

**FACT / VERIFIED REPOSITORY:** The audited pruning workflow sets structured `achieved_sparsity = 1 - P_pruned / P_dense` and records a separate `achieved_channel_sparsity` diagnostic where applicable. This does **not** retroactively correct pre-audit output files; do not mix metrics across schema/protocol versions without reconciliation.

## 7. Experimental protocol — accepted principles and items to lock

**DECISION / minimum principles:**

1. Use a shared dense checkpoint and identical split for methods compared under the same architecture, seed, and experimental condition. Avoid incompatible checkpoint reuse across architectures/seeds.
2. Use training data for model fitting and **validation** for architecture, hyperparameter, pruning-policy, and sparsity-budget selection. Reserve test labels for the final assessment under a frozen selection protocol. Transductive graph features may be visible in node-classification tasks where explicitly permitted; do not use test labels for selection.
3. Use a cheap one-seed smoke stage, then a small multi-seed pilot; perform the final repeated study only for shortlisted conditions. Separate pilot results from final estimates.
4. Compare accuracy against **achieved parameter reduction** and actual measured speedup; report target misses, early stops, and accuracy-budget violations explicitly. Also report results at comparable realized compression levels rather than equating nominal targets.
5. Use matched hardware/software, inference mode, graph/data scope, warm-ups, repeats, and synchronization for latency comparisons. Do not pool latency from Colab, a university server, and unrelated cloud instances as if it were one hardware benchmark.
6. Store Git SHA, resolved config, input graph/split identity, seed, model and pruning checkpoints, environment version, hardware, and output files. Historical, pre-audit results and corrected post-audit results belong to separate provenance groups.

**OPEN — lock before final runs:** primary model-selection metric and tie-breakers; valid architecture/search budget for each method; official vs random dataset splits; number of independent seeds and interval estimator; whether/when fine-tuning is allowed; accounting for RL policy-search overhead; which deployment setting and actual latency endpoint to measure.

## 8. Compute and software constraints

- Existing resources discussed: local laptop/CPU and notebook services (including Colab); a university server with older Tesla P100 GPUs has been available in principle but currently faces driver/CUDA/library compatibility and reliability issues. Do not assume stable production access to all GPUs.
- Research should be executable as **small local/debug → moderate core → limited large validation**. Avoid requiring expensive search over many datasets and episodes just to formulate the central claim.
- **FACT / VERIFIED REPOSITORY:** `pyproject.toml` now declares `requires-python >=3.10` (formerly `>=3.8`), matching existing type-annotation syntax; this is a package compatibility declaration, not proof of CUDA compatibility or a required OS-wide Python upgrade.
- **OPEN:** record actual Python, PyTorch, PyG, CUDA, driver, GPU and CPU versions from the audited/tested environment; determine a portable fallback environment before designing the final study.

## 9. Research candidates — deliberately not final decisions

- **Model-channel pruning with adaptive resource allocation:** assess whether sequential adaptive decisions improve the quality/compression/latency frontier over fixed or greedy policies.
- **Learnable/differentiable structural pruning:** would require a clearly defined channel-gating/compaction mechanism beyond the current unstructured Hard Concrete/Movement implementations.
- **RL-guided structural model pruning:** tabular Q-learning is implemented as a prototype, but an implementation and an interesting reward design are not yet a demonstrated research gap or a reason to make RL the thesis's mandatory contribution.
- **Input-graph pruning / graph sparsification:** distinct object, requiring its own baseline and data-integrity evaluation.
- **Joint graph-and-model pruning:** broad, higher-dependency candidate; do not assume it is manageable or novel without a focused literature review and pilot.
- **Mechanistic/analytical investigation:** relate sensitivity to architecture, graph properties or inference cost; choose a tightly scoped, falsifiable hypothesis rather than merely reporting that performance varies.

Historical conversations sometimes treated RL or inference-pipeline optimization as a working direction; the present conversation explicitly reopened the choice. **Current decision: retain these as candidates until literature, compute and baseline evidence are reviewed.**

## 10. Open scientific questions

1. Which pruning target is the main object of study: model channels, graph elements, or both? Which deployment/inference scenario is being optimized?
2. What is the narrow, falsifiable claim and the documented gap relative to the most relevant prior methods?
3. Which fixed/static, greedy/adaptive, and learned-policy baselines have comparable optimization and compute budgets?
4. How does achieved compression relate to actual latency and to the non-model costs of the chosen GNN inference pipeline?
5. Which datasets and official/random split protocols are needed to test generality without making large compute a single point of failure?
6. Which experimental design avoids reusing the final test set during iterative research decisions?
7. What mathematical analysis is feasible and genuinely connected to the experimental claim?

## 11. Next actions and completion criteria

**Completed — audited code snapshot:**

- Published post-audit commit `882b3c97385a43361e417ff02f8e6854899ede85` in `main`; inspected the public source for the revised aggregation, model width serialization, pruning sparsity definition, validation-selection helpers, docs and Python metadata.
- Previous code-freeze TODOs such as *commit audit fixes*, *push audit fixes*, and *insert the audited SHA* are now resolved. Do **not** rerun cleanup or rewrite commit history just to re-establish this baseline.
- Codex reported 198 local tests passing; retain this as a reported test result until its exact environment/log is archived or independently reproduced at the pinned SHA.

**Next — finalize provenance and methodology before scale:**

- If the research-state file is added to Git, use a separate documentation commit and keep `AUDITED_CODE_COMMIT_SHA` pinned to the code commit above. Check the worktree status on the actual development machine and preserve the final test log and environment versions.
- Lock a validation-based architecture/pruning-budget selection protocol and rerun a **small canonical pilot** into new directories. Do not treat pre-audit results as post-audit evidence.
- Validate suite aggregation with real run artifacts and at least three independent seeds; inspect raw/aggregate rows, achieved compression and stop reasons. Do not infer correct real-world statistics from unit tests alone.
- Establish a validation-selected, converged Flickr dense baseline **only if justified by compute**; postpone expensive Reddit and full-grid studies.
- Conduct focused literature mapping and select the research object, hypothesis and feasible compute budget; Q-learning remains an implemented *candidate/baseline*, not an accepted thesis direction.

## 12. Rules for future ChatGPT / Codex sessions

- Start from the **audited code SHA** and this file, then inspect current code before proposing changes.
- Treat `FACT / VERIFIED REPOSITORY`, `REPORTED TEST RESULT`, `DECISION`, `CANDIDATE`, and `OPEN` literally. Never claim that a proposed method is already the selected thesis topic.
- Do not infer empirical success from passing unit tests or from an algorithm's presence in the repository.
- No expensive full-grid experiments until the selected pilot is valid and the cost has been estimated.
- Update this document only when evidence or an explicit decision changes the project state. Keep detailed logs/results in separate versioned artifacts; do not turn this file into a dump of conversations.

## 13. Freeze record — audited code pinned; environment / benchmark pending

```text
AUDITED_CODE_COMMIT_SHA = 882b3c97385a43361e417ff02f8e6854899ede85
PRE_AUDIT_CODE_SHA = 7b0f28089112c163d2aa19741f8ec0d4a4d5eae6
GIT_BRANCH = main (remote, publicly verified at the time of inspection)
PUSHED_TO_REMOTE = YES (audited commit verified on GitHub)
WORKTREE_CLEAN_AFTER_COMMIT = UNKNOWN (local machine not accessible here)
PYTHON_MINIMUM_DECLARED = >=3.10
PYTHON_VERSION_TESTED = PENDING (record exact local Conda interpreter version)
TORCH_VERSION = PENDING
PYG_VERSION = PENDING
CUDA_DRIVER_GPU = PENDING
TEST_COMMAND = /usr/local/Caskroom/miniconda/base/envs/gnn_pruning/bin/python -m pytest -q (Codex-reported)
TEST_RESULT = 198 passed (Codex-reported, not independent CI verification)
TEST_LOG_AT_AUDITED_SHA = PENDING
CANONICAL_EXPERIMENT_OUTPUT_ROOT = PENDING (new, post-audit runs only)
SELECTION_PROTOCOL = PENDING (validation-based, to lock before final test interpretation)
RESEARCH_STATE_DOCUMENT_COMMIT = PENDING (separate documentation commit if published)
```
