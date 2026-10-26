# LOCAL-STRUCTURED-V2 — draft methodology / decision ledger

**Status:** DRAFT, NOT APPROVED FOR IMPLEMENTATION. Prepared 2026-10-26 from researcher discussion. Existing V1 code/results and legacy names MUST NOT be changed. This is a protocol proposal, not empirical evidence.

**Scope:** post-training physical structural hidden-channel pruning for PyG GCN and GraphSAGE node classification. Graph vertices/edges, input feature dimensions and output classes are NOT pruned. Dense checkpoints are fixed and shared across methods in any paired comparison.

## 1. Decision status (do not silently convert OPEN into APPROVED)

| Topic | Status | Decision / constraint |
|---|---|---|
| Separate V2 registered methods | APPROVED | Preserve V1 cascade implementations, output files and identifiers. New local alternatives get distinct names/configuration identifiers. |
| Surgery | APPROVED | One shared **local physical surgery** for all V2 methods; prune output channels of one hidden conv and corresponding inputs of next conv, handle PyG GCNConv and SAGEConv; NOT mask-only final models. |
| Allowed layers | APPROVED | Any hidden layer only; never input feature dimension or output class channels. |
| Minimum hidden width | PROVISIONAL APPROVED | Four channels per hidden layer. |
| Compression metric | APPROVED | Actual reduction in trainable model parameters: s=1-P(M)/P0, including disappearing biases. No inference speedup inferred from sparsity. |
| Static V2 | APPROVED IN PRINCIPLE | One dense-model scoring stage; candidate complete hidden-channel masks; one mask per feasible width tuple via per-layer rankings; find best candidate by method-specific surrogate. Approximate, not global mask optimization. |
| Static budget | APPROVED IN PRINCIPLE | Target s* with tolerance 0.01 (±1 percentage point) configurable; choose candidate in feasible target band. If infeasible, explicitly report nearest feasible attainable compression and gap. Candidate tie-break and score orientation still require full tests/specification. |
| Static methods | APPROVED IN PRINCIPLE | Random, Magnitude, SNIP, conditional GraSP. Group Lasso excluded from core V2 study. |
| GraSP | CONDITIONAL | Correct Hessian-gradient product, preserve sign and justify group/channel adaptation. Failure of mathematical/numerical gates => exclude; do not silently substitute a heuristic. |
| Adaptive V2 actions | APPROVED | (hidden_layer_index, k) where k in {1,2,4,8} absolute channels from current layer. Invalid actions filtered, not clipped into duplicates. Allow any order and repeat layers. |
| Adaptive V2 greedy | APPROVED | Evaluate feasible candidate actions with shared immediate reward; account for applicable terminal target penalty. Requires actual validation evaluation for candidate lookahead; log candidate counts and wall time. |
| Q-learning V2 | APPROVED | Retain tabular policy and explicit STOP action; share action-level reward and structural action feasibility; apply local surgery. |
| Immediate reward | APPROVED FOR PAPER | Existing Q reward uses speed_proxy=Δs, so R=0.6Δs-0.4D_t, D_t=max(0,A_val(dense)-A_val(current)); 0.6 is param-reduction proxy, not measured latency. |
| Target tolerance and terminal penalty | APPROVED FORM, OPEN COEFFICIENT | eps=0.01 default configurable; P_terminal=-lambda*max(0,abs(s_T-s*)-eps). lambda to choose using predeclared validation-only pilot. Reconcile with existing terminal bonus/gap penalties to avoid double counting. |
| Adaptive stopping | APPROVED | STOP automatically when s_t >= s* - eps (including overshoot), or max_steps reached, or no feasible action; record overshoot even beyond band. |
| Q stopping | APPROVED | Explicit agent STOP, max_steps or no feasible action; do NOT automatically stop at target, negative reward or accuracy-drop threshold. Q STOP can be available under predeclared common rules, potentially excluding zero-step early STOP (OPEN). |
| Horizon | PROVISIONAL APPROVED | max_steps=20 for both; Q number of training episodes subject to pilot (current 30; consider higher if needed). |
| Final reported model | APPROVED | Model from actual termination of algorithm, not retrospectively selected best checkpoint. Log trajectory and optionally retain intermediate snapshots for diagnostic analysis, NOT post-hoc replace final model. |
| Accuracy drop budget | APPROVED PHILOSOPHY | Validation-only feedback; no mandatory auto-stop at 7 percentage-point drop. Preserve loss/violation logs. Test set never used in actions/tuning. |
| Additional analyses | OPTIONAL | Spearman ranking similarity, Jaccard removal overlap, one-shot vs iterative Magnitude control if paper resources allow; latency separately measured. |
| Data expansion | LATER | Larger and structurally varied datasets only after freezing methodology; avoid uncontrolled interaction with new code. |
| External independent review | MANDATORY BEFORE CODE | Audit independent of discussions using this spec + reference papers + repository code; see INDEPENDENT_AUDIT_BRIEF.md. |

## 2. Existing implementation issues requiring V2 checks

- Legacy static methods score the first hidden layer and apply cascade surgery; unsuitable for direct six-method V2 comparison.
- Existing Q local/cascade configurable surgery; Adaptive currently cascade and monotone layer order.
- Adaptive currently uses 10% steps, 10 max; Q uses 5/10% steps, 20 max and STOP. V2 requires distinct implementations or versioned method names.
- Existing GraSP computes grad(grad(loss) dot W), yielding H*W + g rather than desired H*g (under usual derivatives), and takes absolute value, losing sign. Must not advertise it as canonical GraSP.
- Existing Group Lasso is group-regularized one-backward scoring proxy, NOT full group sparsity regularized training. Do not include as real Group Lasso.
- SNIP/GraSP originals are before-training weight pruning; this study is after-training hidden-channel pruning. Names, manuscript, and methods must reflect adaptation and limits.
- Existing full-state penalty D_t is counted repeatedly across successive steps; for this paper retain it to avoid new scientific factor. Alternative potential-difference reward in backlog.
- Existing Q terminal bonus and target early-stop penalties require reconciliation with the proposed symmetric terminal penalty, without changing pre-agreed base reward.
- POLUS and local environment/checkpoints may differ despite same experimental commit and shape. Do not pool or claim matched weights without hashes/numerical checks.

## 3. Planned naming (NOT YET APPROVED)

- V1 legacy names remain: random, global_magnitude, snip, grasp, group_lasso, adaptive_layerwise, q_learning_tabular.
- Potential new names: random_local_v2, magnitude_local_v2, snip_pt_local_v2, grasp_pt_local_v2, adaptive_greedy_local_v2, q_learning_local_v2.
- 'pt' means post-training, NOT PyTorch. These are scientific adaptations and should not be mislabeled as unmodified original algorithms.
- For legacy GraSP historical analysis, describe actual proxy rather than claiming canonical GraSP.

## 4. Required evidence and logs

For each run: Git SHA, environment Python/Torch/PyG, hardware, model and dataset, frozen dense checkpoint hash, split indices/hash, seed(s), method and V1/V2 name, scoring train subset, eval/train mode, target and achieved parameter reduction, target deviation, hidden widths and original kept indices, removal mask, accuracy/F1, raw inference time with repeats, evaluation split, pruning-search time, validation calls, candidate count, selected actions, rewards/terminal penalty, STOP reason, model state or checkpoint hash, acceptance status, source of randomness and tie break.

## 5. Must-pass acceptance tests before publication experiments

1. Across GCN/SAGE, complete masks from dense model can be converted to local physical surgery in any hidden-layer order; final models have matching surviving weights, logits (eval), and parameter counts.
2. Precisely matched activation/weight-masked dense forward is numerically equivalent to physically rebuilt model for valid masks. Ordinary torch.nn.utils.prune mask alone does NOT guarantee correct mask semantics.
3. Count only unique removed parameters for surrogate mask score; parameter-count reduction must include biases. Ensure proper handling of SAGE neighbor/root projections.
4. Prefix-sum full-mask score matches direct enumeration of retained parameter scores for random masks/widths, including signed scores and biases.
5. Same scoring input/seed/split on same dense checkpoint; score computation no test leakage; determinism/ties/NaN/Inf documented.
6. Feasible action filtration, min width=4, k integer actions, repeat and nonmonotone layer order, STOP, max steps, target overshoot and unreachable target all tested.
7. Q training episode reset and deployment correct; legacy outcomes unaffected and are never mixed with new V2 results.
8. Guard against comparing static and adaptive outputs at wildly different achieved compression; analyze achieved sparsity vs accuracy, report target gap, do not misstate apples-to-apples comparisons.

## 6. Open scientific points to resolve before implementation

- A. Precise full-mask surrogate score for each method, including signed GraSP direction and bias treatment; candidate ranking may approximate coupled removal.
- B. Random sampling rule over feasible widths/masks, deterministic seed, avoid random-with-oracle-selection bias.
- C. When eps=0.01 is unachievable: nearest feasible rule based on all width pairs, handling target tolerance in terminal penalties and whether to renormalize, if at all.
- D. Penalty lambda and reconciliation with existing terminal reward; STOP eligibility at step zero; tie-breaking; candidate validation budget fairness.
- E. Q episodes and convergence diagnostics, compute-cost reporting; external environment and checkpoint reproducibility.
- F. Distinguish optional ablations from mandatory core article runs.

## 7. Evidence trail

- Current repository: `src/gnn_pruning/pruning/methods.py`, `src/gnn_pruning/rl/environment.py`, `src/gnn_pruning/surgery/structural.py`, `configs/experiments/kaggle_pilot/cora_gcn_l3_h32.yaml`; verify against actual implementation SHA when reviewing, not just this document.
- Existing research hub: `research/README.md`, `RESEARCH_STATE.md`, `research/CLAIMS.md`, `research/EVIDENCE.md`.
- Literature to verify explicitly: original SNIP (Lee et al. 2019), GraSP (Wang, Zhang, Grosse ICLR 2020), network-channel architecture search (MetaPruning, ICCV 2019; Revisiting Random Channel Pruning, CVPR 2022), optimization with resource constraints (HALP, NeurIPS 2022), RL pruning (AMC, ECCV 2018; PuRL). The citations do not automatically establish identity of our adapted algorithms with originals.
