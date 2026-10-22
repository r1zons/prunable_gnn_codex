# Research history and decision provenance

**Compiled:** 2026-10-21 from accessible earlier Project conversations, historical coursework, the 24 September 2026 presentation, and the existing audited code-state document. **Not an exhaustive verbatim transcript.** Dates and historical figures are contextual, not canonical experimental provenance.

## Timeline

### March 2026 — experimental framework
- Initial PyTorch Geometric node-classification framework for GCN/GraphSAGE; other architectures deferred.
- Reproducibility principles: seeded train/validation/test splits (default 60/20/20), shared dense checkpoint and split for paired method comparisons, real **structural** channel removal rather than permanent masking.
- Dense train → evaluate → prune → evaluate → optional fine-tune → evaluate.
- Methods established in code over time: random, global/layerwise magnitude, SNIP and other static baselines.
- Early mask-only behavior required correction; subsequent examples demonstrated physically reduced model parameter counts. These were historical diagnostic runs, **not** the final article results.

### April 2026 — capacity and interpretation
- Discussion: small citation graphs and/or overparameterized networks could conceal inference benefit or lead to accuracy recovery after fine-tuning.
- Observed similar global/layerwise magnitude curves in certain shallow architectures; could be expected when only one independent hidden group is prunable. Need multi-layer controls to interpret this effect.
- Earlier mixed CSV schema issues prompted special caution about comparing historical output files.
- A possible broader thesis question emerged: in what regime does model/channel pruning materially benefit GNNs? **Not finalized.**

### May 2026 — tabular Q-learning prototype
- Researcher chose real but small RL, initially PubMed/GraphSAGE, without DQN/PPO, without subgraph extension in MVP, and **without fine-tuning inside RL episodes**.
- Agent state: bucketed graph properties and pruning progress. Actions: hidden layer × pruning ratio, plus STOP. Reward trades compression against validation-accuracy degradation.
- RL episodes restart from dense; deployment uses learned greedy policy.
- Early debugging runs mentioned limited visited states and premature max-step stopping; **do not reuse those preliminary figures as final results**.
- Important unresolved tension: graph descriptors stay constant while pruning one fixed graph. Their usefulness for transferable policies was a future hypothesis, not a demonstrated capability.

### Summer 2026 — broader benchmarks and infrastructure
- Explored Flickr and additional architectures/datasets, but large-scale converged benchmark was not established by a GitHub code listing alone.
- Considered comparing adaptive strategy vs static heuristics at matched achieved compression.
- August review raised provenance and repeatability questions: historical CSVs and results were not committed, so public repository does not independently prove empirical performance.

### September 2026 — audited code, research re-framing
- Code audit fixed important framework issues (aggregate statistics, non-uniform structural channel widths/checkpoints, consistent achieved parameter sparsity) and documented audited commit `882b3c97385a43361e417ff02f8e6854899ede85`; see `RESEARCH_STATE.md`.
- Local test report claimed 198 passing tests, **not an independently rerun CI result**.
- The 24 September presentation proposed sequence:
  1. broaden data and assess decision logging;
  2. analyze behavior/choice of actions and graph/model dependence;
  3. only then improve state/actions/reward based on evidence;
  4. investigate cross-graph transfer *without training the policy from scratch*.
- The researcher preferred this sequential evidence-first plan; GA+RL was deferred.
- Prior discussion identified an interpretability/logging gap: existing traces include actions/states/rewards/stopping, but complete rankings or Q-values of all feasible actions, exploration-vs-greedy choice reason and Q-values before/after updates may not be persisted. **Verify current branch and raw traces before implementing more logging.**

### October 2026 — first publication and separate PaVT checkpoint
- Existing coursework **already** compares methods on Cora/CiteSeer/PubMed + GCN/GraphSAGE and covers accuracy vs achieved compression and accuracy-loss budgets. Do **not** present that as a newly proposed experiment.
- Researcher clarified dense baselines were selected from candidate architecture/width grids after dense training by classification accuracy. Example for PubMed/GCN: exploring depth and width choices such as widths 16/32/64; precise grid and **which accuracy split was used** remain to be recovered from configs/local runs.
- User remembers **averaging over three seeds**; this is USER_REPORTED pending original result files.
- **All original result artifacts are stored on the researcher's local computer**, intentionally not on GitHub. They are not missing; not yet inspected by this documentation audit.
- Supervisor prefers submitting through VMK MSU's November 2026 conference for the journal «Современные информационные технологии и ИТ-образование». The registration/submission pathway and cutoff require immediate organizer confirmation; researcher had **not registered** as of 2026-10-21.
- Second, scientifically distinct PaVT-2027 candidate: evaluate inference latency and computational regime for pruned GNNs; a size-matched compact GNN trained from scratch is a promising comparison.
- Researcher offers a locally executed Codex High audit to index local provenance, audit existing methodology and report history/errors before deciding any reruns.

## Status and boundaries

- **Verified code** and **reproduced experiments** are different: code correctness does not establish a scientific advantage.
- Prior chats and slides are secondary evidence of intent/history; **local config/log/result manifests control numerical truth**.
- No additional RL algorithm, broad new benchmark, subgraph transfer experiment, or agent redesign is approved for the November submission.
- Do not use test labels to choose dense architectures, RL policies, or compression hyperparameters; check historical selection behavior before interpreting the graphs.
- Prompt local Codex in English, specify **High** reasoning, request read-only work first, no file modifications, destructive operations or costly reruns, and require evidence paths and exact command/config lineage.

## Next concrete action

Use **Codex High locally** to inventory old results and trace for each coursework graph:
`plot/paper figure → run directories → aggregation CSV(s) → seed rows → resolved config → dense checkpoint/split → Git commit/environment (when available)`.
Report complete, partial and unrecoverable links; do not fabricate missing provenance. Then review the findings with the researcher and update `EVIDENCE.md` and `CLAIMS.md`. Only afterward decide whether any new experiments are needed.

### 22 October 2026 — separate canonical Kaggle pilot approved

- The local evidence audits found that the historical debug benchmark does not establish general Q-learning superiority: only seeds 42 and 43 are complete, dense training used a 30-epoch debug budget, historical sparsity definitions differ, and static/RL structural surgery capabilities are not equivalent.
- The earlier instruction that a new benchmark was merely conditional is superseded by a narrow decision to prepare a **separate** canonical pilot. Historical artifacts remain unchanged and must never be pooled with the new estimates.
- Pilot seed: 101. Datasets: Cora, CiteSeer and PubMed. Architectures: GCN and GraphSAGE with three layers and hidden width 32.
- Candidate methods: Random, structured Global Magnitude, SNIP, GraSP, Group Lasso, Adaptive Layerwise and tabular Q-learning. Method compatibility and structural capability must be verified before execution.
- Dense training may run for up to 500 epochs with validation-loss early stopping patience 50 and convergence diagnostics. Q-learning targets are 0.30 and 0.50 with validation accuracy-drop budget 0.07.
- Kaggle Notebooks replace Google Colab for this experiment. The researcher manually starts every run; Codex must not start training, pruning, smoke tests or benchmarks without explicit permission.
- Fine-tuning is disabled so the primary endpoint is raw post-prune behavior. The pilot is an execution/compatibility/runtime gate, not a one-seed winner-selection study. Extension to seeds 202 and 303 is conditional on seed 101 passing the protocol gate.
- The only immediate publication objective is a scientifically defensible submission by the researcher-designated date 2026-11-02. Unrelated algorithms, HPC benchmarking and later papers are deferred.

The complete prospective protocol is maintained in `research/experiments/kaggle_pilot_2026/PLAN.md`.
