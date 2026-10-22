# SITITO / VMK MSU — first manuscript plan

**Status:** active research/editorial plan, revised 2026-10-22. **Only immediate objective:** prepare a scientifically defensible submission by the researcher-designated target of **2026-11-02**. Selection for the journal is not guaranteed.

Target journal: «Современные информационные технологии и ИТ-образование». Verify whether the manuscript is to be entered **via the conference** or as a **direct journal submission**, because templates/abstract requirements can differ.

## Existing work — do not present as newly performed experiments

- Coursework already evaluated structural hidden-channel pruning on GCN / GraphSAGE using Cora, CiteSeer and PubMed.
- Already included random, global/layerwise magnitude, SNIP, and Q-learning.
- Already included accuracy vs **achieved** compression and accuracy-drop budgets (0.05 / 0.07 / 0.10).
- Dense model architectures were preselected for strong pre-pruning task performance, per researcher; the exact selection method still needs archival confirmation.
- Coursework reported no consistent realized inference acceleration in the evaluated setting.

## Immediate scientific work

1. Preserve the historical audit and use it to constrain, not inflate, manuscript claims.
2. Prepare the separate canonical pilot defined in [`../../experiments/kaggle_pilot_2026/PLAN.md`](../../experiments/kaggle_pilot_2026/PLAN.md).
3. Run seed 101 only after configuration, compatibility, artifact and runtime checks are reviewed and the researcher manually starts the Kaggle Notebook.
4. Freeze the protocol using validation evidence only; then extend to at least three complete seeds if the pilot passes.
5. Build the manuscript claim-to-evidence matrix in parallel with experiment preparation. Do not assume that “Q-learning beats baselines” will be the conclusion.
6. Review the current venue template, publication policies, abstract and reference requirements from the official site before finalizing.

## Intended paper positioning (provisional)

**Working title:** «Исследование адаптивного структурного прунинга графовых нейронных сетей на основе табличного Q-learning».

A fair and narrowly scoped experimental contribution is acceptable as an **editorial working hypothesis**; venue acceptance and sufficiency of novelty remain unverified. Distinguish model compression from real inference speedup. Avoid declaring the method hardware-aware.

## Dates — verify official current guidance

- Researcher-designated submission target: **2026-11-02**.
- This date is a project decision, not independently verified organizer guidance. Recheck the official submission path, registration cutoff and manuscript deadline immediately.
- Do not submit under deadline pressure if scientific claims cannot be supported.

## Completion criteria

- Every empirical statement in the draft points to one evidence entry and traceable results.
- No test-label leakage in model selection/adaptive policy (if not verifiable, explicitly qualify).
- Any final graph uses a clear metric definition and the exact dataset/model/selection condition.
- Manuscript follows the current venue template and authorship requirements.
- Scientific supervisor approves final submission.

## Deferred work

Unrelated algorithm development, HPC benchmarking and later publication candidates are deferred until after the November submission effort. ПаВТ-2027 remains historical context only, not an active workstream.

## Urgent reassessment after local Codex audit (2026-10-22)

The existing coursework comparison cannot currently support a headline of Q-learning superiority. Codex reports **only two complete seeds**, nonmatching historical definitions of achieved sparsity, Q-learning target misses, and non-equivalent static cascade versus RL local structural action spaces. See `research/audits/2026-10-22_local-codex.md`.

**Priority for the November deadline**: preserve the historical reanalysis, then execute only the newly approved narrow canonical protocol. The new benchmark uses parameter-count reduction consistently, paired checkpoints/splits, validation-only decisions and explicit method-capability reporting. Its results remain separate from historical debug runs.

**Avoid**: claiming a strict accuracy-loss guarantee (post-action checks can overshoot); presenting global vs layerwise magnitude as distinct structural baselines when identical; inferring RL-specific superiority from an experiment with different structural-surgery freedom; claiming measured latency was a trained objective.

Immediate administrative checkpoint: confirm registration and manuscript submission rules directly with the conference organizers; researcher reported not registered as of 2026-10-21.

## Canonical experiment gate

- Platform: Kaggle Notebooks, manually started by the researcher.
- Pilot: seed 101; Cora/CiteSeer/PubMed; GCN/GraphSAGE L3/H32; up to 500 dense epochs; patience 50.
- Methods: Random, structured Global Magnitude, SNIP, GraSP, Group Lasso, Adaptive Layerwise and tabular Q-learning.
- Q-learning: targets 0.30 and 0.50; validation accuracy-drop budget 0.07.
- Fine-tuning: disabled; report raw post-prune behavior.
- Decision rule: the pilot checks execution, compatibility, convergence, artifact completeness and runtime. It must not select a winner from test results.
- Expansion: seeds 202 and 303 only after seed 101 passes the protocol gate and before publication-level comparative estimates are made.
- Execution restriction: Codex must never start training, pruning, smoke tests or benchmarks without explicit user permission.
