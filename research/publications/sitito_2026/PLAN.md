# SITITO / VMK MSU — first manuscript plan

**Status:** draft research/editorial plan, 2026-10-20. **Goal:** submit in the November 2026 VMK MSU conference cycle *if* a defensible manuscript can be prepared in time; selection for the journal is **not** guaranteed.

Target journal: «Современные информационные технологии и ИТ-образование». Verify whether the manuscript is to be entered **via the conference** or as a **direct journal submission**, because templates/abstract requirements can differ.

## Existing work — do not present as newly performed experiments

- Coursework already evaluated structural hidden-channel pruning on GCN / GraphSAGE using Cora, CiteSeer and PubMed.
- Already included random, global/layerwise magnitude, SNIP, and Q-learning.
- Already included accuracy vs **achieved** compression and accuracy-drop budgets (0.05 / 0.07 / 0.10).
- Dense model architectures were preselected for strong pre-pruning task performance, per researcher; the exact selection method still needs archival confirmation.
- Coursework reported no consistent realized inference acceleration in the evaluated setting.

## Immediate scientific work

1. Recover the exact source artifacts/protocol behind course figures and the dense-model selection.
2. Build a manuscript claim-to-evidence matrix using `research/CLAIMS.md` and `research/EVIDENCE.md`.
3. Identify a single supported main conclusion. Do not assume that “Q-learning beats baselines” has been established by statistical evidence.
4. Only if a material evidence gap remains, design the **smallest possible** additional experiment and approve the cost first.
5. Review current venue template, publication policies, abstract and reference requirements from the official site before finalizing.

## Intended paper positioning (provisional)

**Working title:** «Исследование адаптивного структурного прунинга графовых нейронных сетей на основе табличного Q-learning».

A fair and narrowly scoped experimental contribution is acceptable as an **editorial working hypothesis**; venue acceptance and sufficiency of novelty remain unverified. Distinguish model compression from real inference speedup. Avoid declaring the method hardware-aware.

## Dates — verify official current guidance

- Internal target: recover data and settle main conclusion by late October 2026.
- Conference registration/final submission were previously discussed as around **2 / 9 November 2026**, respectively; **not a verified deadline in this repository**. Recheck official organizers' pages immediately.
- Do not submit under deadline pressure if scientific claims cannot be supported.

## Completion criteria

- Every empirical statement in the draft points to one evidence entry and traceable results.
- No test-label leakage in model selection/adaptive policy (if not verifiable, explicitly qualify).
- Any final graph uses a clear metric definition and the exact dataset/model/selection condition.
- Manuscript follows the current venue template and authorship requirements.
- Scientific supervisor approves final submission.

## Separate downstream paper

ПаВТ-2027 is a possible later checkpoint on actual latency/compute behavior, potentially including a same-size compact GNN trained from scratch. Reusing the same prior result as a separate publication is not sufficient.

## Urgent reassessment after local Codex audit (2026-10-22)

The existing coursework comparison cannot currently support a headline of Q-learning superiority. Codex reports **only two complete seeds**, nonmatching historical definitions of achieved sparsity, Q-learning target misses, and non-equivalent static cascade versus RL local structural action spaces. See `research/audits/2026-10-22_local-codex.md`.

**Priority for the November deadline**: first reanalyze historical CSV/JSON without rerunning, using `1 - P_pruned/P_dense` for *all* structural methods, per-seed reporting, validation-only feasibility and compression-matched comparison. Recover the actual architecture selection criterion and map each selected figure back to input artifacts. A 3–5-seed canonical subset is a conditional follow-up, not authorization for a full grid.

**Avoid**: claiming a strict accuracy-loss guarantee (post-action checks can overshoot); presenting global vs layerwise magnitude as distinct structural baselines when identical; inferring RL-specific superiority from an experiment with different structural-surgery freedom; claiming measured latency was a trained objective.

Immediate administrative checkpoint: confirm registration and manuscript submission rules directly with the conference organizers; researcher reported not registered as of 2026-10-21.
