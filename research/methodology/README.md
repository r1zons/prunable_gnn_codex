# LOCAL-STRUCTURED-V2 methodology — audit workspace

**Canonical document:** [LOCAL_STRUCTURED_V2_SPEC.md](LOCAL_STRUCTURED_V2_SPEC.md). Status: ready for an **independent audit**, not approved for code implementation.

Read in order:

1. [Complete V2 specification](LOCAL_STRUCTURED_V2_SPEC.md): user-approved methodology, experimental minimum (Cora/CiteSeer/PubMed), local workstation + POLUS. Additional datasets are optional if time permits. Neither Kaggle nor Colab is planned for V2.
2. [Mask scoring, prefix sums, and physical-surgery equivalence](MASK_SCORING_AND_EQUIVALENCE.md).
3. [Future research backlog](../backlog/IDEAS.md): Cascade vs Local ablation, alternative Q rewards, large models etc. — not core paper scope.
4. [Independent audit assignment](INDEPENDENT_AUDIT_BRIEF.md): detailed read-only red-team instructions.
5. [Earlier V2 decision draft](LOCAL_STRUCTURED_V2_DRAFT.md): historical discussion ledger, not authoritative where it differs from the complete specification.

Other historic evidence: `RESEARCH_STATE.md`, `research/README.md`, `research/experiments/kaggle_pilot_2026/PLAN.md`. The older Kaggle pilot is **not** the V2 execution environment. Do not silently edit historic evidence or run notebooks.

Next: pin exact repository commit SHA, carry out independent audit in Work, review blockers, and seek user authorization **before any source-code changes or experiments**.
