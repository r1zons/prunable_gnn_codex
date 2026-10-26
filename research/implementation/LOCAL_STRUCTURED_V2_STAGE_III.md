# Stage III — Adaptive Local V2 and integration

2026-10-27, after Stage II Green `4df84c5`. Runtime: user-selected conda
gnn_pruning, Python3.8.20 / Torch2.2.2 / PyG2.6.1 / pytest8.3.5.

## Red

`python -m pytest tests/test_adaptive_local_v2.py tests/test_local_v2_integration.py --tb=line -r fEs`

Measured Red: 21 failed, exit1. Adaptive stubs raise NotImplementedError;
integration fails because V2 methods are not registered/routed. No import errors.
Tests precede adaptive implementation and production registration/routing.
Coverage: integer actions/min4/repeat layers; STOP0/crossing/horizon/once-only P;
dense-relative drop; Greedy candidate rewards; legal-only/unvisited/terminal
Bellman targets; exact widths/horizon state; episode/deployment resets and sparse
Q statistics; six V2 identifiers, strict checkpoint reload, full config/provenance
and mismatch rejection without split regeneration. Only synthetic graphs and
short mocked-validation policy episodes, no neural-network fit.

## Green / review

Pending.
