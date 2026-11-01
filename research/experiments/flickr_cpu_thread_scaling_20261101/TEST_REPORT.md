# Local validation before native submission

Existing isolated verification environment: Python3.12.9, Torch2.2.2, PyG2.6.1;
CPU only. This is local regression evidence, not POLUS timing or native completion.

- Focused initial tests:12 PASS/0 FAIL/0 SKIPPED,8.06s.
- New adapter plus budget-V5, frozen cache and Flickr structural regression:
  158 PASS/0 FAIL/0 SKIPPED,25.10s.
- Final focused selection with the additional matched-state numerical-difference
  check: 13 PASS/0 FAIL/0 SKIPPED,6.74s. Selections overlap; do not sum counts.

Tests independently compare instrumented and untouched calls using real tiny GCN
and GraphSAGE forwards at both targets: all decisions/states/rewards/Q values,
retained identities, final tensors, RNG state, cache accounting and three complete
episodes agree exactly. Separate checks cover physical cores versus SMT, original
forecast limits, genuine uncached forward counts, bounded20-slot allocation,
serial fresh-process order, INTENT acceptance/ambiguity and duplicate prevention.

No full pytest suite rerun is claimed. Scientific engines are not edited. Raw
outputs: ${GNN_ARTIFACT_ROOT}/flickr-cpu-thread-scaling-20261101/*tests.log.
