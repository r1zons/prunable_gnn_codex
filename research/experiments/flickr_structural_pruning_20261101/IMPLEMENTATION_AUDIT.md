# Implementation audit

Audit base: `5e9710840dac45be65bc1ff3b4fae78a2d150f42`; previous full POLUS pruning
execution source `55d0fc242b543c5725276144bd8e970eedc48ecd`, identity map preserved
in `polus_complete_pruning_20261031/protocol.yaml` and `complete_campaign.search`.

| Method | Historical implementation | Historical action/stopping behavior |
|---|---|---|
| Static Random | static_v2.plan_static(random) | Uniform eligible width pair, then uniform retained IDs; band or nearest fallback |
| Static Magnitude | static_v2.plan_static(magnitude) | Fixed dense incident rankings; minimize unique removed saliency in band/nearest |
| Static SNIP | static_v2.compute_scores(snip), plan_static | Train-only eval-mode CE gradient×weight absolute saliency; same width optimization |
| Sequential Random | experimental_v3.random_reachable_v3 | Uniform legal reachable layer/deletion, magnitude channels; exact band/max20 |
| Adaptive Greedy | adaptive_v2.run_greedy | All immediate rewards; stable enumeration ties; threshold target−tolerance, permits overshoot |
| Lookahead | lookahead.run_lookahead | Exhaustive r1+.9r2, first action only, replan; same threshold as native Greedy |
| Tabular Q | adaptive_v2.run_q_learning |30episodes/.1/.9/.2, unseen0, stable ties; soft STOP/max20, no target-conditioned key |

The preceding native Q results are not budget-matched controls: all36 reached
max_steps and mean87.441% compression; target pairs shared final masks. They are
not reused or relabeled as corrected results.

Three-layer restrictions exist in `static_v2.plan_static`, `LocalV2Env.__init__`,
and V3/V4 two-width geometry. The audited Lookahead explicitly rejects other env
classes. These modules remain byte-for-byte unchanged. The established surgery
`prune_hidden_channels_local_v2` and model constructors already support arbitrary
hidden-layer indexing/nonuniform widths. It slices selected outputs and biases
plus immediate downstream inputs, updates metadata and clears GCN caches;
GraphSAGE requires mean/root weights/no projection/no normalization. It copies
parent tensors, preserves device/dtype/flags/modes and isolates CPU RNG consumption.

The new controller reuses that exact surgery and existing scoring/ranking,
static application, Bellman update and model reconstruction. It supports1/2/3
hidden layers without touching historical entry points. Parameter geometry sums
all adjacent matrix products (twice for mean/root GraphSAGE) and actual biases.
Static union saliency counts shared removed coordinates once. Optimized retained
score prefix sums are independently checked against exhaustive unique-coordinate
scores. Selection is fixed-rank static, not a sequential Magnitude substitute.

Existing unit suites cover historical scoring, surgery, reset, reward, frozen
loading, V3/V4 reachability and Lookahead. New tests independently enumerate width
goals/continuations, compare masked-reference real forwards at depths2/3/4, check
all seven policies and checkpoints, and verify depth2 decisions/replanning with
independent ranks, rewards and real forward scores. Validation-only callbacks
and training-only SNIP remain distinct; test labels remain masked throughout.
