# V5 compatibility audit

The scientific module and all historical implementation/result files remain
byte-identical to parent2da2a9a. New orchestration delegates directly to
`budget_v5.Budget`, `static_plan`, `run`, and existing physical surgery. No policy,
reward, menu, stopping, model tensor or Dense fitting code is altered.

Independent citation geometry tests enumerate every legal width pair4..32 for
both GCN/GraphSAGE and each real feature/class dimension, checking exact rational
band membership, counts including biases, and descendant reachability. Existing
real-forward V5 tests cover physical masked equivalence, arbitrary hidden-layer
surgery, parent nonmutation, strict reload, depth2 independent path/replanning
oracle, deterministic ties, depth1 Greedy reference, closest fallback/one terminal
penalty, Q retained-state identity and Bellman updates. New bridge tests compare
complete native-engine traces, masks/tensors and Q tables against direct unchanged
V5 calls, with real small-graph forward passes and test labels masked.

All18 original checkpoint SHAs match the historical independently audited table.
Strict model reconstruction, full graph provenance before masking, saved disjoint
60/20/20 split, architecture/seed/training protocol and model tensor hashes are
verified. Validation accuracy/F1 reproduced locally within1e−12; validation loss
is additionally saved. No current test evaluation is performed. Native identity
checks repeat on POLUS before technical search.

The exact cache is per run and pins frozen graph/split/input and exact candidate
architecture/tensors. Cache hits never count as actual validation forwards. V5
candidate_evaluations counts newly constructed preview transitions, not necessarily
new neural forward calls; reporting preserves both definitions. Static SNIP has
one separate training-gradient forward. Final validation and inference forwards
are outside policy search; static intermediate validation is explicitly absent.

Limitations: repeated validation queries can overfit the validation split; the
campaign has no pruning-time adaptation of weights and no test-selection defense
beyond isolation. Q state IDs greatly enlarge the state space;30 episodes may
provide sparse deployment coverage. Depth2 is a truncated discounted objective,
not an optimal full-horizon policy; Greedy and Q gamma.9 are distinct objectives.
Static versus adaptive methods share final budget geometry, but static plans do
not traverse the adaptive action menu. Numerical cross-thread equivalence must
be checked rather than inferred from mathematical operations.

No demonstrated scientific correctness defect was found in the local review.
Production remains gated on tests and native technical integrity/resources.
