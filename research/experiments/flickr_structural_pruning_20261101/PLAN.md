# Flickr structural pruning — predeclared56-condition campaign

Four exact verified Dense seed101 checkpoints from the collected Flickr Dense
sweep: GraphSAGE3/128,4/128,4/256 and GCN2/256. Each checkpoint is shared by all
seven methods and both30%/50% parameter-compression targets. Generated60/20/20
split seed101 is frozen. Dense seed, split seed and policy seed101 are distinct
fields, not independent repetitions. No Dense training or fine-tuning. No held-out
test labels or metrics; the uploaded graph already masks test labels to−1.

Methods: Static Random, Static Magnitude, Static SNIP, Sequential Random,
Adaptive Greedy, exhaustive receding-horizon Lookahead depth2, tabular Q-learning.
All have new `structural_budget_v5` identities. Historical methods/results remain
unchanged. Changes and limitations: [METHOD_CHANGES.md](METHOD_CHANGES.md).

Common minimum hidden width4, adaptive deletion sizes{1,2,4,8}, exact physical parameter
counts including biases, symmetric target band±0.01. Shared finite action horizon
is the initial number of removable hidden channels, not20. If the band is empty,
all methods use the globally closest feasible width tuples and explicitly label
the outcome outside-band; an unchanged model cannot be successful pruning.

Reward remains0.6×compression gain−0.4×max(0,dense validation accuracy−current
validation accuracy), plus the original lambda2 terminal gap penalty exactly once.
No reward/architecture/accuracy tuning. Q uses30episodes (the preceding full
campaign's budget), alpha0.1, gamma0.9, epsilon0.2, policy seed101; deterministic
enumeration-order exploitation ties, including unseen zero values. Training
episodes reset to the same Dense tensors and deployment resets again.

Gates: synthetic depths2/3/4/both models/all methods; independent geometry,
physical masked-reference/strict reload, real-forward depth2 oracle, target-aware
Q and evaluation accounting; full pytest regression; all four archived Dense
input/hash/metric checks; native four-thread technical smoke for all methods,
including4/256. Smoke uses target50%, static complete plans, one full Q training
episode and at most two adaptive deployment actions. Technical prefixes remain
outside research output roots and are never described as completed conditions.
All eight target geometries are checked before submission.

Native measurements drive a resource forecast, never a quality gate. Forecast
uses the harder50% trajectory prefix for both targets,31× one-episode Q cost,
1.75× safety plus180seconds; unbound jobs receive an additional1.75× host margin.
A condition above170minutes or32GiB safety memory
fails the complete matrix gate; no method, target or episode is silently removed.
Forecasts are not worst-case runtime guarantees; variable selected paths remain
a limitation. Queue waiting is separate. Recheck current queue/hosts/user jobs.

After all gates PASS, submit all56 independent jobs at four threads, exclusive,
normal queue,180minutes,32768MB. Large4/256 and Lookahead/Q jobs prefer the verified
C4 node by hard binding; other work is unbound. No inferred CPU-frequency labels
or threading optimization. INTENT precedes exactly one bsub; ambiguous responses
halt, and existing claims cannot be silently retried.

After scheduler acceptance, record IDs, hashes, resources and receipts locally
and stop. Do not wait for research completion or collect scientific results in
this launch session. No push/merge is authorized by this task. Future collection
requires56 DONE identities and complete inventories before analysis. Prepared
analysis reports all outcomes and actual compression; a single seed cannot
establish method robustness or causal architecture effects.
