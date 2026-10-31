# Native V2 tabular Q-learning: observed behavior and mechanisms

This is the fixed 30-episode,alpha.1,gamma.9,epsilon.2,lambda 2 native V2 policy
used in this campaign, not constrained V3/V4 Q. "Deployment" below means the final
pruning-policy rollout, not per-node classifier inference. No training, Q update, new rollout,
retuning or model inference is performed in this analysis. The earlier independent
audit reconstructed Bellman updates; this analysis reads its verified final tables
and recorded observations.

## Confirmed current-campaign observations

- 36/36 deployments terminate `max_steps`, each after 20 successful removals; STOP
  is never selected during deployment. First hidden width is 4 in every run.
- Mean achieved compression over 36 recorded target-labelled outcomes:
  **87.440664%**, range 87.243182–87.708048%.
  Each of 18 dense checkpoints has identical final masks, action sequence and final
  quality for 30% and 50%. These repeated targets are not independent replicas.
- Their Q tables differ in 18/18 pairs. Targets affect terminal rewards and learned
  numbers, but do not change these recorded greedy deployments. The terminal
  penalty for 50% is.4 higher (less negative) than 30% at the shared final model.
- Each deployment crosses target−.01 after 2–8 actions and continues 12–18 more.
  Only 4/36 paths visit the symmetric band at an intermediate step (all target 50%);
  none ends there. Threshold crossing is not equivalent to a band hit.
- Training did use STOP:404/1,080 episodes stop,674 exhaust the horizon and 2 exhaust
  legal removals. STOP is implemented/available; training episodes are process
  observations, not additional trained GNNs or statistical n.

[36 run diagnostics](Q_RUN_DIAGNOSTICS.csv),[720 deployment decisions](Q_DEPLOYMENT_DECISIONS.csv),
[1,080 episode summaries](Q_EPISODE_SUMMARY.csv),[18 target-pair checks](Q_TARGET_PAIR_CHECK.csv).

## Confirmed source/decision mechanisms

Reward is `r_t=.6(s_t−s_prev)−.4 max(0,A_dense−A_t)` plus the once-only terminal
penalty `−2 max(0,|s_t−target|−.01)`. It repeats the dense-to-current drop, rather
than charging only incremental loss. The goal penalty is delayed until STOP,
max_steps or no actions; **crossing the target is not a terminal condition**.
The menu is first layer then second layer, sizes 1/2/4/8 ascending; STOP is last and
legal only after a removal. Magnitude determines channels, policy chooses layer/size.

The state is `(local_v2,(width0,width1),remaining_horizon,floor(accuracy/.01),terminal)`.
Original retained IDs are omitted. Target is fixed per separate policy rather
than encoded as an explicit distance feature. Its absence from the tuple alone
is not proof that a fixed-target MDP cannot learn; the separate tables demonstrably
differ. No test label enters the validation-only reward.

Deployment picks the **first legal maximum** of the sparse learned row, with
unseen actions valued 0. This exactly matches all 720 recorded decisions:

| Diagnostic | Count / denominator | Interpretation |
|---|---:|---|
| Selected action has no learned entry |466/720 (64.72%)|default 0 is often a decision value|
| Entire state has no learned row |328/720 (45.56%)|legal enumeration determines a zero tie|
| STOP legal |684/720|all except 36 initial states|
| STOP tied for maximum |452/684 (66.08%)|pruning appears first and wins the tie|
| STOP has learned entry |78/684|most legal STOP values remain default 0|
| Unseen action beats strictly negative known actions |138/720|confirmed sparse-value optimism effect|
| Selected learned positive value |248/720|early behavior is not entirely random/unseen|

The analytic immediate STOP reward is also recorded from its known formula;
it needs no new model score and is explicitly labelled analytic, not a new
counterfactual forward. Unrecorded pruning alternatives' validation quality is
not estimated from Q values. Q is an approximate return estimate, not accuracy.

First-layer cuts have larger parameter-gain contributions because F≫hidden 32 and
classes. Once the first layer reaches minimum 4, much of the remaining legal menu
is in layer 1. Missing/negative values and ordered zero ties directly explain the
corresponding **observed** action choices, but do not establish that any proposed
alternative would improve final validation quality.

## Coverage, aliasing and credit: association versus hypothesis

Mean recorded learned state-action coverage is 14.479% (one minus table sparsity),
computed per run; this is not a sample-size claim. In training there are
**100 aliased aggregate states among 11938 run-scoped observed state keys**,
and 18 repeated state-action pairs have reward variation beyond float 32 tolerance.
During deployment,334/720 concrete masks were absent from training under
their aggregate state;328 are entirely unseen aggregate states, leaving only
6 known aggregate states with a new concrete mask. Thus the present
records provide much more direct evidence of sparse coverage/default choices
than of aliasing as the dominant cause. State aliasing can mix physical contexts;
its independent performance effect was not experimentally isolated.

A negative terminal return updates the last visited action without bootstrap;
earlier credit propagates through sampled legal next states with gamma.9. A
20-action terminal signal has factor.9^19≈.135 relative to a first action. Sparse
revisitation, greedy-versus-exploratory occupancy and the repeated accuracy-drop
penalty are plausible contributors to poor target-sensitive learning. These
are hypotheses consistent with the traces, not measured causal ablations here.
Thirty episodes is the tested budget; this dataset does not prove that any larger
budget or changed initialization/reward/state would repair behavior.

## What can be compared scientifically?

The Q outcomes are valid audited physical models at roughly 87.4% compression and
measured search cost. Paired Q/Greedy search ratios have median 29.13× (26.62–31.71×),
but their endpoints are not compression matched. It is meaningful to compare
**target compliance, controller behavior and recorded operating points**; it is
not meaningful to infer an intrinsic quality disadvantage of Q-learning from
comparing 87.4% models to 30/50% baselines. No simple baseline was measured at the
same 87.4% operating points in this campaign. Q's36/36 3-objective Pareto memberships
are protected by extreme compression and do not imply useful requested-target control.

## Historical context, kept separate

The [V2 pilot](../../local_structured_v2_q_random_pilot_20261027/README.md) on Cora
and fixed training seed 101 also observed overcompression and target-insensitive
endpoints under 30/100 episodes and targeted lambda diagnostics. Its platforms,
policy repetitions and scope differ; none is added to current rows or n.
The [V3 report](../../local_structured_v3_sprint_20261027/results/REPORT.md) and
[V4 report](../../polus_v4_diagnostic_20261029/results/REPORT.md) study different
controllers/ablations, including exact-band reachability. Those can enforce
feasible target arrival without establishing a learned quality advantage.
Historical V4's92 results are not current Q controls and are never pooled.
These are supporting methodological context, not new results of this campaign.

**Conclusion:** H4 is not supported for compression control by this particular
native V2 configuration on these audited conditions. The strongest direct
mechanisms are no target-stop constraint and sparse/ordered action values.
Delayed credit and representation aliasing remain plausible contributors;
"Q-learning in general cannot work" is not a supported conclusion.
