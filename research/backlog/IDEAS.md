# Research backlog — deferred ideas and ablations

**Status:** research ideas only, NOT part of the obligatory LOCAL-STRUCTURED-V2 paper run. Date: 2026-10-26. Prioritize fast, credible, reproducible article completion over speculative improvements.

## B01 — Compare search space and surgery (user-approved deferral)

Hypothesis: the same importance criterion (e.g. Magnitude) may behave differently under alternative spaces of allowable hidden-channel removals and surgery.

Possible controlled comparisons:
1. Legacy cascade operation with matched search policy, if technically possible.
2. Local surgery with restricted layer/width selection.
3. Local surgery with static width-pair enumeration.

**Essential confound:** Changing surgery and candidate space simultaneously is NOT a one-factor ablation. Design paired controls and match achieved parameter sparsity before making causal claims. Measure accuracy, inference, selected channels, structural widths, and search cost. Do NOT include this ablation in the current paper unless core publication work is complete.

## B02 — Reward as change in model utility (deferred)

Current paper reward: R_t=0.6(s_t-s_{t-1})-0.4*max(0,A0-A_t). The accumulated accuracy loss is charged repeatedly across multiple steps.

Alternative: U_t=0.6*s_t-0.4*max(0,A0-A_t); R_t=U_t-U_{t-1}. With undiscounted returns the sum telescopes to U_T-U_0. With Q discount gamma<1, terminal rewards and STOP this distinction needs formal analysis. For myopic greedy choice from the same state the two immediate rewards differ by an action-independent constant (without terminal adjustments), so greedy action ranking may be identical. Test effect chiefly on Q policy; do not silently switch the paper reward.

## B03 — Preference scenarios / multi-objective RL

Evaluate three objectives: compression-priority, accuracy-priority, balanced; to change Q behavior, re-train with separately predeclared reward weights. Distinguish retrospective Pareto interpretation of results from re-optimizing a new objective. Possible joint target: desired compression and desired accuracy drop, with appropriate penalties and STOP rule. Reserve for later.

## B04 — Action space for wider models

For H32, V2 candidate fixed integer channels to remove is k∈{1,2,4,8}. For larger hidden dimensions consider e.g. {1,2,4,8,16,32} or adaptive scale; test Q sample complexity, agent performance, overlarge-step overshoot and computational cost. Do not assume extra actions always help.

## B05 — Q training budget and convergence

Current 30 episodes; pilot 60/100 if necessary, log validation calls, policy quality, reward and stop-reason distribution across seeds, timing. Separate training episodes from max pruning steps. Avoid tuning against test data.

## B06 — Group Lasso proper baseline

Legacy 'group_lasso' scoring is not classical train-time group regularization. Full approach would train with group norm penalty and carefully defined non-overlapping/overlapping neuron groups for PyG GCN/SAGE, tune lambda, then perform physical surgery. This changes weight-training budget and breaks simple same-checkpoint comparisons; scientifically risky under deadline. Exclude from current main study. Torch-Pruning may serve only as a reference until compatibility is verified.

## B07 — GraSP theory and post-training sensitivity

Study whether signed H*g scoring developed for preserving trainability before training predicts post-training structured pruning outcomes without fine-tuning. Compare validation-based marginal effect against signed surrogate, check near-stationary gradient behavior, second-order numerical tests and channel aggregation. Core V2 includes GraSP only if corrected implementation passes formal gates.

## B08 — Larger and diverse graph datasets

After method freeze, diversify: homophilic vs heterophilic, datasets with varied node/edge counts and degree patterns, nonuniform architectures, and resource-cost measurement. Consider Flickr, Actor/Texas/Cornell/Wisconsin, Amazon Computers, Reddit and potentially ogbn-arxiv with its official split and evaluator when supported. Do not infer dataset availability equals completed validated experiments.

## B09 — Test masked dense forward vs physical local surgery

As an acceptance test, construct correct masked-reference inference for GCN and SAGE (including hidden bias and paired downstream projections) and compare logits to local surgery for representative masks. This does not turn masked networks into valid final pruned models.

## B10 — Runtime and architecture effects

Profile cost of saliency computation (including SNIP/GraSP gradients), static width-pair scoring with prefix sums, greedy candidate validation, Q policy training, physical surgery and final inference separately. Study whether legacy cascade reduces planning cost for higher-dimensional networks; do not assert physical cascade is necessarily faster/stronger.

## B11 — Literature/reference implementations audit

Use official PyTorch torch.nn.utils.prune (random_structured, ln_structured, custom_from_mask, remove) as a mask-semantic and scoring reference; note these normally do not physically compact layers. Review Torch-Pruning's DependencyGraph, GroupMagnitudeImportance, GroupTaylorImportance and RandomImportance to avoid reinventing established components. No new dependency in current article unless separate compatibility tests justify it.
