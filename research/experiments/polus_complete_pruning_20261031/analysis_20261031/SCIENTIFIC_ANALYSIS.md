# Scientific analysis: quality, physical compression and adaptive search cost

This is an offline analysis of the audited 252 native POLUS outcomes at executable
source 55d0fc242b543c5725276144bd8e970eedc48ecd, audit commit 8f24513da09c0291a562fbb31a688da2698671b9.
All 160 WARNING rows and six infeasible outcomes remain present. No training,
pruning, inference, new timing measurement, server access/job, algorithm change,
manuscript edit or test-based method selection is part of this analysis.

## Principal findings, ranked by scientific importance

1. **More search usually did not change the final model.** Lookahead and Greedy
   have identical original retained masks and tensors in 35/36 pairs, although
   only 34/36 action sequences match. At identical outputs, Lookahead costs
   6.64–8.54× Greedy search (median 7.38×). This is strong negative evidence for a
   routine depth-2 quality benefit in this campaign, not a universal theorem.
2. **One rare planning gain is real and budget-confounded.** PubMed/GCN/seed 202/50%
   gains 4.286pp validation accuracy and 4.155pp macro-F1,
   with 0.931pp additional parameter compression. Search grows
   11.69× (0.658→7.696s),
   candidates 16→184. Within that dataset/model/target, only 1/3 dense seeds improves.
   It is not a matched-compression causal estimate or evidence of a general advantage.
3. **A simple static rule duplicates Greedy at 50%.** Static Magnitude and Greedy
   match all 18/18 masks at 50%, with zero quality differences; Greedy's median
   paired search cost is 2.84×. Every Greedy path deletes 8+8 first-layer channels,
   leaving widths 16/32. Conditional magnitude-rank invariance explains the equality.
4. **Native V2 Q does not control requested compression here.** All 36 deployments
   exhaust 20 actions, average 87.441% compression (range
   87.243–87.708%), and all 18 checkpoint pairs
   produce identical masks/actions for 30% versus 50%. Q tables differ, so the
   numerical learning updates do respond to target rewards; the final policy does not.
5. **Sparse/default action values are a confirmed decision mechanism.**466/720 Q
   choices select unlearned action values (Q=0);328 states have no learned row.
   STOP ties for maximum in 452/684 states where legal, yet never executes in
   deployment. Source tie order puts pruning before STOP. Delayed credit and
   state aliasing are plausible contributors, not isolated causal explanations.
6. **Sequential adaptation has condition-specific benefits and counterexamples.**
   Greedy improves over Magnitude at Cora 30% (3/3 seeds for both models), but
   compression/layer allocation differ; CiteSeer 30% is identical, and PubMed
   directions vary. SNIP-PT beats Greedy accuracy for all 3 PubMed/GCN 50% seeds.
   Sequential Random is not uniformly better than Static Random and is not a
   clean sequential-only ablation: actions are random but channels are magnitude-based.
7. **Quality metrics and feasible compression matter.**96/252 outcomes are outside
   the symmetric target band; six Sequential Random CiteSeer 30% results do no
   physical pruning. Greedy versus SNIP at CiteSeer/GCN 50% has mean+3.91pp accuracy
   but−2.47pp macro-F1. No single scalar/nominal-target ranking is adequate.
8. **Pareto membership alone is misleading without target feasibility.** All 36 Q
   outcomes are on the recorded 3-objective frontier because their compression is
   extreme, and Static Random is often protected by low cost.35 Lookahead outcomes
   are dominated by their identical-mask, cheaper Greedy pair. These are recorded
   operating points, not significance tests or universal efficiency claims.

## Design, validity and definitions

The population is 3 datasets ×2 architectures ×3 independently trained dense
seeds ×2 requested targets ×7 methods =252. Models have 3 layers, hidden 32,
dropout.5, no pruning-time fine-tuning. Six native seed 101 dense models are reused
from their verified origins; seeds 202/303 supply 12 further fits. Comparisons pair
identical checkpoint, graph and saved split. The unit is **n=3 dense training seeds
per dataset/model/target**, not 36 independent Lookahead pairs, repeated targets,
1,080 Q episodes,720 Q decisions or additional policy-seed replicates. Policy
seed is coupled to dense seed for stochastic methods; their variances cannot be
separated by this design. There is one graph per dataset, and training seeds also
change the generated 60/20/20 split, so seed variation combines fit and split variation.

Requested compression is a design label; achieved compression is `1−P_after/P_dense`.
Physical pruning requires a smaller architecture/parameter count. Within-band
means absolute deviation≤.01; threshold success and infeasibility are separate
outcomes. All missing costs remain missing: static scoring split timings and Q
candidate-enumeration counts are not replaced by 0; all 18 dense inference baselines
are absent. No dense-to-pruned latency speedup is claimed.

The same validation data supports dense early stopping and adaptive pruning.
Adaptive methods search this metric, while static saliency methods do not receive
the same validation oracle. A validation advantage is therefore not an unbiased
held-out generalization estimate. Existing test outcomes remain available under
the fixed protocol, but are not used for hypothesis or case selection here.

Paired differences are **B−A** in percentage points; cost/evaluation ratios are
B/A. [756 individual contrasts](PAIRED_COMPARISONS.csv) cover all 21 method pairs
within 36 checkpoint/target blocks. [252 paired summary cells](PAIRED_SUMMARY.csv)
each contain exactly 3 paired seeds, with mean difference, sample SD, win/tie/loss
counts, compression differences, mask/width equality and cost-ratio distributions.
A numerical tie tolerance of 1e-10pp suppresses only floating-point noise, far below
one-node accuracy changes. SD is not a confidence interval or standard error.
No hypothesis tests, p-values, pooled global quality score or significance claim
are used. With n=3, distributions and rare-event frequency are poorly estimated;
many exploratory comparisons and a post-inspected case preclude confirmatory claims.

[84 method cells](METHOD_SUMMARY.csv) retain every observation and include actual
parameter counts, validation quality/dense changes, compliance, measured search,
oracle counts, available scoring/surgery costs and measured pruned inference
latency. Test outcomes remain in the original audited dataset but do not enter
any new comparison, figure, case selection, hypothesis classification or ranking.

## H1: what sequential pruning adds over static alternatives

### Greedy minus Static Magnitude (n=3 within each row)

| Dataset/model/target | Δ accuracy pp, mean±SD | Δ macro-F1 pp, mean±SD | Accuracy W/T/L | Δ compression pp | Search ratio median | Equal masks |
|---|---:|---:|---:|---:|---:|---:|
| Cora/GCN/30% | +2.773±1.124 | +6.388±5.054 | 3/0/0 | +2.056 | 1.85× | 0/3 |
| Cora/GCN/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 1.59× | 3/3 |
| Cora/GraphSAGE/30% | +5.977±5.939 | +11.683±12.984 | 3/0/0 | +2.074 | 2.74× | 0/3 |
| Cora/GraphSAGE/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 2.17× | 3/3 |
| CiteSeer/GCN/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 3.64× | 3/3 |
| CiteSeer/GCN/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 2.19× | 3/3 |
| CiteSeer/GraphSAGE/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.48× | 3/3 |
| CiteSeer/GraphSAGE/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 3.52× | 3/3 |
| PubMed/GCN/30% | +2.190±3.992 | +2.478±4.060 | 1/1/1 | +2.002 | 4.67× | 0/3 |
| PubMed/GCN/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 4.57× | 3/3 |
| PubMed/GraphSAGE/30% | -0.744±1.904 | -0.836±2.047 | 2/0/1 | +2.023 | 7.01× | 0/3 |
| PubMed/GraphSAGE/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 6.83× | 3/3 |

At 50% all 18 Greedy traces are exactly `L0−8,L0−8`, final 16/32. Static Magnitude
chooses the same widths. The incident absolute score of a remaining layer 0 channel
uses its unchanged input-weight row, bias and next-layer column. If **only layer 0**
is pruned, no retained coordinate in that score changes. Its ordering, including
stable ID ties, therefore stays the dense ordering; two successive removals select
the same top 16 IDs as static ranking. This is a conditional structural explanation,
not a theorem that static always wins or that either width policy is optimal.
The static width allocation is observed, not reoptimized in this analysis.

At 30%, masks differ in 12/18 pairs: all Cora and PubMed conditions. Magnitude chooses
23/16 on Cora and 23/25 on PubMed, while Greedy chooses 22/32. Greedy achieves about
2.00–2.07pp greater compression, with different layer allocation and retained masks.
All 6 CiteSeer 30% pairs match 22/32; static nearest fallback coincides with Greedy's
threshold outcome. Thus higher/lower quality at 30% is associated with controller,
allocation and mask changes, not an identified benefit of sequential decisions alone.
Cora's positive 3/3 directions are an observed consistent small-sample pattern;
PubMed/GCN has one gain, one tie, one loss, and PubMed/GraphSAGE has a negative mean
despite 2/3 positive accuracy differences. The individual pairs matter.

### Greedy minus Static SNIP-PT

| Dataset/model/target | Δ accuracy pp, mean±SD | Δ macro-F1 pp, mean±SD | Accuracy W/T/L | Δ compression pp | Search ratio median | Equal masks |
|---|---:|---:|---:|---:|---:|---:|
| Cora/GCN/30% | +7.640±2.616 | +11.514±7.578 | 3/0/0 | +2.056 | 1.62× | 0/3 |
| Cora/GCN/50% | +3.697±3.636 | +4.397±3.327 | 2/0/1 | -0.119 | 1.38× | 0/3 |
| Cora/GraphSAGE/30% | +18.115±5.694 | +25.594±14.788 | 3/0/0 | +2.074 | 2.29× | 0/3 |
| Cora/GraphSAGE/50% | +0.863±1.776 | -1.789±2.614 | 2/0/1 | -0.033 | 1.86× | 0/3 |
| CiteSeer/GCN/30% | -0.401±1.997 | -2.135±2.606 | 2/0/1 | +0.000 | 2.80× | 0/3 |
| CiteSeer/GCN/50% | +3.910±5.779 | -2.470±5.093 | 2/0/1 | -0.038 | 1.85× | 0/3 |
| CiteSeer/GraphSAGE/30% | +0.000±1.053 | -1.736±1.135 | 2/0/1 | +0.000 | 4.87× | 0/3 |
| CiteSeer/GraphSAGE/50% | +0.150±1.773 | -3.584±3.088 | 2/0/1 | -0.006 | 2.83× | 0/3 |
| PubMed/GCN/30% | +0.862±0.748 | +1.693±1.862 | 2/1/0 | +2.002 | 3.09× | 0/3 |
| PubMed/GCN/50% | -2.731±1.817 | -2.972±1.735 | 0/0/3 | -0.349 | 2.99× | 0/3 |
| PubMed/GraphSAGE/30% | -0.600±1.182 | -0.456±1.168 | 1/0/2 | +2.023 | 4.29× | 0/3 |
| PubMed/GraphSAGE/50% | +0.440±2.145 | +1.166±2.973 | 2/0/1 | -0.114 | 4.34× | 0/3 |

SNIP is a post-training train-label saliency adaptation, not initialization SNIP.
Greedy uses validation reward, so saliency criterion, feedback, trajectory and
sometimes widths/compression all differ. PubMed/GCN 50% is a concrete counterexample:
SNIP accuracy exceeds Greedy in all 3 seeds, mean 2.73pp; its mean macro-F1 advantage
is 2.97pp. On CiteSeer 50%, higher Greedy mean accuracy can coexist with lower macro-F1.
No classwise causal interpretation is made without saved confusion/score evidence.

### Greedy minus Sequential Random

| Dataset/model/target | Δ accuracy pp, mean±SD | Δ macro-F1 pp, mean±SD | Accuracy W/T/L | Δ compression pp | Search ratio median | Equal masks |
|---|---:|---:|---:|---:|---:|---:|
| Cora/GCN/30% | +5.669±2.824 | +7.758±4.482 | 3/0/0 | +1.925 | 1.95× | 0/3 |
| Cora/GCN/50% | +7.209±10.034 | +13.277±14.139 | 2/1/0 | -0.814 | 1.70× | 0/3 |
| Cora/GraphSAGE/30% | +6.531±5.668 | +13.024±13.104 | 3/0/0 | +1.944 | 1.99× | 0/3 |
| Cora/GraphSAGE/50% | +5.176±6.306 | +10.510±13.299 | 2/1/0 | -0.798 | 1.75× | 0/3 |
| CiteSeer/GCN/30% | -0.551±1.608 | -1.228±2.464 | 2/0/1 | +31.190 | 12.05× | 0/3 |
| CiteSeer/GCN/50% | +9.424±11.547 | +8.242±9.440 | 3/0/0 | -0.326 | 1.79× | 0/3 |
| CiteSeer/GraphSAGE/30% | -0.351±0.828 | -1.619±0.824 | 1/0/2 | +31.195 | 14.41× | 0/3 |
| CiteSeer/GraphSAGE/50% | +9.975±15.853 | +13.260±19.721 | 3/0/0 | -0.319 | 1.82× | 0/3 |
| PubMed/GCN/30% | +9.561±7.235 | +10.060±6.808 | 3/0/0 | +1.623 | 2.66× | 0/3 |
| PubMed/GCN/50% | +19.131±11.087 | +22.317±14.250 | 3/0/0 | -0.239 | 1.39× | 0/3 |
| PubMed/GraphSAGE/30% | +23.671±38.456 | +26.380±42.760 | 3/0/0 | +1.668 | 2.70× | 0/3 |
| PubMed/GraphSAGE/50% | +32.649±32.133 | +37.389±35.872 | 3/0/0 | -0.186 | 1.23× | 0/3 |

These are package comparisons. Sequential Random enforces exact reachability;
Greedy permits threshold overshoot. At feasible 50%, Greedy has nonnegative
accuracy differences across all 6 dataset/model strata (positive in 16 seeds, ties 2),
while actual compression still differs. CiteSeer 30% Random is an explicitly dense,
infeasible outcome: it cannot support a claim about preserving quality at 30%.

Static Random samples width pairs and channel subsets; Sequential Random samples
reachability-legal actions but uses magnitude channels. Their paired accuracy
means range from−23.11 to+26.62pp across strata and signs often vary within 3 seeds.
These do not isolate sequential decision-making. The complete contrasts and
macro-F1/compression differences are preserved in the two paired CSVs.

## H2/H3: depth-2 planning versus its Greedy reference

| Dataset/model/target | Δ accuracy pp, mean±SD | Δ macro-F1 pp, mean±SD | Accuracy W/T/L | Δ compression pp | Search ratio median | Equal masks |
|---|---:|---:|---:|---:|---:|---:|
| Cora/GCN/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.05× | 3/3 |
| Cora/GCN/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.67× | 3/3 |
| Cora/GraphSAGE/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 6.94× | 3/3 |
| Cora/GraphSAGE/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 8.31× | 3/3 |
| CiteSeer/GCN/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 6.92× | 3/3 |
| CiteSeer/GCN/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.84× | 3/3 |
| CiteSeer/GraphSAGE/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.03× | 3/3 |
| CiteSeer/GraphSAGE/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.82× | 3/3 |
| PubMed/GCN/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.44× | 3/3 |
| PubMed/GCN/50% | +1.429±2.475 | +1.385±2.399 | 1/2/0 | +0.310 | 8.06× | 2/3 |
| PubMed/GraphSAGE/30% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 6.86× | 3/3 |
| PubMed/GraphSAGE/50% | +0.000±0.000 | +0.000±0.000 | 0/3/0 | +0.000 | 7.71× | 3/3 |

Masks/tensors agree 35/36; action sequences agree 34/36. Besides the quality-changing
PubMed/GCN 202/50% case, PubMed/GCN 303/30% reverses `L0−8,L0−2` to `L0−2,L0−8` but
ends at the same 22/32 mask and quality. Action differences alone do not establish
quality gains. The per-run trajectories are in
[GREEDY_LOOKAHEAD_TRAJECTORIES.csv](GREEDY_LOOKAHEAD_TRAJECTORIES.csv).

A new **cross-method evidence bridge** corroborates all 72 Greedy immediate-reward
choices using Lookahead's already recorded previews at the identical concrete
parent: same graph/checkpoint/split/target, original masks, remaining horizon and
parent validation accuracy. Some evidence comes from a nonselected depth-2 branch.
This extends, without altering, the earlier audit's standalone-Greedy limitation.
It does not invent new scores or rerun Greedy; the recorded selected transition
also matches exactly. [72 bridge rows](GREEDY_PREVIEW_BRIDGE.csv) identify provenance.
The equivalence relies on the verified frozen deterministic environment; aggregate
state/width equality alone is insufficient, and tests reject that unsafe reuse.

Across 36 correlated observational pairs, recorded Lookahead search ratios have
median 7.382, range
6.637–11.691;
new-candidate ratios have median 8.0, range 7.0–11.5. This is a descriptive cost
distribution, not 36 independent training replicates. Added search over all 36
conditions is 159.829s.35 identical-mask outcomes give no measured quality return
for that added search. Exact paired n=3 strata above, not the pooled average,
are the basis for scientific interpretation.

The exceptional cell's paired accuracy differences are 0,+4.286,0pp; mean
1.429±2.475pp (sample SD).
The same exceptional seed's SNIP-PT model is slightly less compressed but has
higher recorded quality at 0.230s search, versus Lookahead's7.696s. A gain over
Greedy therefore does not make Lookahead the best observed quality-cost package.
One seed does not establish a general planning advantage. An application might
value the observed+4.29pp for about 7 additional seconds in this one case, but no
utility threshold or advance rule for identifying such a case was tested.
[The case study](LOOKAHEAD_CASE_STUDY.md) explains the recorded return difference.

## Pareto and cost interpretation

[PARETO_DIAGNOSTICS.csv](PARETO_DIAGNOSTICS.csv) retains 252 rows. Within each
checkpoint/requested-target block, A dominates B only if validation quality and
actual compression are no worse, recorded search time no higher, with one strict
improvement. Separate accuracy and macro-F1 frontiers are reported. Frontiers are
also flagged within physically pruned, symmetric-band outcomes; ineligible rows
remain present, and CiteSeer 30% has no eligible method, not an invented frontier.

On the macro-F1/actual-compression/search frontier, membership counts over 36 blocks
are: Random 36,Magnitude 24,SNIP 22,Sequential Random 26,Greedy 12,Lookahead 1,Q36. Timing
is a single recorded search per condition, so exact dominance is measurement-based,
not a statement that tiny time differences are statistically reproducible.
Q's extreme compression protects it from lower-compression competitors; this does
not make it a useful 30/50% controller.35 Lookahead results are dominated by their
Greedy counterpart. No hardware-aware objective is inferred from parameter counts.

Inference mean/std were measured for pruned models with 5 warmup/20 timed passes;
individual raw samples and standardized dense latencies were not saved. They are
included descriptively in METHOD_SUMMARY, never substituted for search cost or
used to fabricate dense speedup. Reused dense fitting cost is missing, not 0, and
training cost is counted per dense checkpoint rather than 14 times per pruning row.

## H4 and article implications

Native V2 Q reaches 87.441% mean compression, never its requested
band at termination, and exhausts the horizon in 36/36 deployments. It crosses the
lower target threshold after 2–8 actions, then continues 12–18 actions; only 4 paths
visit the symmetric band at all. This is a failed control result for this fixed
configuration, not a defensible matched-quality comparison to 30/50% baselines.
[Q_LEARNING_DIAGNOSTICS.md](Q_LEARNING_DIAGNOSTICS.md) separates confirmed selection
mechanisms from untested hypotheses and keeps historical V3/V4 as separate context.

[HYPOTHESIS_EVALUATION.md](HYPOTHESIS_EVALUATION.md) gives support, counterevidence,
confounders, confidence and unresolved questions for H1–H4. The strongest manuscript
contribution is an audited empirical account of **when extra adaptive search changes
nothing, when it changes a model, and why target semantics must be checked before
quality claims**—not a claim of universally superior pruning.

Recommended main-article material: a compact controller/replication table; the 12
Greedy–Lookahead paired cells; a quality-versus-achieved-compression figure; the
35/36 equality and cost-ratio figure; and the one clearly labelled exploratory
planning case. A Q compliance/coverage figure supports the negative diagnostic
result. Keep all 84 method cells,756 individual contrasts, extra Pareto tables and
full provenance in supplementary material. Accuracy and macro-F1 should both be
reported; no test-driven winner is selected. The manuscript itself is unchanged.

## Prioritized future work (recommendations only; nothing executed)

1. **Matched-budget controls for layer allocation/planning**, scientific value high,
   cost low–moderate: reuse frozen models, compare fixed actual widths/compression
   around the 30% layout difference and the PubMed case; predeclare controls and
   analyze mask identity separately. This addresses the main causal confound.
2. **Prospective replication of the rare planning case**, value high, cost moderate:
   preregister PubMed/GCN 50% and equal-compression controls across additional
   independent training/split seeds. Do not retain only seeds where planning wins.
3. **Nested policy-seed replication for all stochastic methods/cells**, value high,
   cost moderate–high (Q dominates): separate policy/mask variance from dense-seed
   variance; the independent unit remains the dense checkpoint. Use a uniform scope.
4. **Standardized paired dense/pruned inference benchmarks**, value high for any
   acceleration claim, cost low relative to new fitting: fixed host/threads/cache
   controls, repeated measurement, randomized order and dense baselines. Search
   ratios alone do not establish inference acceleration.
5. **Separately identified controller-matched Q diagnostics**, value conditional,
   cost moderate: predeclare a feasibility/STOP/coverage question with compatible
   controls, informed by the separate historical diagnostics. Do not simply spend
   more episodes or tune alpha/lambda to this test set. Any new algorithm/controller
   requires new variant IDs and separate authorization.

No future experiment, article edit, push or merge is performed here.
