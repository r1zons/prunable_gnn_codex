"""Render scientific interpretations from frozen analytical outputs, no ML calls."""
from pathlib import Path
import argparse
import csv
import json
import re
import statistics as st

METHODS=['static_random','static_magnitude','static_snip','sequential_random','adaptive_greedy','lookahead_depth2','tabular_q']
NAMES={'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed','gcn':'GCN','graphsage':'GraphSAGE'}
CELLS=[(d,m,t) for d in ('cora','citeseer','pubmed') for m in ('gcn','graphsage') for t in (.3,.5)]


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def comparison_table(summary,ma,mb):
    selected=[r for r in summary if (r['method_a'],r['method_b'])==(ma,mb)]
    result=['| Dataset/model/target | Δ accuracy pp, mean±SD | Δ macro-F1 pp, mean±SD | Accuracy W/T/L | Δ compression pp | Search ratio median | Equal masks |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for key in CELLS:
        r=next(r for r in selected if (r['dataset'],r['architecture'],float(r['requested_compression']))==key)
        result.append(f"| {NAMES[key[0]]}/{NAMES[key[1]]}/{int(key[2]*100)}% | {float(r['delta_accuracy_pp_mean']):+.3f}±{float(r['delta_accuracy_pp_sd']):.3f} | {float(r['delta_macro_f1_pp_mean']):+.3f}±{float(r['delta_macro_f1_pp_sd']):.3f} | {r['accuracy_wins_b']}/{r['accuracy_ties']}/{r['accuracy_losses_b']} | {float(r['delta_compression_pp_mean']):+.3f} | {float(r['search_ratio_b_over_a_median']):.2f}× | {r['same_mask_pairs']}/3 |")
    return '\n'.join(result)


def render(collection: Path,analysis: Path):
    result=json.loads((analysis/'ANALYSIS_RESULTS.json').read_text());pairs=rows(analysis/'PAIRED_COMPARISONS.csv');summary=rows(analysis/'PAIRED_SUMMARY.csv')
    original=rows(collection/'verified_results.csv')
    snip_case=next(r for r in original if r['run_id']=='pubmed-gcn-s202-static_snip-t50')
    method=rows(analysis/'METHOD_SUMMARY.csv');qrun=rows(analysis/'Q_RUN_DIAGNOSTICS.csv');qd=rows(analysis/'Q_DEPLOYMENT_DECISIONS.csv');front=rows(analysis/'PARETO_DIAGNOSTICS.csv')
    case=json.loads((analysis/'LOOKAHEAD_CASE_EVIDENCE.json').read_text());cp=case['comparison'];q=result['q'];lg=result['greedy_lookahead']
    assert result['validation']['observations']==252 and result['validation']['raw_files_reverified']==1226
    assert lg['same_masks']==35 and lg['same_sequences']==34 and lg['bridges']['corroborated']==72
    assert q['stop_reasons']=={'max_steps':36} and all(p['same_mask'] and p['same_action_sequence'] and not p['same_q_table'] for p in q['target_pairs'])
    g50=[p for p in pairs if p['method_a']=='static_magnitude' and p['method_b']=='adaptive_greedy' and float(p['requested_compression'])==.5]
    assert len(g50)==18 and all(p['same_retained_mask']=='True' for p in g50)
    mg50=st.median(float(p['search_ratio_b_over_a']) for p in g50)
    case_a=case['best_paths_by_first_action']['prune:0:8'];case_b=case['best_paths_by_first_action']['prune:1:8']
    pairedcase=next(r for r in summary if r['dataset']=='pubmed' and r['architecture']=='gcn' and r['requested_compression']=='0.5' and r['method_a']=='adaptive_greedy' and r['method_b']=='lookahead_depth2')
    qalias=sum(int(r['training_aliased_states']) for r in qrun);qstates=sum(int(r['training_aggregate_states']) for r in qrun)
    varying=sum(int(r['repeated_state_action_different_rewards']) for r in qrun);maskunseen=sum(int(r['deployment_training_mask_unseen']) for r in qrun)
    extra=sum(float(p['search_b_seconds'])-float(p['search_a_seconds']) for p in pairs if p['method_a']=='adaptive_greedy' and p['method_b']=='lookahead_depth2')
    core='''This is an offline analysis of the audited252 native POLUS outcomes at executable
source55d0fc242b543c5725276144bd8e970eedc48ecd, audit commit8f24513da09c0291a562fbb31a688da2698671b9.
All160 WARNING rows and six infeasible outcomes remain present. No training,
pruning, inference, new timing measurement, server access/job, algorithm change,
manuscript edit or test-based method selection is part of this analysis.'''
    text=f'''# Scientific analysis: quality, physical compression and adaptive search cost

{core}

## Principal findings, ranked by scientific importance

1. **More search usually did not change the final model.** Lookahead and Greedy
   have identical original retained masks and tensors in35/36 pairs, although
   only34/36 action sequences match. At identical outputs, Lookahead costs
   6.64–8.54× Greedy search (median7.38×). This is strong negative evidence for a
   routine depth-2 quality benefit in this campaign, not a universal theorem.
2. **One rare planning gain is real and budget-confounded.** PubMed/GCN/seed202/50%
   gains{cp['delta_accuracy_pp']:.3f}pp validation accuracy and{cp['delta_macro_f1_pp']:.3f}pp macro-F1,
   with{cp['delta_compression_pp']:.3f}pp additional parameter compression. Search grows
   {cp['search_ratio_b_over_a']:.2f}× ({cp['search_a_seconds']:.3f}→{cp['search_b_seconds']:.3f}s),
   candidates16→184. Within that dataset/model/target, only1/3 dense seeds improves.
   It is not a matched-compression causal estimate or evidence of a general advantage.
3. **A simple static rule duplicates Greedy at50%.** Static Magnitude and Greedy
   match all18/18 masks at50%, with zero quality differences; Greedy's median
   paired search cost is{mg50:.2f}×. Every Greedy path deletes8+8 first-layer channels,
   leaving widths16/32. Conditional magnitude-rank invariance explains the equality.
4. **Native V2 Q does not control requested compression here.** All36 deployments
   exhaust20 actions, average{100*q['compression']['mean']:.3f}% compression (range
   {100*q['compression']['min']:.3f}–{100*q['compression']['max']:.3f}%), and all18 checkpoint pairs
   produce identical masks/actions for30% versus50%. Q tables differ, so the
   numerical learning updates do respond to target rewards; the final policy does not.
5. **Sparse/default action values are a confirmed decision mechanism.**466/720 Q
   choices select unlearned action values (Q=0);328 states have no learned row.
   STOP ties for maximum in452/684 states where legal, yet never executes in
   deployment. Source tie order puts pruning before STOP. Delayed credit and
   state aliasing are plausible contributors, not isolated causal explanations.
6. **Sequential adaptation has condition-specific benefits and counterexamples.**
   Greedy improves over Magnitude at Cora30% (3/3 seeds for both models), but
   compression/layer allocation differ; CiteSeer30% is identical, and PubMed
   directions vary. SNIP-PT beats Greedy accuracy for all3 PubMed/GCN50% seeds.
   Sequential Random is not uniformly better than Static Random and is not a
   clean sequential-only ablation: actions are random but channels are magnitude-based.
7. **Quality metrics and feasible compression matter.**96/252 outcomes are outside
   the symmetric target band; six Sequential Random CiteSeer30% results do no
   physical pruning. Greedy versus SNIP at CiteSeer/GCN50% has mean+3.91pp accuracy
   but−2.47pp macro-F1. No single scalar/nominal-target ranking is adequate.
8. **Pareto membership alone is misleading without target feasibility.** All36 Q
   outcomes are on the recorded3-objective frontier because their compression is
   extreme, and Static Random is often protected by low cost.35 Lookahead outcomes
   are dominated by their identical-mask, cheaper Greedy pair. These are recorded
   operating points, not significance tests or universal efficiency claims.

## Design, validity and definitions

The population is3 datasets ×2 architectures ×3 independently trained dense
seeds ×2 requested targets ×7 methods =252. Models have3 layers, hidden32,
dropout.5, no pruning-time fine-tuning. Six native seed101 dense models are reused
from their verified origins; seeds202/303 supply12 further fits. Comparisons pair
identical checkpoint, graph and saved split. The unit is **n=3 dense training seeds
per dataset/model/target**, not36 independent Lookahead pairs, repeated targets,
1,080 Q episodes,720 Q decisions or additional policy-seed replicates. Policy
seed is coupled to dense seed for stochastic methods; their variances cannot be
separated by this design. There is one graph per dataset, and training seeds also
change the generated60/20/20 split, so seed variation combines fit and split variation.

Requested compression is a design label; achieved compression is `1−P_after/P_dense`.
Physical pruning requires a smaller architecture/parameter count. Within-band
means absolute deviation≤.01; threshold success and infeasibility are separate
outcomes. All missing costs remain missing: static scoring split timings and Q
candidate-enumeration counts are not replaced by0; all18 dense inference baselines
are absent. No dense-to-pruned latency speedup is claimed.

The same validation data supports dense early stopping and adaptive pruning.
Adaptive methods search this metric, while static saliency methods do not receive
the same validation oracle. A validation advantage is therefore not an unbiased
held-out generalization estimate. Existing test outcomes remain available under
the fixed protocol, but are not used for hypothesis or case selection here.

Paired differences are **B−A** in percentage points; cost/evaluation ratios are
B/A. [756 individual contrasts](PAIRED_COMPARISONS.csv) cover all21 method pairs
within36 checkpoint/target blocks. [252 paired summary cells](PAIRED_SUMMARY.csv)
each contain exactly3 paired seeds, with mean difference, sample SD, win/tie/loss
counts, compression differences, mask/width equality and cost-ratio distributions.
A numerical tie tolerance of1e-10pp suppresses only floating-point noise, far below
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

{comparison_table(summary,'static_magnitude','adaptive_greedy')}

At50% all18 Greedy traces are exactly `L0−8,L0−8`, final16/32. Static Magnitude
chooses the same widths. The incident absolute score of a remaining layer0 channel
uses its unchanged input-weight row, bias and next-layer column. If **only layer0**
is pruned, no retained coordinate in that score changes. Its ordering, including
stable ID ties, therefore stays the dense ordering; two successive removals select
the same top16 IDs as static ranking. This is a conditional structural explanation,
not a theorem that static always wins or that either width policy is optimal.
The static width allocation is observed, not reoptimized in this analysis.

At30%, masks differ in12/18 pairs: all Cora and PubMed conditions. Magnitude chooses
23/16 on Cora and23/25 on PubMed, while Greedy chooses22/32. Greedy achieves about
2.00–2.07pp greater compression, with different layer allocation and retained masks.
All6 CiteSeer30% pairs match22/32; static nearest fallback coincides with Greedy's
threshold outcome. Thus higher/lower quality at30% is associated with controller,
allocation and mask changes, not an identified benefit of sequential decisions alone.
Cora's positive3/3 directions are an observed consistent small-sample pattern;
PubMed/GCN has one gain, one tie, one loss, and PubMed/GraphSAGE has a negative mean
despite2/3 positive accuracy differences. The individual pairs matter.

### Greedy minus Static SNIP-PT

{comparison_table(summary,'static_snip','adaptive_greedy')}

SNIP is a post-training train-label saliency adaptation, not initialization SNIP.
Greedy uses validation reward, so saliency criterion, feedback, trajectory and
sometimes widths/compression all differ. PubMed/GCN50% is a concrete counterexample:
SNIP accuracy exceeds Greedy in all3 seeds, mean2.73pp; its mean macro-F1 advantage
is2.97pp. On CiteSeer50%, higher Greedy mean accuracy can coexist with lower macro-F1.
No classwise causal interpretation is made without saved confusion/score evidence.

### Greedy minus Sequential Random

{comparison_table(summary,'sequential_random','adaptive_greedy')}

These are package comparisons. Sequential Random enforces exact reachability;
Greedy permits threshold overshoot. At feasible50%, Greedy has nonnegative
accuracy differences across all6 dataset/model strata (positive in16 seeds, ties2),
while actual compression still differs. CiteSeer30% Random is an explicitly dense,
infeasible outcome: it cannot support a claim about preserving quality at30%.

Static Random samples width pairs and channel subsets; Sequential Random samples
reachability-legal actions but uses magnitude channels. Their paired accuracy
means range from−23.11 to+26.62pp across strata and signs often vary within3 seeds.
These do not isolate sequential decision-making. The complete contrasts and
macro-F1/compression differences are preserved in the two paired CSVs.

## H2/H3: depth-2 planning versus its Greedy reference

{comparison_table(summary,'adaptive_greedy','lookahead_depth2')}

Masks/tensors agree35/36; action sequences agree34/36. Besides the quality-changing
PubMed/GCN202/50% case, PubMed/GCN303/30% reverses `L0−8,L0−2` to `L0−2,L0−8` but
ends at the same22/32 mask and quality. Action differences alone do not establish
quality gains. The per-run trajectories are in
[GREEDY_LOOKAHEAD_TRAJECTORIES.csv](GREEDY_LOOKAHEAD_TRAJECTORIES.csv).

A new **cross-method evidence bridge** corroborates all72 Greedy immediate-reward
choices using Lookahead's already recorded previews at the identical concrete
parent: same graph/checkpoint/split/target, original masks, remaining horizon and
parent validation accuracy. Some evidence comes from a nonselected depth-2 branch.
This extends, without altering, the earlier audit's standalone-Greedy limitation.
It does not invent new scores or rerun Greedy; the recorded selected transition
also matches exactly. [72 bridge rows](GREEDY_PREVIEW_BRIDGE.csv) identify provenance.
The equivalence relies on the verified frozen deterministic environment; aggregate
state/width equality alone is insufficient, and tests reject that unsafe reuse.

Across36 correlated observational pairs, recorded Lookahead search ratios have
median{lg['search_ratio_descriptive_over_36']['median']:.3f}, range
{lg['search_ratio_descriptive_over_36']['min']:.3f}–{lg['search_ratio_descriptive_over_36']['max']:.3f};
new-candidate ratios have median8.0, range7.0–11.5. This is a descriptive cost
distribution, not36 independent training replicates. Added search over all36
conditions is{extra:.3f}s.35 identical-mask outcomes give no measured quality return
for that added search. Exact paired n=3 strata above, not the pooled average,
are the basis for scientific interpretation.

The exceptional cell's paired accuracy differences are0,+4.286,0pp; mean
{float(pairedcase['delta_accuracy_pp_mean']):.3f}±{float(pairedcase['delta_accuracy_pp_sd']):.3f}pp (sample SD).
The same exceptional seed's SNIP-PT model is slightly less compressed but has
higher recorded quality at0.230s search, versus Lookahead's7.696s. A gain over
Greedy therefore does not make Lookahead the best observed quality-cost package.
One seed does not establish a general planning advantage. An application might
value the observed+4.29pp for about7 additional seconds in this one case, but no
utility threshold or advance rule for identifying such a case was tested.
[The case study](LOOKAHEAD_CASE_STUDY.md) explains the recorded return difference.

## Pareto and cost interpretation

[PARETO_DIAGNOSTICS.csv](PARETO_DIAGNOSTICS.csv) retains252 rows. Within each
checkpoint/requested-target block, A dominates B only if validation quality and
actual compression are no worse, recorded search time no higher, with one strict
improvement. Separate accuracy and macro-F1 frontiers are reported. Frontiers are
also flagged within physically pruned, symmetric-band outcomes; ineligible rows
remain present, and CiteSeer30% has no eligible method, not an invented frontier.

On the macro-F1/actual-compression/search frontier, membership counts over36 blocks
are: Random36,Magnitude24,SNIP22,Sequential Random26,Greedy12,Lookahead1,Q36. Timing
is a single recorded search per condition, so exact dominance is measurement-based,
not a statement that tiny time differences are statistically reproducible.
Q's extreme compression protects it from lower-compression competitors; this does
not make it a useful30/50% controller.35 Lookahead results are dominated by their
Greedy counterpart. No hardware-aware objective is inferred from parameter counts.

Inference mean/std were measured for pruned models with5 warmup/20 timed passes;
individual raw samples and standardized dense latencies were not saved. They are
included descriptively in METHOD_SUMMARY, never substituted for search cost or
used to fabricate dense speedup. Reused dense fitting cost is missing, not0, and
training cost is counted per dense checkpoint rather than14 times per pruning row.

## H4 and article implications

Native V2 Q reaches{100*q['compression']['mean']:.3f}% mean compression, never its requested
band at termination, and exhausts the horizon in36/36 deployments. It crosses the
lower target threshold after2–8 actions, then continues12–18 actions; only4 paths
visit the symmetric band at all. This is a failed control result for this fixed
configuration, not a defensible matched-quality comparison to30/50% baselines.
[Q_LEARNING_DIAGNOSTICS.md](Q_LEARNING_DIAGNOSTICS.md) separates confirmed selection
mechanisms from untested hypotheses and keeps historical V3/V4 as separate context.

[HYPOTHESIS_EVALUATION.md](HYPOTHESIS_EVALUATION.md) gives support, counterevidence,
confounders, confidence and unresolved questions for H1–H4. The strongest manuscript
contribution is an audited empirical account of **when extra adaptive search changes
nothing, when it changes a model, and why target semantics must be checked before
quality claims**—not a claim of universally superior pruning.

Recommended main-article material: a compact controller/replication table; the12
Greedy–Lookahead paired cells; a quality-versus-achieved-compression figure; the
35/36 equality and cost-ratio figure; and the one clearly labelled exploratory
planning case. A Q compliance/coverage figure supports the negative diagnostic
result. Keep all84 method cells,756 individual contrasts, extra Pareto tables and
full provenance in supplementary material. Accuracy and macro-F1 should both be
reported; no test-driven winner is selected. The manuscript itself is unchanged.

## Prioritized future work (recommendations only; nothing executed)

1. **Matched-budget controls for layer allocation/planning**, scientific value high,
   cost low–moderate: reuse frozen models, compare fixed actual widths/compression
   around the30% layout difference and the PubMed case; predeclare controls and
   analyze mask identity separately. This addresses the main causal confound.
2. **Prospective replication of the rare planning case**, value high, cost moderate:
   preregister PubMed/GCN50% and equal-compression controls across additional
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
'''
    (analysis/'SCIENTIFIC_ANALYSIS.md').write_text(text)
    case_text=f'''# Recorded Lookahead case study: PubMed / GCN / seed202 / target50%

Case selection was explicitly requested by the researcher after validation/mask
inspection. It is exploratory, not a prospectively chosen confirmation sample.
The underlying checkpoint/split is identical across methods. The original audit
already checked frozen tensors/metrics; this analysis only reads recorded traces.

## Outcomes and paired context

| Outcome | Greedy | Lookahead depth2 | L−G / L:G |
|---|---:|---:|---:|
| Validation accuracy |{100*cp['accuracy_a']:.6f}%|{100*cp['accuracy_b']:.6f}%|+{cp['delta_accuracy_pp']:.6f}pp|
| Validation macro-F1 |{100*cp['macro_f1_a']:.6f}%|{100*cp['macro_f1_b']:.6f}%|+{cp['delta_macro_f1_pp']:.6f}pp|
| Actual parameter compression |{100*cp['compression_a']:.6f}%|{100*cp['compression_b']:.6f}%|+{cp['delta_compression_pp']:.6f}pp|
| Remaining parameters |{int(cp['parameters_a'])}|{int(cp['parameters_b'])}|{int(cp['parameters_b']-cp['parameters_a'])}|
| Hidden widths |16/32|16/24|8 additional layer1 channels removed|
| Search |{cp['search_a_seconds']:.6f}s|{cp['search_b_seconds']:.6f}s|{cp['search_ratio_b_over_a']:.6f}×|
| New candidate evaluations |16|184|11.5×|
| Validation oracle calls |17|185|{cp['validation_evaluation_ratio']:.6f}×|

The two other independent seeds in this cell have zero quality difference;
accuracy paired mean±sample SD is{float(pairedcase['delta_accuracy_pp_mean']):.6f}±{float(pairedcase['delta_accuracy_pp_sd']):.6f}pp (n=3).
A single positive seed, selected for explanation, cannot establish a general effect.

The same seed's Static SNIP-PT outcome has validation accuracy
{100*float(snip_case['pruned_val_accuracy']):.6f}%,macro-F1
{100*float(snip_case['pruned_val_macro_f1']):.6f}%,compression
{100*float(snip_case['achieved_compression']):.6f}%,search
{float(snip_case['search_seconds']):.6f}s. It has higher measured quality than
Lookahead at slightly less compression, and Lookahead costs
{cp['search_b_seconds']/float(snip_case['search_seconds']):.2f}× its search. This is
not strict3-objective domination because compression differs, but it shows why a
local gain over Greedy alone does not establish superiority over all methods.

## The first divergence and exact recorded returns

Both execute `prune:0:8`, reaching24/32 at24.809449% compression and87.446105%
recorded intermediate validation accuracy. Original kept IDs, remaining horizon19,
reward inputs and frozen tensors are shared. Lookahead's **initial** best path was
`L0−8,L0−8`; it did not execute that fixed two-action plan. It replanned after
executing the first action.

At this common parent, Greedy selects `prune:0:8` and terminates16/32; Lookahead
selects `prune:1:8`, then replans and executes `prune:0:8`, ending16/24.

| Recorded path from the shared parent | r1 | r2 | r1+.9r2 |
|---|---:|---:|---:|
| L0−8 (terminal) |{case_a['rewards'][0]:.12f}|not applicable|{case_a['score']:.12f}|
| L1−8 → L0−8 |{case_b['rewards'][0]:.12f}|{case_b['rewards'][1]:.12f}|{case_b['score']:.12f}|

Lookahead's score advantage is{case_b['score']-case_a['score']:.12f}. It accepts a small
immediate reward to obtain a larger recorded next reward. Both terminal penalties
are0 because both final compressions are inside±.01. The second terminal reward
is discounted once; no extra STOP or cutoff penalty is invented. All other legal
first-action maxima are preserved in [LOOKAHEAD_CASE_EVIDENCE.json](LOOKAHEAD_CASE_EVIDENCE.json).
The Greedy immediate argmax is corroborated from those recorded root previews,
not from new model evaluations or invented Greedy counterfactuals.

## Stepwise observed consequences

Greedy:32/32 →24/32 →16/32. Lookahead:32/32 →24/32 →24/24 →16/24.
After its layer1 deletion Lookahead's recorded accuracy is87.192494%; after its
final layer0 deletion it is85.518640%, versus Greedy's81.232566% float32 trace.
Intermediate macro-F1 was not saved and is not invented. Final independently
audited float64-reported accuracies agree with these within
expected float32 rounding. The improvement therefore persists in the actual final
frozen model, not merely a predicted planning score.

The **first-layer retained16 IDs are identical** in both final models. Lookahead
additionally removes original layer1 IDs **10,11,18,23,24,26,29,31**. Thus equality
of the first-layer width/mask does not imply equality of the whole mask. The
second-layer deletion and its incident output/input coordinates are the observed
final-model difference; no retraining occurs. It would be unsupported to claim
that an unseen alternative mask or equal-compression Greedy variant would have
lower quality: those counterfactuals were not measured.

The fixed repeated-drop reward includes.6Δcompression−.4max(0,A_dense−A_current).
Its truncated return depends on intermediate quality, path length, termination
and extra compression, not just final accuracy. This example supports a local
recorded planning advantage under that objective. It does not separate layer
allocation from actual budget or prove full-horizon optimality/generalization.

[Paired trace/score figure](FIGURES/08_lookahead_recorded_case.pdf).
All36 comparisons remain in [the trajectory table](GREEDY_LOOKAHEAD_TRAJECTORIES.csv).
A second changed trajectory, PubMed/GCN303/30%, only reverses two layer0 deletion
sizes and preserves the final mask and metrics. It is a useful negative control
against equating a changed action sequence with a quality improvement.
'''
    (analysis/'LOOKAHEAD_CASE_STUDY.md').write_text(case_text)
    qtext=f'''# Native V2 tabular Q-learning: observed behavior and mechanisms

This is the fixed30-episode,alpha.1,gamma.9,epsilon.2,lambda2 native V2 policy
used in this campaign, not constrained V3/V4 Q. "Deployment" below means the final
pruning-policy rollout, not per-node classifier inference. No training, Q update, new rollout,
retuning or model inference is performed in this analysis. The earlier independent
audit reconstructed Bellman updates; this analysis reads its verified final tables
and recorded observations.

## Confirmed current-campaign observations

-36/36 deployments terminate `max_steps`, each after20 successful removals; STOP
  is never selected during deployment. First hidden width is4 in every run.
-Mean achieved compression over36 recorded target-labelled outcomes:
  **{100*q['compression']['mean']:.6f}%**, range{100*q['compression']['min']:.6f}–{100*q['compression']['max']:.6f}%.
  Each of18 dense checkpoints has identical final masks, action sequence and final
  quality for30% and50%. These repeated targets are not independent replicas.
-Their Q tables differ in18/18 pairs. Targets affect terminal rewards and learned
  numbers, but do not change these recorded greedy deployments. The terminal
  penalty for50% is.4 higher (less negative) than30% at the shared final model.
-Each deployment crosses target−.01 after2–8 actions and continues12–18 more.
  Only4/36 paths visit the symmetric band at an intermediate step (all target50%);
  none ends there. Threshold crossing is not equivalent to a band hit.
-Training did use STOP:404/1,080 episodes stop,674 exhaust the horizon and2 exhaust
  legal removals. STOP is implemented/available; training episodes are process
  observations, not additional trained GNNs or statistical n.

[36 run diagnostics](Q_RUN_DIAGNOSTICS.csv),[720 deployment decisions](Q_DEPLOYMENT_DECISIONS.csv),
[1,080 episode summaries](Q_EPISODE_SUMMARY.csv),[18 target-pair checks](Q_TARGET_PAIR_CHECK.csv).

## Confirmed source/decision mechanisms

Reward is `r_t=.6(s_t−s_prev)−.4 max(0,A_dense−A_t)` plus the once-only terminal
penalty `−2 max(0,|s_t−target|−.01)`. It repeats the dense-to-current drop, rather
than charging only incremental loss. The goal penalty is delayed until STOP,
max_steps or no actions; **crossing the target is not a terminal condition**.
The menu is first layer then second layer, sizes1/2/4/8 ascending; STOP is last and
legal only after a removal. Magnitude determines channels, policy chooses layer/size.

The state is `(local_v2,(width0,width1),remaining_horizon,floor(accuracy/.01),terminal)`.
Original retained IDs are omitted. Target is fixed per separate policy rather
than encoded as an explicit distance feature. Its absence from the tuple alone
is not proof that a fixed-target MDP cannot learn; the separate tables demonstrably
differ. No test label enters the validation-only reward.

Deployment picks the **first legal maximum** of the sparse learned row, with
unseen actions valued0. This exactly matches all720 recorded decisions:

| Diagnostic | Count / denominator | Interpretation |
|---|---:|---|
| Selected action has no learned entry |466/720 (64.72%)|default0 is often a decision value|
| Entire state has no learned row |328/720 (45.56%)|legal enumeration determines a zero tie|
| STOP legal |684/720|all except36 initial states|
| STOP tied for maximum |452/684 (66.08%)|pruning appears first and wins the tie|
| STOP has learned entry |78/684|most legal STOP values remain default0|
| Unseen action beats strictly negative known actions |138/720|confirmed sparse-value optimism effect|
| Selected learned positive value |248/720|early behavior is not entirely random/unseen|

The analytic immediate STOP reward is also recorded from its known formula;
it needs no new model score and is explicitly labelled analytic, not a new
counterfactual forward. Unrecorded pruning alternatives' validation quality is
not estimated from Q values. Q is an approximate return estimate, not accuracy.

First-layer cuts have larger parameter-gain contributions because F≫hidden32 and
classes. Once the first layer reaches minimum4, much of the remaining legal menu
is in layer1. Missing/negative values and ordered zero ties directly explain the
corresponding **observed** action choices, but do not establish that any proposed
alternative would improve final validation quality.

## Coverage, aliasing and credit: association versus hypothesis

Mean recorded learned state-action coverage is14.479% (one minus table sparsity),
computed per run; this is not a sample-size claim. In training there are
**{qalias} aliased aggregate states among{qstates} run-scoped observed state keys**,
and{varying} repeated state-action pairs have reward variation beyond float32 tolerance.
During deployment,{maskunseen}/720 concrete masks were absent from training under
their aggregate state;328 are entirely unseen aggregate states, leaving only
{maskunseen-328} known aggregate states with a new concrete mask. Thus the present
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

The Q outcomes are valid audited physical models at roughly87.4% compression and
measured search cost. Paired Q/Greedy search ratios have median29.13× (26.62–31.71×),
but their endpoints are not compression matched. It is meaningful to compare
**target compliance, controller behavior and recorded operating points**; it is
not meaningful to infer an intrinsic quality disadvantage of Q-learning from
comparing87.4% models to30/50% baselines. No simple baseline was measured at the
same87.4% operating points in this campaign. Q's36/36 3-objective Pareto memberships
are protected by extreme compression and do not imply useful requested-target control.

## Historical context, kept separate

The [V2 pilot](../../local_structured_v2_q_random_pilot_20261027/README.md) on Cora
and fixed training seed101 also observed overcompression and target-insensitive
endpoints under30/100 episodes and targeted lambda diagnostics. Its platforms,
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
'''
    (analysis/'Q_LEARNING_DIAGNOSTICS.md').write_text(qtext)
    h='''# Hypotheses, evidence strength and unresolved questions

This evaluation uses only the current audited252 outcomes. The three independently
trained dense seeds are paired within each dataset/model/target; repeated targets,
policy choices and episodes are correlated process data. No p-values, significance
claims, confidence intervals or test-based hypothesis selection are used.

| Hypothesis | Supporting observations | Contradictions / limits | Assessment and confidence |
|---|---|---|---|
| **H1 Sequential pruning improves quality over static pruning** |Greedy exceeds Magnitude accuracy in3/3 Cora30% seeds for each architecture; Greedy often improves over random sequential allocation at feasible50% |Magnitude and Greedy have identical18/18 masks at50%, plus6/6 CiteSeer30%; SNIP beats all3 PubMed/GCN50% Greedy accuracies; Random sequential is inconsistent;30% widths/compression/controllers differ |No general support. Moderate descriptive evidence of selected package benefits; low confidence in causal sequential superiority |
| **H2 Depth2 improves quality over Greedy** |PubMed/GCN202/50% improves4.286pp accuracy/4.155pp F1 under the recorded planning objective |35/36 masks/tensors identical; only1/3 seeds improves in exceptional cell; additional.931pp compression confounds a matched-budget claim |Rare observed benefit, no general advantage established. High confidence in recorded equality/counts, low in generalizable benefit frequency |
| **H3 Additional planning cost is justified** |The exceptional gain costs7.038 additional seconds (11.69×) and may be worthwhile for an application that values it |All35 equal-mask pairs pay6.64–8.54× with zero quality improvement; no utility threshold, general latency baseline or advance case-selection rule tested |Not justified as a routine default by this evidence. Conditional application-specific value is plausible, not demonstrated generally |
| **H4 Tabular Q provides useful adaptive compression control** |Training uses STOP; Q entries respond to target rewards; models reach high real compression |All36 end max_steps20, achieve87.24–87.71%, all18 target pairs share masks/actions;466/720 choices use unseen values |Not supported for this fixed native V2 setup. High confidence in control failure within scope, no claim about all RL/Q methods |

## What is observed, explained and unproven?

**Observed:** fixed masks, metrics, actions, native costs,36 Q horizons, learned
entries/default values, STOP availability, target insensitivity, one changed
Lookahead final model. These are traced to hashes, audited data and recorded trees.

**Source/algebra supported:** immediate-reward tie rules; target crossing does not
stop native Q; once-only terminal penalty; receding-horizon one-action execution;
conditional invariance of layer0 magnitude ranks when only layer0 is removed.
The cross-method preview bridge corroborates72 Greedy choices without new scores.

**Associations/hypotheses:** architecture allocation as a reason for some30% gains;
sparse revisitation, delayed credit and aliasing as contributors to learned-control
failure; possible benefit in future training seeds. No equal-budget causal contrast,
learning ablation or prospective replication was performed in this task.

## Statistical and practical boundaries

- n=3 supports paired descriptive means/sample SD and seed directions, not robust
  distributional estimates, reliable rare-event probabilities or superiority tests.
- Targets share a dense model, and policy seeds are coupled to training seeds.
  Seeds also change splits; variance is not pure optimizer variance. Random mask
  variation cannot be separated from dense training variation with these repetitions.
- One fixed graph per dataset and one architecture setting per model limit transfer
  to other graph regimes, widths, hardware and large datasets.
- Validation reward optimizes validation accuracy, not macro-F1 or test performance.
  Validation exploration can itself bias selected scores; no claim of generalization
  or a test-selected winner is made. Test outcomes remain descriptively archived.
- 96 outside-band and6 infeasible outcomes remain in every full population table.
  Pareto eligibility is explicit; no post hoc failure exclusion improves a method.
- Time ratios are recorded native search costs, not inference acceleration. Dense
  latency baselines and raw pass samples are missing. Identical-mask latency jitter
  is not a strategy improvement; equal parameter count is not equal latency.

## Article contributions and recommended material

1. A reproducible physical-pruning comparison with original channel identities,
   explicit controller semantics, paired dense seeds and cost accounting.
2. Strong, bounded negative evidence:35/36 Lookahead outputs duplicate Greedy;
   at50%, Magnitude duplicates Greedy18/18 at lower measured search cost.
3. A transparent rare planning case explaining a two-step return and replanning,
   with its unequal-compression/post-inspection caveats next to the gain.
4. An audited control failure/coverage diagnostic for native V2 Q, separating
   correct software execution from scientific qualification of a learned policy.

Main article: controller/replication table; paired Greedy/Lookahead quality and cost
figures; quality versus achieved compression; the explicitly exploratory planning
case; Q target/STOP/coverage figure. Supplement: all84 method cells,756 individual
paired contrasts,252 paired summaries and mask/Pareto/trajectory diagnostics.
This is an empirical/methodological contribution, not a claim of universal pruning
superiority. No manuscript wording or file has been edited by this task.

Future work is ranked in SCIENTIFIC_ANALYSIS.md by concrete unresolved question,
scientific value and cost. None is launched or implicitly authorized here.
'''
    (analysis/'HYPOTHESIS_EVALUATION.md').write_text(h)
    # Presentation only: leave code spans and link destinations byte-for-byte intact.
    for name in ('SCIENTIFIC_ANALYSIS.md','LOOKAHEAD_CASE_STUDY.md','Q_LEARNING_DIAGNOSTICS.md','HYPOTHESIS_EVALUATION.md'):
        path=analysis/name
        chunks=re.split(r'(`[^`]*`|\[[^\]]*\]\([^)]+\))',path.read_text())
        for i in range(0,len(chunks),2):
            chunks[i]=re.sub(r'(?m)^-(?=\S)','- ',chunks[i])
            chunks[i]=re.sub(r'(?<![A-Za-z0-9_])([A-Za-z]{2,})(?=[0-9])',r'\1 ',chunks[i])
        path.write_text(''.join(chunks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    a=p.parse_args();render(a.collection,a.analysis)
