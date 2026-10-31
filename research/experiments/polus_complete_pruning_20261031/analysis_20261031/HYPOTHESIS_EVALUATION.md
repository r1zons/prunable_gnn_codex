# Hypotheses, evidence strength and unresolved questions

This evaluation uses only the current audited 252 outcomes. The three independently
trained dense seeds are paired within each dataset/model/target; repeated targets,
policy choices and episodes are correlated process data. No p-values, significance
claims, confidence intervals or test-based hypothesis selection are used.

| Hypothesis | Supporting observations | Contradictions / limits | Assessment and confidence |
|---|---|---|---|
| **H1 Sequential pruning improves quality over static pruning** |Greedy exceeds Magnitude accuracy in 3/3 Cora 30% seeds for each architecture; Greedy often improves over random sequential allocation at feasible 50% |Magnitude and Greedy have identical 18/18 masks at 50%, plus 6/6 CiteSeer 30%; SNIP beats all 3 PubMed/GCN 50% Greedy accuracies; Random sequential is inconsistent;30% widths/compression/controllers differ |No general support. Moderate descriptive evidence of selected package benefits; low confidence in causal sequential superiority |
| **H2 Depth 2 improves quality over Greedy** |PubMed/GCN 202/50% improves 4.286pp accuracy/4.155pp F1 under the recorded planning objective |35/36 masks/tensors identical; only 1/3 seeds improves in exceptional cell; additional.931pp compression confounds a matched-budget claim |Rare observed benefit, no general advantage established. High confidence in recorded equality/counts, low in generalizable benefit frequency |
| **H3 Additional planning cost is justified** |The exceptional gain costs 7.038 additional seconds (11.69×) and may be worthwhile for an application that values it |All 35 equal-mask pairs pay 6.64–8.54× with zero quality improvement; no utility threshold, general latency baseline or advance case-selection rule tested |Not justified as a routine default by this evidence. Conditional application-specific value is plausible, not demonstrated generally |
| **H4 Tabular Q provides useful adaptive compression control** |Training uses STOP; Q entries respond to target rewards; models reach high real compression |All 36 end max_steps20, achieve 87.24–87.71%, all 18 target pairs share masks/actions;466/720 choices use unseen values |Not supported for this fixed native V2 setup. High confidence in control failure within scope, no claim about all RL/Q methods |

## What is observed, explained and unproven?

**Observed:** fixed masks, metrics, actions, native costs,36 Q horizons, learned
entries/default values, STOP availability, target insensitivity, one changed
Lookahead final model. These are traced to hashes, audited data and recorded trees.

**Source/algebra supported:** immediate-reward tie rules; target crossing does not
stop native Q; once-only terminal penalty; receding-horizon one-action execution;
conditional invariance of layer 0 magnitude ranks when only layer 0 is removed.
The cross-method preview bridge corroborates 72 Greedy choices without new scores.

**Associations/hypotheses:** architecture allocation as a reason for some 30% gains;
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
- 96 outside-band and 6 infeasible outcomes remain in every full population table.
  Pareto eligibility is explicit; no post hoc failure exclusion improves a method.
- Time ratios are recorded native search costs, not inference acceleration. Dense
  latency baselines and raw pass samples are missing. Identical-mask latency jitter
  is not a strategy improvement; equal parameter count is not equal latency.

## Article contributions and recommended material

1. A reproducible physical-pruning comparison with original channel identities,
   explicit controller semantics, paired dense seeds and cost accounting.
2. Strong, bounded negative evidence:35/36 Lookahead outputs duplicate Greedy;
   at 50%, Magnitude duplicates Greedy 18/18 at lower measured search cost.
3. A transparent rare planning case explaining a two-step return and replanning,
   with its unequal-compression/post-inspection caveats next to the gain.
4. An audited control failure/coverage diagnostic for native V2 Q, separating
   correct software execution from scientific qualification of a learned policy.

Main article: controller/replication table; paired Greedy/Lookahead quality and cost
figures; quality versus achieved compression; the explicitly exploratory planning
case; Q target/STOP/coverage figure. Supplement: all 84 method cells,756 individual
paired contrasts,252 paired summaries and mask/Pareto/trajectory diagnostics.
This is an empirical/methodological contribution, not a claim of universal pruning
superiority. No manuscript wording or file has been edited by this task.

Future work is ranked in SCIENTIFIC_ANALYSIS.md by concrete unresolved question,
scientific value and cost. None is launched or implicitly authorized here.
