# Historical investigation carried forward

Read-only investigation at `00743e237dde54ad034af8cbca9a1ec317cc0ed5` recovered
11 checkpoint-bearing Flickr records, six unique weight states, all seed42 and
all five training epochs. Two further GraphSAGE2/64 directories had configurations
for150 epochs but no checkpoint/metrics. They do not establish completed fits.
Historical inputs/results remain separate from this new seed101 campaign.

| Unique Dense configuration | Train accuracy | Validation accuracy | Validation macro-F1 | Best epoch (1-based) |
|---|---:|---:|---:|---:|
| GCN2/64 |.423721|.419384|.084476|5|
| GraphSAGE2/64 |.423660|.419272|.084414|4|
| GraphSAGE2/128 |.096620|.093389|.024406|4|
| GraphSAGE3/64 |.254379|.261625|.059249|3|
| GraphSAGE3/128 |.254379|.261625|.059249|3|
| GraphSAGE4/128 |.254379|.261625|.059249|3|

All11 strict checkpoint reloads, finite tensors, recorded architecture and parameter
counts passed inspection. Different architectures did not reuse the same weight
state. Repeated same-architecture outputs are not independent seeds. Metrics were
read from saved reports, not recomputed. The one preserved complete Dense curve,
`runs/presentation/flickr_002/progress.log`, shows sharply oscillating early losses;
training lasted4.29 seconds across five epochs. No converged Flickr baseline was
recovered. Undertraining is a confirmed limitation; learning-rate sensitivity,
feature scale and oversmoothing remain unisolated hypotheses.

The recovered cache matches raw x/edge/y:89,250 nodes,500 float32 features,7 classes,
899,756 symmetric unique edge entries, no self-loops; no pre-transform/filter.
Historical generated seed42 split60/20/20 differs from official50/25/25 roles.
Historical majority validation accuracy is41.933%; near-majority accuracy and low
macro-F1 do not establish useful classification. Prediction distributions were
not saved. Contemporary graph/package/commit pins were absent.

Git milestones: `ad09c68` (Mar30) Flickr/debug sweep; `a2d5133` (Mar30) multilayer
pruned-checkpoint compatibility; `1334e6b` (Mar31) isolation/intended150epochs;
`c9fd576`/`d1496ea` (May4) longer study/readiness fixes; `882b3c9` (Sep23) validation
selection fixes. The preserved old `sweep_results.csv` lacks validation columns,
and `best_summary.json` selected a post-fine-tuning row using test accuracy. Neither
is a valid Dense architecture-selection result. Current YAML alone proves no run.

Historical roots inspected: canonical ignored `runs/`, adjacent
`${GNN_ARTIFACT_ROOT}`, Documents diploma backup and wider local diploma project.
No additional completed Flickr fits were recovered. Nothing was pooled or altered.

Current pruning comparability remains a later gate: Static V2 and the adaptive
environment explicitly require three layers. With20 actions of sizes1/2/4/8,
4/256 and5/256 also cannot reach50% compression (optimistic upper bounds46.3% and
37.0%). This Dense-only campaign does not remove those guards or modify pruning.
