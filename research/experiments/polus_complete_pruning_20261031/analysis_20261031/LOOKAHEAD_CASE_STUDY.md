# Recorded Lookahead case study: PubMed / GCN / seed 202 / target 50%

Case selection was explicitly requested by the researcher after validation/mask
inspection. It is exploratory, not a prospectively chosen confirmation sample.
The underlying checkpoint/split is identical across methods. The original audit
already checked frozen tensors/metrics; this analysis only reads recorded traces.

## Outcomes and paired context

| Outcome | Greedy | Lookahead depth 2 | L−G / L:G |
|---|---:|---:|---:|
| Validation accuracy |81.232564%|85.518641%|+4.286077pp|
| Validation macro-F1 |80.245948%|84.400970%|+4.155023pp|
| Actual parameter compression |49.618898%|50.549834%|+0.930936pp|
| Remaining parameters |8659|8499|-160|
| Hidden widths |16/32|16/24|8 additional layer 1 channels removed|
| Search |0.658251s|7.695858s|11.691364×|
| New candidate evaluations |16|184|11.5×|
| Validation oracle calls |17|185|10.882353×|

The two other independent seeds in this cell have zero quality difference;
accuracy paired mean±sample SD is 1.428692±2.474567pp (n=3).
A single positive seed, selected for explanation, cannot establish a general effect.

The same seed's Static SNIP-PT outcome has validation accuracy
86.025869%,macro-F1
85.204459%,compression
50.200733%,search
0.230427s. It has higher measured quality than
Lookahead at slightly less compression, and Lookahead costs
33.40× its search. This is
not strict 3-objective domination because compression differs, but it shows why a
local gain over Greedy alone does not establish superiority over all methods.

## The first divergence and exact recorded returns

Both execute `prune:0:8`, reaching 24/32 at 24.809449% compression and 87.446105%
recorded intermediate validation accuracy. Original kept IDs, remaining horizon 19,
reward inputs and frozen tensors are shared. Lookahead's **initial** best path was
`L0−8,L0−8`; it did not execute that fixed two-action plan. It replanned after
executing the first action.

At this common parent, Greedy selects `prune:0:8` and terminates 16/32; Lookahead
selects `prune:1:8`, then replans and executes `prune:0:8`, ending 16/24.

| Recorded path from the shared parent | r1 | r2 | r1+.9r2 |
|---|---:|---:|---:|
| L0−8 (terminal) |0.124002534658|not applicable|0.124002534658|
| L1−8 → L0−8 |0.006805416638|0.138912586397|0.131826744395|

Lookahead's score advantage is 0.007824209738. It accepts a small
immediate reward to obtain a larger recorded next reward. Both terminal penalties
are 0 because both final compressions are inside±.01. The second terminal reward
is discounted once; no extra STOP or cutoff penalty is invented. All other legal
first-action maxima are preserved in [LOOKAHEAD_CASE_EVIDENCE.json](LOOKAHEAD_CASE_EVIDENCE.json).
The Greedy immediate argmax is corroborated from those recorded root previews,
not from new model evaluations or invented Greedy counterfactuals.

## Stepwise observed consequences

Greedy:32/32 →24/32 →16/32. Lookahead:32/32 →24/32 →24/24 →16/24.
After its layer 1 deletion Lookahead's recorded accuracy is 87.192494%; after its
final layer 0 deletion it is 85.518640%, versus Greedy's81.232566% float 32 trace.
Intermediate macro-F1 was not saved and is not invented. Final independently
audited float 64-reported accuracies agree with these within
expected float 32 rounding. The improvement therefore persists in the actual final
frozen model, not merely a predicted planning score.

The **first-layer retained 16 IDs are identical** in both final models. Lookahead
additionally removes original layer 1 IDs **10,11,18,23,24,26,29,31**. Thus equality
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
All 36 comparisons remain in [the trajectory table](GREEDY_LOOKAHEAD_TRAJECTORIES.csv).
A second changed trajectory, PubMed/GCN 303/30%, only reverses two layer 0 deletion
sizes and preserves the final mask and metrics. It is a useful negative control
against equating a changed action sequence with a quality improvement.
