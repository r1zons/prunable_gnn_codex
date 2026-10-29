"""Package compact verified analysis; keep raw traces and per-episode rows external."""
import argparse,collections,csv,hashlib,json,re,shutil,statistics
from pathlib import Path

ORDER=['q_v4_menu12','q_v4_percent','q_v4_300','q_v4_decay300','q_v4_random_ties','q_v4_incremental_drop','q_v4_gamma0','greedy_v4_menu12','greedy_v4_percent']
LABELS=['A: {1,2}','A: width %','B: 300 episodes','C: decaying epsilon','D: random ties','E: incremental drop','F: gamma0','Greedy {1,2}','Greedy width %']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def csv_write(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def mu_sd(values,percent=False):
 factor=100 if percent else 1;mu=statistics.mean(values)*factor
 return f'{mu:.2f} ± {statistics.stdev(values)*factor:.2f}' if len(values)>1 else f'{mu:.2f} (n=1)'

def package(processed,analysis,collection,out):
 out.mkdir(exist_ok=False)
 for name in ('runs.csv','runs.json','summary.csv','summary.json','baseline_runs.csv','baseline_runs.json','baseline_summary.csv',
              'comparisons.csv','comparisons.json','verification.json','provenance.json','learning_blocks.csv','snapshots.csv',
              'deployment_decisions.csv','prefix_checks.csv','prefix_checks.json','V3_platform_bridge.csv'):
  shutil.copy2(processed/name,out/name)
 shutil.copytree(analysis/'figures',out/'figures')
 runs=read(processed/'runs.json');baselines=read(processed/'baseline_runs.json');summaries=read(processed/'summary.json');comparisons=read(processed/'comparisons.json');v=read(processed/'verification.json')
 assert len(runs)==92 and len(summaries)==36 and all(r['target_hit'] for r in runs)
 prefixes=read(processed/'prefix_checks.json')
 assert len(prefixes)==16 and all(p['same_first100_training_trace'] and p['same_q_keys'] and p['max_abs_q_difference']==0 for p in prefixes)
 assert collections.Counter(r['terminal_reason'] for r in runs)=={'target_band':89,'max_steps':3}
 native=[r for r in baselines if r['platform']=='polus'];cells=[('gcn',.3),('gcn',.5),('graphsage',.3),('graphsage',.5)]
 def group(model,target,variant):return [r for r in runs if r['model']==model and r['target']==target and r['variant']==variant]
 lsf=[]
 for job,condition in [(1607752,'capacity'),(1607753,'gcn-t30'),(1607754,'gcn-t50'),(1607755,'graphsage-t30'),(1607756,'graphsage-t50')]:
  p=collection/'polus/logs/lsf'/f'v4-cpu.{job}.out';s=p.read_text();assert 'Successfully completed.' in s
  number=lambda key:float(re.search(key+r'\s*:\s*([\d.]+) sec',s)[1])
  active,turnaround=number('Run time'),number('Turnaround time')
  lsf.append(dict(job_id=job,condition=condition,active_wall_seconds=active,turnaround_seconds=turnaround,
       queue_wait_seconds=turnaround-active,cpu_seconds=number('CPU time'),source_lsf_stdout_sha256=sha(p)))
 csv_write(out/'execution_times.csv',lsf)
 greedy_rows=[]
 for variant,label in zip(ORDER[:7],LABELS[:7]):
  c=[r for r in comparisons if r['contrast']=='Q_vs_Greedy' and r['treatment']==variant]
  greedy_rows.append(dict(variant=variant,comparisons=12,accuracy_better=sum(r['accuracy_delta_pp']>1e-8 for r in c),
    accuracy_equal=sum(abs(r['accuracy_delta_pp'])<=1e-8 for r in c),accuracy_worse=sum(r['accuracy_delta_pp']<-1e-8 for r in c),
    min_search_ratio=min(r['search_ratio'] for r in c),max_search_ratio=max(r['search_ratio'] for r in c),
    reward_control_matched=variant!='q_v4_incremental_drop'))
 csv_write(out/'greedy_comparison_summary.csv',greedy_rows)
 reward_rows=[]
 for r in runs:
  if not r['variant'].startswith('q_'):continue
  reward_rows.append({k:r[k] for k in ('run_id','model','target','variant','policy_seed','deployment_return','deployment_discounted_return','common_repeated_return_gamma09',
      'training_bootstrap_contribution_fraction','training_exploration_fraction','training_unseen_fraction','observed_aliased_states','state_action_reward_conflicts')})
 csv_write(out/'reward_and_exploration.csv',reward_rows)
 controlled=[c for c in comparisons if c['scope']=='POLUS_matched_seed']
 native_baseline=[]
 for model,target in cells:
  r=next(r for r in native if r['model']==model and r['target']==target and r['variant']=='q_reachable_v3')
  native_baseline.append([model.upper(),f'{target:.0%}',f"{100*r['validation_accuracy']:.2f}",f"{100*r['validation_macro_f1']:.2f}",f"{r['width0']}/{r['width1']}",f"{100*r['achieved_sparsity']:.3f}",f"{r['search_seconds']:.2f}"])
 report=['# V4 POLUS controlled tabular-Q campaign: verified analysis','',
 '**92/92 results pass independent offline verification.** All models reach the exact target band. '
 'Menu {1,2} matches its Greedy control in all12 policy comparisons at much higher search cost; '
 'neither300 episodes, exploration decay, reward reformulation nor gamma0 produces a consistent quality advantage across the four model–target conditions. '
 'Small improvements in some random-tie/gamma0 runs coexist with different achieved compression/widths and worse cases. '
 'These are validation-only diagnostics on one frozen training checkpoint per model.','',
 '## Evidence and independent verification','',
 'Analysis date:2026-10-29. Frozen scientific source:`61e693440b53d3cac3136600f3208a73c02492db`; '
 'archive SHA256:`d80ea49be02c5e08ec7169a323b91c368875e4b42fab5dbeffb655ffca468704`. '
 'The primary archive contains84 Q tasks and8 deterministic menu-matched Greedy tasks. '
 'No capacity/smoke task, skipped optional seed or duplicate platform repeat enters primary summaries.','',
 f"All1,285 inventory files and92 completed-marker output sets were rehashed. Verification covers92 physical checkpoints plus20 historical references, "
 f"{v['primary_training_episodes']:,} primary learning episodes, {v['primary_transitions']:,} transitions, {v['primary_q_entries']:,} final Q entries, "
 'and all300 snapshots. It independently checks exact integer parameter geometry, width-dependent BFS reachability/horizon, legal actions, '
 'magnitude-based retained IDs at every transition, reward/terminal penalty arithmetic, seeded RNG/exploration/tie choices, Bellman updates, '
 'visited-cell Q coverage and snapshot Q/RNG state. Final models reload strictly and equal dense-tensor projections; '
 'all final validation accuracy/F1 values are independently recomputed and agree.','',
 'Verification uses existing local Torch2.2.2/PyG2.6.1 on CPU, two threads; recorded POLUS results use Torch2.0.1/PyG2.4.0, '
 'POWER8 ppc64le, four threads. Archived native metrics/timings remain the reported results. '
 'Every non-validation label is masked to−1; no test-set metric or new search/training/rollout/timing experiment is performed. '
 'Intermediate validation observations are audited/replayed, not reevaluated. Greedy counterfactual candidate outputs were not recorded, '
 'so their argmax is not independently reexecuted; chosen-action legality, counts, rewards, projection, reload and final metrics are checked. '
 'The checker has10 offline tests, including rejection of altered rewards, Q values, IDs, terminal behavior, actions and coverage.','',
 '## Correct controls and statistical units','',
 '- **A menus, B300, D ties, E reward, F gamma:** the controlled remote reference is V3 `q_reachable_v3`, POLUS seed101. '
 'Thus native single-factor comparisons for these variants have **n=1 per model/target**, not n=3. D is compared to constrained Q, '
 'not to the old unconstrained V3 random-tie variant.','',
 '- **C decay:** compare to V4 B300 (same300 episodes), with all three matched POLUS policy seeds101/5001/5002. '
 'Comparing C to the100-episode reference would change two factors.','',
 '- **Local V3 seeds5001/5002:** retained as explicit cross-platform developmental references only. '
 'Their contrasts are labelled `cross_platform_reference_only`; cross-platform search ratios are deliberately absent. '
 'Local101 remains a bridge reference, not an additional independent sample. No n=4 pooling occurs.','',
 '- **Matched Greedy:** each alternative menu has one deterministic POLUS control per model/target; compare three Q policy seeds to that fixed control. '
 'Standard-menu Greedy is reused from V3 POLUS101. For E it remains a descriptive control with a reward mismatch, not an incremental-reward Greedy ablation.','',
 'All Q seeds share dense training seed101 and the same graph/split; they quantify policy RNG variation, not variation across trained GNNs or datasets. '
 'SD below is sample SD across three policy seeds, not a confidence interval. Greedy n=1 has no estimated SD. '
 'Within-band compression differences are retained; equal requested targets do not imply equal architectures.','',
 'Native historical constrained-Q reference (n=1, seed101):','',
 table(['Model','Target','Acc %','Macro-F1 %','Widths','Compression %','Search s'],native_baseline),'',
 'All16 existing Q references were checked against committed V3 completion/checkpoint hashes. '
 'For **all16 references**, B300 has exactly the same first100 training observations and exactly the same Q table in its100-episode snapshot. '
 'This supports the controlled episode-prefix implementation. A snapshot is not a new100-episode greedy deployment evaluation, '
 'and this agreement does not create missing POLUS timing/replicate evidence for5001/5002.','',
 'Local reference endpoints (reported individually, not pooled into native controls):','']
 local_table=[]
 for model,target in cells:
  values=[]
  for seed in (5001,5002):
   r=next(r for r in baselines if r['platform']=='local' and r['model']==model and r['target']==target and r['policy_seed']==seed)
   values.append(f"{100*r['validation_accuracy']:.2f}/{100*r['validation_macro_f1']:.2f}; {r['width0']}/{r['width1']}")
  local_table.append([model.upper(),f'{target:.0%}',*values])
 report += [table(['Model','Target','Local5001: Acc/F1; widths','Local5002: Acc/F1; widths'],local_table),'',
 '## Primary POLUS results','',
 'Accuracy/F1: percentages, mean±sample SD for Q (n=3). Compression: exact parameter reduction, displayed as a percentage mean; '
 '`runs.csv` preserves every fraction, count, width, retained ID and policy seed. Width lists follow101;5001;5002, with one width pair for Greedy. '
 'Search cost includes policy/environment/recording work; it is not inference latency.','']
 for model,target in cells:
  rows=[]
  for variant,label in zip(ORDER,LABELS):
   g=group(model,target,variant)
   rows.append([label,len(g),mu_sd([r['validation_accuracy'] for r in g],True),mu_sd([r['validation_macro_f1'] for r in g],True),
       f"{100*statistics.mean(r['achieved_sparsity'] for r in g):.3f}",'; '.join(f"{r['width0']}/{r['width1']}" for r in sorted(g,key=lambda r:r['policy_seed'])),
       f"{statistics.mean(r['search_seconds'] for r in g):.2f}"])
  report += [f'### {model.upper()}, target {target:.0%}','',table(['Variant','n','Accuracy %','Macro-F1 %','Compression %','Widths','Search s (mean)'],rows),'']
 report += ['![Quality versus search cost](figures/quality_cost.png)','',
 'Figure1: individual policy seeds and Q means±sample SD; controls and V3 references have n=1. '
 'The log cost axis is deliberate. Apparent quality comparisons also require achieved compression and retained-ID checks.','',
 '## Controlled factor effects','',
 'Each table entry is treatment−correct control in **accuracy / macro-F1 percentage points**. '
 'A/B/D/E/F entries use matched native seed101 only. C entries are the mean of three native seed-paired differences against B300; '
 'individual deltas and compression changes are in `comparisons.csv`.','']
 effect=[]
 for variant,label in zip(ORDER[:7],LABELS[:7]):
  values=[]
  for model,target in cells:
   g=[c for c in controlled if c['model']==model and c['target']==target and c['treatment']==variant]
   assert len(g)==(3 if variant=='q_v4_decay300' else 1)
   values.append(f"{statistics.mean(c['accuracy_delta_pp'] for c in g):+.2f} / {statistics.mean(c['f1_delta_pp'] for c in g):+.2f}")
  effect.append([label,*values])
 report += [table(['Factor','GCN30','GCN50','GraphSAGE30','GraphSAGE50'],effect),'',
 '![Controlled native effects](figures/controlled_effects.png)','',
 '**A — action menus.** {1,2} produces the same accuracy/F1 as matched Greedy in12/12 policy comparisons, '
 'including both models and targets; the width-percent menu ties in10/12 and is worse in2/12, with no improvements. '
 'At30%, {1,2} ends at23/16 in all six Q runs. The percent menu sometimes removes an additional second-layer channel. '
 'Higher observed Q coverage for small menus partly reflects a smaller legal-action denominator; it is not evidence of greater full-state learning coverage. '
 'Percent deletions use ceiling on the current width, deduplicated:32→{2,4},30→{2,3},10→{1}; this is not a fixed {1,2} menu.','',
 '**B — episode budget.** All100-episode prefixes match archived controls exactly. At native seed101, increasing to300 episodes '
 'reduces GCN30 accuracy by5.18pp/F1 by10.88pp while improving GraphSAGE30 by2.59pp/4.23pp; '
 'both50% endpoints remain unchanged. This is not consistent improvement with more episodes. '
 'Recorded seed101 search is2.75–2.93× the historical100-episode reference, but V4 adds decision/snapshot bookkeeping '
 'and uses a different reachability implementation; this inter-campaign timing ratio is descriptive, not an isolated causal estimate of episode cost.','',
 '**C — exploration schedule.** Against B300, mean30% accuracy changes are+0.43pp (GCN) and+1.36pp (GraphSAGE), '
 'with F1 changes+1.12pp/+3.18pp. In each30% cell two seeds improve and one worsens. All50% accuracy/F1 endpoints tie. '
 'The schedule changes initial/final exploration and mean nominal epsilon (.225 versus.2), not decay alone at a fixed exploration budget. '
 'Observed training exploration fractions are23.7–26.3% for C versus19.7–20.2% for B, weighted by actual trajectory lengths. '
 'Mean recorded C search is6.0%,21.6%,14.8%,23.1% lower than B for GCN30/GCN50/GraphSAGE30/GraphSAGE50 respectively; '
 'some individual pairs are slower. These are observed search/evaluation differences, not hardware speedups or convergence guarantees.','',
 '**D — equal-Q ties.** The same reachability filter is retained. At seed101, GCN endpoints and GraphSAGE50 tie the constrained reference; '
 'GraphSAGE30 gains2.59pp accuracy/4.23pp F1 and changes23/15→23/16. Across all V4 seeds, '
 'GraphSAGE50 has two small improvements over standard Greedy, but with extra second-layer pruning (28/30 channels instead of32). '
 'The observed result is conditional on tie resolution and selected architectures; it is not a general superiority claim.','',
 'D uses the same Python RNG for epsilon decisions and tied-max choices. Extra tie draws can also change later realized exploration draws. '
 'Thus matched seed means the same initial RNG seed, not a frozen common exploration-action stream; the contrast evaluates the declared random-tie policy as a whole.','',
 '**E — incremental-drop reward.** At seed101, GCN30 loses0.55pp accuracy/1.47pp F1 and GraphSAGE30 loses2.59pp/7.60pp relative to the native baseline; '
 'both50% cells tie. GraphSAGE30 selects23/9–23/10 across seeds, compared with Greedy23/16, and its mean F1 is58.07% versus72.87% for Greedy. '
 'Achieved compression therefore differs: the reward effect includes its change to the selected trajectory/model, not quality at fixed widths.','',
 'The E equation is r_t=0.6Δs−0.4(D_t−D_prev)+P_t. Independently checked undiscounted sums telescope to0.6s_T−0.4D_T+P_T. '
 'With gamma.9, however, the discounted drop term is gamma^(T−1)D_T + (1−gamma)Σ[t=1..T−1]gamma^(t−1)D_t. '
 'It retains intermediate-gap weighting and is not a policy-invariant terminal-only reformulation. '
 'Reward values from different definitions/gammas must not be ranked as a common quality metric. '
 '`reward_and_exploration.csv` includes offline rescoring of each already stored deployment path under the common repeated-drop/gamma.9 objective; '
 'no new policy was run.','',
 '**F — gamma0.** Every update has zero future-credit contribution, as verified from the trace. '
 'At seed101, GCN30 loses2.22pp accuracy/4.82pp F1, GCN50 gains0.92pp/1.87pp with widths16/21 instead of16/32 '
 '(0.56pp more compression), and GraphSAGE endpoints tie the native reference. '
 'Gamma0 does not make the learned tabular policy equivalent to immediate Greedy: unseen actions, training visits and the aggregated state remain relevant.','',
 'In particular, gamma0 Q does not evaluate every legal candidate immediately before choosing, whereas Greedy does.','',
 '## Matched Greedy and quality versus cost','',
 'The following are **descriptive counts across12 heterogeneous model–target–seed comparisons per variant**, '
 'not12 independent trained models or a pooled treatment effect. Each row reports accuracy better/tied/worse than its menu control. '
 'For E the repeated-drop standard Greedy reward is mismatched, so that row is descriptive only.','',
 table(['Variant','Better / tied / worse','Observed search ratio Q/Greedy','Reward matched'],
   [[label,f"{g['accuracy_better']} / {g['accuracy_equal']} / {g['accuracy_worse']}",f"{g['min_search_ratio']:.1f}–{g['max_search_ratio']:.1f}×",'yes' if g['reward_control_matched'] else 'no; descriptive'] for label,g in zip(LABELS[:7],greedy_rows)]),'',
 'For the two new menus the fully matched control costs make the tradeoff particularly clear: '
 '{1,2} Q costs44–56× Greedy with no accuracy improvement; percent-menu Q costs43–53× with ties or losses. '
 'Some D/F runs have slightly higher validation accuracy than standard Greedy at greater search cost and different compression. '
 'They are not a uniform win, and there is no measured inference-latency advantage in this campaign. '
 'Use the per-seed F1, exact compression, widths and retained-ID comparisons, rather than selecting the best policy seed.','',
 '## Coverage, unseen actions and stopping','',
 '![Coverage and unseen actions](figures/coverage_and_unseen.png)','',
 'Coverage means learned Q entries divided by legal state–action cells in the **visited aggregated states**, including terminal states with0 legal cells. '
 'It is not coverage of every possible state or retained-channel mask. Menus change the denominator. '
 'At30%, menu {1,2} has mean55–56% coverage and percent menus64–65%, while the native standard reference has36–38% (seed101). '
 'Absolute entries/visited states/denominators and SDs are retained in the CSVs.','',
 'All24 B/C deployments choose only previously updated actions, yet several30% endpoints are below Greedy. '
 'Conversely D can choose unseen actions and still improve particular endpoints. Unseen fraction alone is not a monotone quality indicator. '
 'Observed retained-ID aliasing occurs in82/84 Q runs (0–28 aliased aggregate states per run); '
 'state–action reward conflicts are reported separately. The representation is unchanged across factors, '
 'so this diagnostic does not causally isolate aliasing as the reason for an outcome.','',
 '**Stopping:**89/92 deployments end with `target_band`; three menu-{1,2}30% deployments end at the20-action limit '
 '(GCN seed5002; GraphSAGE seeds101/5002). All three are in band and preserve the original terminal-priority rule. '
 'Every deployment pays terminal penalty once (zero inside the band). STOP is unavailable outside the band; '
 'there is no evidence here that Q learned when to stop independently of the feasibility controller.','',
 'All audited training endpoints are also in band, so terminal penalties are zero throughout these constrained trajectories. '
 'The reward contrast concerns the running drop penalty, not successful propagation of a nonzero off-target terminal penalty.','',
 '![Exploratory learning endpoints](figures/learning_endpoints.png)','',
 'Training endpoint curves summarize exploratory episodes, not greedy deployment policies. Shading is SD across three25-episode policy-seed block means, '
 'not a confidence interval. Snapshots store Q and training RNG state only; no extra intermediate greedy policy evaluation is available.','',
 '## Execution cost and queue waiting','',
 f"Primary search sums to{sum(r['search_seconds'] for r in runs)/60:.2f} minutes. LSF primary active job time sums to{sum(j['active_wall_seconds'] for j in lsf[1:])/60:.2f} minutes; "
 f"including capacity job1607752 it is{sum(j['active_wall_seconds'] for j in lsf)/60:.2f} minutes. "
 'The reserved upper bound was250 minutes, below the strict300-minute maximum. Actual active time was well below the4–5h planning target; '
 'the campaign was not expanded or repeated to fill unused capacity. Optional seeds5003/5004 remain uniformly absent.','',
 table(['Condition','Job ID','Active min','Queue-wait min','Turnaround min'],[[j['condition'],j['job_id'],f"{j['active_wall_seconds']/60:.2f}",f"{j['queue_wait_seconds']/60:.2f}",f"{j['turnaround_seconds']/60:.2f}"] for j in lsf]),'',
 'Queue waiting is LSF turnaround minus run time. It is excluded from active budget. Waiting intervals overlap across jobs, '
 'so their sum is not elapsed campaign duration; maximum primary turnaround is216.87 minutes. '
 'CPU seconds are also distinct from wall seconds and are preserved in `execution_times.csv`. '
 'Search/worker time differences include imports, frozen-input validation, serialization and orchestration. '
 'Timing observations are not repeated isolated performance benchmarks; native V3 comparisons additionally differ in instrumentation, '
 'and local V3 timings are never pooled with POLUS.','',
 '## Reproducibility and limits','',
 '`runs.csv` has exactly92 primary rows; `summary.csv` has36 native model/target/variant groups; '
 '`baseline_runs.csv` keeps20 historical references and their platforms separate. '
 '`comparisons.csv` separates36 native seed-paired factor comparisons,48 cross-platform references and84 Greedy comparisons. '
 '`learning_blocks.csv`, `snapshots.csv`, `deployment_decisions.csv`, `prefix_checks.csv`, `reward_and_exploration.csv`, '
 '`verification.json` and `provenance.json` retain supporting diagnostics. Full13,200 per-episode rows remain outside Git with the analysis record.','',
 f"External analysis root:`{analysis}`. Original collection:`{collection}`. "
 'The original archive,1,285 source files, launch/collection receipts, algorithms, earlier evidence and SITITO manuscript remain unchanged. '
 'Analysis tools are versioned here under `tools/`; plotting reads only derived files using the existing Python3.8/Matplotlib3.7.5 environment. '
 'No package/environment/network/server operation is needed for this analysis. Compact artifact hashes are in `analysis_manifest.json`.','',
 'Reproduce verification/aggregation using the existing Torch environment, from repository root:','',
 '```bash',
 'PYTHONPATH=src python research/experiments/polus_v4_diagnostic_20261029/results/tools/verify_and_aggregate.py \\',
 f'  --collection "{collection}" \\',
 f'  --inputs "{collection.parent}/polus-v4-diagnostic-20261029/inputs/manifest.json" \\',
 '  --output /path/to/fresh-external-output',
 '```','',
 'Interpretation remains developmental: one graph, one dense training checkpoint per model, n=3 policy seeds, '
 'n=1 matched native historical reference for most factors, validation used for adaptive decisions, no test-set evaluation. '
 'No causal assertion extends beyond these controlled conditions. The evidence supports keeping strong menu-matched Greedy controls '
 'and qualifying any learning benefit at comparable achieved compression, rather than declaring a final best reward/gamma/exploration setting.','']
 text='\n'.join(report)
 text=re.sub(r'\b(All|all|Full|full|three|four|both|Both|at|At|to|by|of|for|in|with|from|and|the|than|including|exactly|across|has|contains|over|under|is|was|sums)(?=\d)',r'\1 ',text)
 text=re.sub(r'(?<=[A-Za-z]),(?=\d)',', ',text)
 for a,b in [('All1,285','All 1,285'),('All16','All 16'),('all16','all 16'),('All24','All 24'),('all300','all 300'),
   ('contains84','contains 84'),('and8 ','and 8 '),('across12','across 12'),('in12/','in 12/'),('in10/','in 10/'),('in2/','in 2/'),
   ('At30%','At 30%'),('All50%','All 50%'),('all92','all 92'),('one100-','one 100-'),('exactly92','exactly 92'),
   ('has36','has 36'),('keeps20','keeps 20'),('separates36','separates 36'),('across12','across 12'),('has10','has 10'),
   ('checks92','checks 92'),('covers92','covers 92'),('plus20','plus 20'),('across16','across 16'),('on one','on one')]:text=text.replace(a,b)
 (out/'REPORT.md').write_text(text)
 save(out/'analysis_record.json',dict(source_commit=v['source_commit'],processed_root=str(processed),analysis_root=str(analysis),
   audit_log_sha256=sha(analysis/'audit-final.log'),test_log_sha256=sha(analysis/'analysis-tests-final.log'),offline_tests=10,
   external_episode_rows=dict(path=str(processed/'episodes.csv'),rows=13200,sha256=sha(processed/'episodes.csv')),
   verification_script_sha256=sha(Path(__file__).with_name('verify_and_aggregate.py')),report_script_sha256=sha(__file__),
   plot_script_sha256=sha(Path(__file__).with_name('plot_results.py')),new_experiments=False,algorithms_modified=False))
 print(json.dumps(dict(report=str(out/'REPORT.md'),primary_rows=len(runs),summary_groups=len(summaries),historical_references=len(baselines)),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--processed',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
 p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();package(a.processed,a.analysis,a.collection,a.output)
