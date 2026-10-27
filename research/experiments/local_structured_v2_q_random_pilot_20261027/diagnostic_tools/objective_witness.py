"""Derive feasible Q action-sequence witnesses from prior validation-only Greedy traces."""
import csv,json
from pathlib import Path
from pilot_runner import ROOT
prior=list(csv.DictReader((Path.cwd()/'research/experiments/local_structured_v2_cora_matrix_20261027/summary.csv').open()))
prov=json.loads((ROOT/'provenance.json').read_text());comparators=[];witnesses=[]
for row in prior:
    if row['method'] not in ('random_local_v2','magnitude_local_v2','adaptive_greedy_local_v2'):continue
    import torch
    ck=torch.load(Path(row['artifact_dir'])/'pruned.pt',map_location='cpu',weights_only=True)
    comparators.append({**{k:row[k] for k in ('model','method','target_sparsity','achieved_sparsity','validation_accuracy','validation_macro_f1','search_seconds')},'hidden_widths':list(map(len,ck['original_kept_ids'])),'checkpoint':str(Path(row['artifact_dir'])/'pruned.pt')})
    if row['method']!='adaptive_greedy_local_v2':continue
    trace=json.loads((Path(row['artifact_dir'])/'trace.json').read_text())['deployment'];assert len(trace)==2
    model=row['model'];target=float(row['target_sparsity']);dense=float(prov['selected'][model]['dense_validation_metrics']['accuracy'])
    import numpy as np
    dense=float(np.float32(dense));prev=0.;rewards=[]
    for e in trace:
        assert e['action'].startswith('prune:')
        rewards.append(.6*(e['sparsity']-prev)-.4*max(0.,dense-e['validation_accuracy']));prev=e['sparsity']
    # Q does not auto-terminate on target: append the legal STOP as its own transition.
    rewards.append(-2.*max(0.,abs(prev-target)-.01))
    witness_return=sum(.9**i*r for i,r in enumerate(rewards))
    comparisons=[]
    for p in sorted(ROOT.glob(f'q_runs/{model}_target{int(target*100)}_*lambda2/result.json')):
        q=json.loads(p.read_text());deployment=json.loads((p.parent/'trace.json').read_text())['deployment'];ret=sum(.9**i*e['reward'] for i,e in enumerate(deployment))
        comparisons.append(dict(q_run=q['run_id'],q_deployed_discounted_return=ret,witness_minus_q=witness_return-ret))
    witnesses.append(dict(model=model,target_sparsity=target,source_trace=str(Path(row['artifact_dir'])/'trace.json'),actions=[e['action'] for e in trace]+['stop'],final_hidden_widths=trace[-1]['widths_after'],final_compression=prev,final_validation_accuracy=trace[-1]['validation_accuracy'],q_compatible_rewards=rewards,q_discounted_return=witness_return,comparisons=comparisons,scope='A feasible action-sequence witness computed from previously verified validation traces, under unchanged Q reward/STOP semantics with horizon20 and lambda2. Not a matched-compression algorithm-superiority claim; no new pruning/model evaluation.'))
(ROOT/'previous_validation_comparators.json').write_text(json.dumps(comparators,indent=2));(ROOT/'objective_witness.json').write_text(json.dumps(witnesses,indent=2))
for w in witnesses:print(w['model'],w['target_sparsity'],'feasible return',w['q_discounted_return'],'vs',w['comparisons'])
