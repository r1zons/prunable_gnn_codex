"""Read archived CSV/JSON and code only. Never imports torch or executes a GNN."""
from pathlib import Path
import csv, json, hashlib, subprocess, statistics as st
from fractions import Fraction
from collections import defaultdict

OUT=Path(__file__).resolve().parents[1]
REPO=Path('/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_codex')
HEAD='34e8ca89599b56ef14bb404527b6a72d6b26d403'
V3='research/experiments/local_structured_v3_sprint_20261027/'
V2='research/experiments/local_structured_v2_q_random_pilot_20261027/'
paths=[V3+'results/'+x for x in ['REPORT.md','runs.csv','summary.csv','platform_pairs.csv','fixed_probe_timing.csv','verification.json','provenance.json','logit_comparison.json']]
paths += [V3+'README.md']+[V2+x for x in ['README.md','random_trials.csv','random_summary.csv','q_pilot.csv']]
paths += ['research/experiments/local_structured_v2_cora_matrix_20261027/'+x for x in ['summary.csv','README.md','provenance.json']]
paths += ['research/methodology/LOCAL_STRUCTURED_V2_SPEC.md','research/experiments/local_cpu_2026/audit/AUDIT_REPORT.md','research/experiments/local_cpu_2026/audit/aggregate_by_condition_method_target.csv','research/experiments/local_cpu_2026/audit/condition_matrix.csv','research/experiments/polus_cpu_2026/PLATFORM_COMPARISON.md','research/experiments/polus_cpu_2026/README.md','research/experiments/polus_cpu_2026/evidence/pilot_validation_summary.csv','research/publications/sitito_2026/PLAN.md']
paths += ['src/gnn_pruning/'+x for x in ['pruning/adaptive_v2.py','pruning/experimental_v3.py','pruning/static_v2.py','surgery/local_v2.py','experiments/v3_sprint.py']]
manifest=[]
for p in paths:
    b=(REPO/p).read_bytes()
    frozen=subprocess.check_output(['git','-C',str(REPO),'show',HEAD+':'+p])
    assert b==frozen,p
    sha=hashlib.sha256(b).hexdigest()
    blob=subprocess.check_output(['git','-C',str(REPO),'rev-parse',HEAD+':'+p],text=True).strip()
    dest=OUT/'evidence'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    manifest.append(dict(path=p,commit=HEAD,git_blob_sha1=blob,sha256=sha,bytes=len(b),url=f'https://github.com/r1zons/prunable_gnn_codex/blob/{HEAD}/{p}'))
def read(p):return list(csv.DictReader((REPO/p).open()))
def save(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('source_manifest.csv',manifest)
runs=read(V3+'results/runs.csv'); local=[r for r in runs if r['platform']=='local' and r['variant']!='fixed_probe_v3']
assert len(runs)==132 and len(local)==96
groups=defaultdict(list)
for r in runs:
    factor=1 if r['model']=='gcn' else 2
    a,b=int(r['width0']),int(r['width1'])
    p=factor*(1433*a+a*b+7*b)+a+b+7
    p0=factor*(1433*32+32*32+7*32)+32+32+7
    assert p==int(r['parameters_after']) and p0==int(r['parameters_before'])
    s=Fraction(p0-p,p0); target=Fraction(r['target'])
    assert abs(float(s)-float(r['achieved_sparsity']))<1e-14
    assert (abs(s-target)<=Fraction(1,100))==(r['target_hit']=='True')
    if r in local:groups[(r['model'],r['target'],r['variant'])].append(r)
summary=read(V3+'results/summary.csv'); checked=[]
for (m,t,v),rows in groups.items():
    z=next(r for r in summary if r['platform']=='local' and r['model']==m and r['target']==t and r['variant']==v)
    result={'model':m,'target':t,'variant':v,'n_policy':len(rows),'n_dense_training':1,'hits':sum(r['target_hit']=='True' for r in rows)}
    for col in ['achieved_sparsity','validation_accuracy','validation_macro_f1','search_seconds']:
        vals=[float(r[col]) for r in rows]
        assert abs(st.mean(vals)-float(z[col+'_mean']))<1e-12
        assert abs(st.stdev(vals)-float(z[col+'_sample_std']))<1e-12
        result[col+'_mean']=st.mean(vals);result[col+'_sd']=st.stdev(vals)
    result['width_pairs']='; '.join(f"{r['policy_seed']}:{r['width0']}x{r['width1']}" for r in rows)
    result['parameter_counts']='; '.join(f"{r['policy_seed']}:{r['parameters_after']}" for r in rows)
    result['validation_evaluations']='; '.join(r['validation_evaluations'] for r in rows)
    result['source_run_ids']='; '.join(r['run_id'] for r in rows)
    checked.append(result)
save('verified_v3_results.csv',checked)
random=read(V2+'random_trials.csv');rs=read(V2+'random_summary.csv');rv=[]
assert len(random)==80
for z in rs:
    rows=[r for r in random if r['model']==z['model'] and r['target_sparsity']==z['target_sparsity']]
    vals=[float(r['validation_accuracy']) for r in rows]
    assert len(rows)==20 and abs(st.mean(vals)-float(z['accuracy_mean']))<1e-12 and abs(st.stdev(vals)-float(z['accuracy_sd']))<1e-12
    rv.append(dict(model=z['model'],target=z['target_sparsity'],n_masks=20,mean=st.mean(vals),sd=st.stdev(vals),minimum=min(vals),maximum=max(vals)))
save('verified_random_results.csv',rv)
pairs=read(V3+'results/platform_pairs.csv');assert len(pairs)==34
for r in pairs:
    for col in ['same_actions','same_retained_ids','same_checkpoint_tensors','same_training_actions']:
        assert r[col]=='True'
    assert float(r['dense_validation_accuracy_delta'])==0 and float(r['final_validation_accuracy_delta'])==0
verification=json.loads((REPO/(V3+'results/verification.json')).read_text())
assert (verification['verified_training_episodes'],verification['verified_transitions'],verification['verified_q_entries'])==(9600,110830,75467)
# Inspect historical table coverage without importing its test metrics into claims.
historical={p:len(read(p)) for p in paths if p.endswith('.csv') and ('local_cpu_2026' in p or 'polus_cpu_2026' in p)}
audit={'date':'2026-10-27','research_runs_executed':0,'models_loaded':0,'head':HEAD,'v3_executable_sha':'7e2c760e988b605e305615dea147236e8b0c59a3','source_files_checked':len(manifest),'v3_rows_checked':len(runs),'local_method_rows':96,'summary_groups_recomputed':len(checked),'random_masks_recomputed':80,'paired_platform_rows':34,'archived_verification_not_rerun':{k:verification[k] for k in ['status','verified_training_episodes','verified_transitions','verified_q_entries']},'historical_tables_read_separately':historical}
(OUT/'analysis_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(audit,ensure_ascii=False,indent=2))
