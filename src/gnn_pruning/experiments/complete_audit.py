"""Offline independent audit of the submitted POLUS campaign; inference only.

The explicit matrix comes from the pinned source manifest, not inferred folder
names or a newly generated split. No training, pruning strategy or LSF is called.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import platform

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import f1_score

from gnn_pruning.models import build_model
from .complete_collection import digest,inventory_audit,required_files
from .complete_audit_trace import projected_state,context,audit_behavior,require,close

METRIC_TOL=1e-12
COMPRESSION_TOL=1e-12


def tensor_identity(tensors: dict) -> str:
    """Canonical named-tensor hash independently specified from saved file bytes."""
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def save(path: Path | str,value: dict) -> None:
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def metrics(model: torch.nn.Module,data: Data,split: dict) -> dict:
    """One finite frozen forward, reused for final val/test metric verification."""
    model.eval()
    with torch.no_grad():logits=model(data)
    require(tuple(logits.shape)==(int(data.num_nodes),int(data.y.max())+1),'FORWARD_SHAPE')
    require(bool(torch.isfinite(logits).all()),'NONFINITE_FORWARD')
    pred=logits.argmax(-1).cpu().numpy();labels=data.y.cpu().numpy()
    result={}
    for name in ('val','test'):
        idx=np.array(split[name],dtype=np.int64);truth=labels[idx];prediction=pred[idx]
        result[name]=dict(accuracy=float(np.mean(truth==prediction)),macro_f1=float(f1_score(truth,prediction,average='macro',zero_division=0)))
    return result


def compare_metrics(actual: dict,recorded: dict,prefix: str) -> None:
    for split in ('val','test'):
        for metric in ('accuracy','macro_f1'):
            close(actual[split][metric],recorded[split][metric],prefix+'_'+split+'_'+metric,METRIC_TOL)


def validate_split(split: dict,n: int) -> str:
    require(set(split)=={'train','val','test'},'SPLIT_KEYS')
    flat=[value for key in ('train','val','test') for value in split[key]]
    require(all(type(x)is int for x in flat) and sorted(flat)==list(range(n)),'SPLIT_NOT_DISJOINT_COVER')
    require(len(split['train'])==int(n*.6) and len(split['val'])==int(n*.2) and len(split['test'])==n-int(n*.6)-int(n*.2),'SPLIT_RATIOS')
    return hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()


def reconstruct(cp: dict,kind: str) -> torch.nn.Module:
    require(cp['model_name']==kind,'CHECKPOINT_MODEL_IDENTITY')
    model=build_model(kind,**cp['model_config']).eval()
    model.load_state_dict(cp['model_state_dict'],strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in cp['model_state_dict'].values()),'NONFINITE_CHECKPOINT')
    return model


def inspect_marker(directory: Path) -> None:
    marker=read(directory/'completed.json')
    actual={p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='completed.json'}
    require(marker['files']==actual,'COMPLETED_OUTPUT_HASH_MISMATCH')
    require(set(actual)=={'report.json','pruned.pt'},'UNEXPECTED_RUN_OUTPUT_SET')


def dense_audit(root: Path,c: dict,protocol: dict,data: Data,graph_hash: str) -> dict:
    directory=root/'dense'/c['id'];cp=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    training=read(directory/'training.json');cert=read(directory/'validated.json')
    architecture=dict(in_channels=int(data.num_features),out_channels=int(data.y.max())+1,**protocol['architecture'])
    require(cp['model_config']==architecture,'DENSE_ARCHITECTURE')
    require(cp['seed']==c['seed'] and cp['training_protocol']==protocol['training'],'DENSE_SEED_OR_TRAINING')
    require(cp['source_commit']==protocol['source_commit']==cert['source_commit'],'DENSE_SOURCE')
    require(cp['graph_tensor_sha256']==graph_hash==cert['graph_tensor_sha256'],'DENSE_GRAPH_IDENTITY')
    split=cp['split'];split_hash=validate_split(split,int(data.num_nodes))
    require(split_hash==cert['split_sha256'],'SAVED_SPLIT_HASH')
    checkpoint_hash=digest(directory/'checkpoint.pt')
    require(checkpoint_hash==training['checkpoint_sha256']==cert['checkpoint_sha256'],'DENSE_CHECKPOINT_FILE_HASH')
    require(cert['status']=='PASS','DENSE_EXECUTION_VALIDATION_FAILED')
    model=reconstruct(cp,c['model']);identity=tensor_identity(cp['model_state_dict'])
    require(identity==cp['tensor_sha256']==cert['tensor_sha256'],'DENSE_TENSOR_HASH')
    actual=metrics(model,data,split)
    compare_metrics(actual,dict(val=cert['validation'],test=cert['test']),'DENSE_METRIC')
    for key in ('accuracy','macro_f1'):close(actual['val'][key],training['validation'][key],'DENSE_TRAINING_VALIDATION_'+key,METRIC_TOL)
    require(training['source_commit']==protocol['source_commit'] and not training['test_labels_exposed'],'DENSE_TRAINING_PROVENANCE')
    reused=bool(training.get('reused'))
    if reused:
        require(c['seed']==101 and cp.get('origin')==training['origin'],'DENSE_REUSE_ORIGIN')
        origin=training['origin'];proof=root/'inputs/reuse'/c['id']
        require(read(proof/'verified.json')==origin,'DENSE_ORIGIN_PROOF')
        require(digest(proof/'checkpoint.pt')==origin['checkpoint_sha256'],'DENSE_ORIGINAL_BYTES')
        old=torch.load(proof/'checkpoint.pt',map_location='cpu',weights_only=True)
        require(tensor_identity(old['model_state_dict'])==identity==origin['tensor_sha256'],'DENSE_REUSED_TENSORS')
        require(origin['split']==split and origin['graph_tensor_sha256']==graph_hash,'DENSE_REUSED_GRAPH_SPLIT')
        training_seconds=None
    else:
        history=training['history'];result=training['result']
        require(1<=len(history)<=protocol['training']['max_epochs'],'DENSE_TRAINING_EPOCH_COUNT')
        require(int(result['epochs_ran'])==len(history),'DENSE_RECORDED_EPOCH_COUNT')
        require([int(h['epoch']) for h in history]==list(range(1,len(history)+1)),'DENSE_EPOCH_ORDER')
        losses=[h['val_loss'] for h in history];best=min(range(len(losses)),key=lambda i:losses[i])
        require(int(result['best_epoch'])==best,'DENSE_BEST_EPOCH_VALIDATION_SELECTION')
        close(result['best_val_loss'],losses[best],'DENSE_BEST_VALIDATION_LOSS')
        require(all(math.isfinite(h[k]) for h in history for k in ('val_loss','train_loss','val_acc','train_acc','elapsed_sec')),'NONFINITE_TRAINING_LOG')
        running_best=float('inf');counter=0;best_index=-1
        for index,h in enumerate(history):
            require(counter<protocol['training']['early_stopping_patience'],'DENSE_TRAINING_AFTER_PATIENCE')
            if h['val_loss']<running_best:running_best=h['val_loss'];counter=0;best_index=index
            else:counter+=1
            require(h['early_stopping_counter']==counter and h['best_epoch']==best_index+1,'DENSE_EARLY_STOPPING_LOG')
        require(len(history)==protocol['training']['max_epochs'] or counter>=protocol['training']['early_stopping_patience'],'DENSE_PREMATURE_TRAINING_END')
        close(result['final_val_loss'],history[-1]['val_loss'],'DENSE_FINAL_VALIDATION_LOSS')
        close(result['final_train_loss'],history[-1]['train_loss'],'DENSE_FINAL_TRAINING_LOSS')
        training_seconds=training['wall_seconds'];require(training_seconds>0,'DENSE_TRAINING_TIME')
        require(sum(h['elapsed_sec'] for h in history)<=training_seconds+1e-6,'DENSE_EPOCH_WALL_ACCOUNTING')
    return dict(checkpoint=cp,model=model,split=split,metrics=actual,checkpoint_hash=checkpoint_hash,
        split_hash=split_hash,tensor_hash=identity,parameters=sum(p.numel() for p in model.parameters()),
        reused=reused,training_seconds=training_seconds,checkpoint_bytes=(directory/'checkpoint.pt').stat().st_size,
        origin_source=cp.get('origin',{}).get('origin_source_commit',protocol['source_commit']))


def basic_row(c: dict,run: dict) -> dict:
    return dict(run_id=run['id'],condition_id=c['id'],dataset=c['dataset'],architecture=c['model'],
        dense_training_seed=c['seed'],method=run['method'],implementation=run['implementation'],
        policy_seed=run['policy_seed'] if run['method'] in ('static_random','sequential_random','tabular_q') else None,
        requested_compression=run['target'],collected=False,verified=False,audit_status='NOT_VERIFIED',issues=[])


def audit_run(root: Path,c: dict,run: dict,protocol: dict,data: Data,dense: dict,duplicates: tuple | dict = ()) -> dict:
    row=basic_row(c,run);directory=root/'runs'/run['id'];row['collected']=(directory/'report.json').is_file()
    try:
        require(run['id'] not in duplicates,'DUPLICATE_RESULT_IDENTITY')
        inspect_marker(directory);report=read(directory/'report.json')
        require(report['run']==run,'RUN_IDENTITY_MISMATCH')
        require(report['condition']=={k:v for k,v in c.items() if k!='runs'},'CONDITION_IDENTITY_MISMATCH')
        require(report['source_commit']==protocol['source_commit'],'RUN_SOURCE_MISMATCH')
        require(report['checkpoint_sha256']==dense['checkpoint_hash'] and report['split_sha256']==dense['split_hash'],'RUN_DENSE_OR_SPLIT_MISMATCH')
        require(report['graph_tensor_sha256']==dense['checkpoint']['graph_tensor_sha256'],'RUN_GRAPH_MISMATCH')
        require(report['test_labels_exposed'] is False,'TEST_LABEL_EXPOSURE_REPORTED')
        cp=torch.load(directory/'pruned.pt',map_location='cpu',weights_only=True)
        require(cp['source_checkpoint_sha256']==dense['checkpoint_hash'],'PRUNED_SOURCE_CHECKPOINT')
        keeps=report['retained_original_ids'];dense_widths=dense['model'].hidden_channel_dims
        require(len(keeps)==2 and cp['original_kept_ids']==keeps,'RETAINED_IDENTITY_METADATA')
        for layer,ids in enumerate(keeps):
            require(ids==sorted(set(ids)) and all(type(i)is int and 0<=i<dense_widths[layer] for i in ids),'INVALID_RETAINED_IDS')
            require(protocol['adaptive']['min_channels']<=len(ids)<=dense_widths[layer],'INVALID_FINAL_WIDTH')
        expected_arch=dense['model'].export_architecture_config();expected_arch.pop('name');expected_arch['hidden_channels']=len(keeps[0]) if len(keeps[0])==len(keeps[1]) else list(map(len,keeps))
        # Dense scalar width and exported nonuniform width list are equivalent only
        # when the submitted model explicitly exports that representation.
        require(cp['model_config']==expected_arch,'PRUNED_ARCHITECTURE')
        model=reconstruct(cp,c['model']);require(model.hidden_channel_dims==report['final_widths']==list(map(len,keeps)),'FINAL_WIDTH_METADATA')
        projected=projected_state(dense['checkpoint']['model_state_dict'],keeps,int(data.num_features),int(data.y.max())+1)
        require(projected.keys()==cp['model_state_dict'].keys(),'PRUNED_TENSOR_KEYS')
        require(all(torch.equal(projected[k],cp['model_state_dict'][k]) for k in projected),'DENSE_TENSOR_PROJECTION_MISMATCH')
        before,after=dense['parameters'],sum(p.numel() for p in model.parameters());achieved=1-after/before
        require(report['parameters_before']==before and report['parameters_after']==after,'PHYSICAL_PARAMETER_COUNTS')
        require(after<=before,'MODEL_GREW')
        require(report['checkpoint_bytes']==(directory/'pruned.pt').stat().st_size,'CHECKPOINT_SIZE')
        require(report['parameter_bytes']==sum(p.numel()*p.element_size() for p in model.parameters()),'PARAMETER_MEMORY')
        close(report['achieved_compression'],achieved,'EXACT_COMPRESSION',COMPRESSION_TOL)
        close(report['signed_target_deviation'],achieved-run['target'],'TARGET_DEVIATION')
        close(report['overshoot'],max(0.,achieved-run['target']),'OVERSHOOT')
        outside=abs(achieved-run['target'])>protocol['adaptive']['tolerance']
        require(report['outside_symmetric_band']==outside,'TARGET_BAND_REPORTING')
        actual=metrics(model,data,dense['split']);compare_metrics(actual,dict(val=report['validation'],test=report['test']),'PRUNED_METRIC')
        report['_dense_validation_accuracy']=dense['metrics']['val']['accuracy']
        ctx=context(dense['checkpoint'],data,report,protocol);behavior=audit_behavior(report,ctx,protocol)
        reason=report['statistics']['terminal_reason']
        require(after<before or (run['method']=='sequential_random' and reason=='initial_target_infeasible'),'UNEXPLAINED_UNPRUNED_MODEL')
        inference=report.get('inference')
        if inference:
            for key in ('inference_time_mean_sec','inference_time_std_sec','inference_time_mean_ms','inference_time_std_ms'):
                require(math.isfinite(inference[key]) and inference[key]>=0,'INVALID_INFERENCE_TIME')
            close(inference['inference_time_mean_ms'],inference['inference_time_mean_sec']*1000,'INFERENCE_UNITS')
            close(inference['inference_time_std_ms'],inference['inference_time_std_sec']*1000,'INFERENCE_STD_UNITS')
            require(inference['inference_warmup_passes']==protocol['benchmark']['warmup_passes'] and inference['inference_timed_passes']==protocol['benchmark']['timed_passes'],'INFERENCE_PASS_COUNTS')
        stats=report['statistics']
        for key in ('scoring_seconds','surgery_seconds'):
            if key in stats:require(math.isfinite(stats[key]) and 0<=stats[key]<=report['search_seconds']+1e-6,'INVALID_COMPONENT_COST')
        warnings=[]
        if outside:warnings.append('TARGET_OUTSIDE_SYMMETRIC_BAND')
        if reason=='initial_target_infeasible':warnings.append('DECLARED_INFEASIBLE_UNPRUNED')
        if stats.get('static_plan',{}).get('fallback'):warnings.append('STATIC_NEAREST_FALLBACK')
        if run['method']=='adaptive_greedy':warnings.append('GREEDY_UNSELECTED_CANDIDATES_NOT_RECORDED')
        if not inference:warnings.append('INFERENCE_NOT_MEASURED')
        if dense['reused']:warnings.append('REUSED_DENSE_ORIGINAL_TRAINING_WALL_NOT_SAVED')
        row.update(verified=True,audit_status='WARNING' if warnings else 'PASS',issues=warnings,
            parameters_before=before,parameters_after=after,achieved_compression=achieved,signed_target_deviation=achieved-run['target'],
            outside_symmetric_band=outside,physical_pruning=after<before,final_widths=json.dumps(list(map(len,keeps))),
            retained_original_ids=json.dumps(keeps,separators=(',',':')),terminal_reason=reason,
            dense_checkpoint_bytes=dense['checkpoint_bytes'],pruned_checkpoint_bytes=report['checkpoint_bytes'],parameter_bytes=report['parameter_bytes'],
            search_seconds=report['search_seconds'],validation_evaluations=report['observed_validation_evaluations'],
            candidate_evaluations=stats.get('candidate_evaluations'),q_entries=stats.get('q_entries'),q_table_sparsity=stats.get('q_table_sparsity'),
            dense_training_seconds=dense['training_seconds'],dense_reused=dense['reused'],
            current_campaign_dense_training_seconds=0. if dense['reused'] else dense['training_seconds'],
            inference_mean_ms=inference['inference_time_mean_ms'] if inference else None,
            inference_std_ms=inference['inference_time_std_ms'] if inference else None,
            dense_inference_mean_ms=None,peak_process_rss_kib=report.get('peak_rss_kib'),
            source_commit=protocol['source_commit'],dense_origin_source_commit=dense['origin_source'],
            dense_checkpoint_sha256=dense['checkpoint_hash'],pruned_checkpoint_sha256=digest(directory/'pruned.pt'),
            dense_tensor_sha256=dense['tensor_hash'],pruned_tensor_sha256=tensor_identity(cp['model_state_dict']),
            graph_tensor_sha256=report['graph_tensor_sha256'],split_sha256=dense['split_hash'],
            output_report_sha256=digest(directory/'report.json'),artifact_integrity='PASS',checkpoint_reload='PASS',tensor_projection='PASS',
            final_metrics_recomputed='PASS',trace_audit='PASS',greedy_argmax_verified=False if run['method']=='adaptive_greedy' else None,
            metric_max_absolute_error=max(abs(actual[sp][mt]-report[{'val':'validation','test':'test'}[sp]][mt]) for sp in ('val','test') for mt in ('accuracy','macro_f1')),
            behavior_details=behavior)
        for split in ('val','test'):
            for metric in ('accuracy','macro_f1'):
                base=dense['metrics'][split][metric];final=actual[split][metric]
                row['dense_'+split+'_'+metric]=base;row['pruned_'+split+'_'+metric]=final;row['delta_'+split+'_'+metric]=final-base
    except FileNotFoundError as exc:
        row.update(audit_status='FAIL',issues=['MISSING_REQUIRED_ARTIFACT'],error=Path(exc.filename).name if exc.filename else 'missing')
    except (ValueError,KeyError,TypeError,RuntimeError,AssertionError,IndexError,ZeroDivisionError) as exc:
        row.update(audit_status='FAIL',issues=[str(exc) or type(exc).__name__],error=type(exc).__name__)
    return row


def parse_accounting(text: str,submission: dict,raw: Path) -> list[dict]:
    """Separate scheduler outcomes from artifact completeness; use bhist + LSF logs."""
    blocks={int(match.group(1)):match.group(2) for match in re.finditer(r'(?:^|\n)Job <(\d+)>(.*?)(?=\nJob <\d+>|\Z)',text,re.S)}
    rows=[]
    for job in submission['jobs']:
        node=job['job']['id'];jobid=job['job_id'];block=blocks.get(jobid,'')
        outpath=raw/'logs'/f'{node}.{jobid}.out';out=outpath.read_text(errors='replace') if outpath.exists() else ''
        if 'Done successfully.' in block:status='DONE'
        elif re.search(r'Exited|exit code|TERM_',block):status='CANCELLED' if re.search(r'TERM_OWNER|TERM_ADMIN|killed by user',block,re.I) else 'FAILED'
        elif re.search(r'Starting|Running',block):status='RUNNING_OR_HISTORY_INCOMPLETE'
        elif 'Submitted' in block:status='PENDING_OR_HISTORY_INCOMPLETE'
        else:status='ACCOUNTING_MISSING'
        times=re.search(r'PEND\s+PSUSP\s+RUN\s+USUSP\s+SSUSP\s+UNKWN\s+TOTAL\s*\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',block)
        cpu=re.search(r'CPU time used is ([\d.]+) seconds',block)
        mem=re.search(r'MAX MEM:\s*([\d.]+)\s*(\w+)',block)
        allocated=re.search(r'Allocated (\d+)\s+Slot',block)
        rows.append(dict(node=node,job_id=jobid,stage=job['job']['stage'],condition=job['job'].get('condition'),status=status,
            bhist_present=bool(block),stdout_success='Successfully completed.' in out,
            stdout_present=outpath.exists(),stderr_present=(raw/'logs'/f'{node}.{jobid}.err').exists(),
            active_wall_seconds=float(times[3]) if times else None,queue_pending_seconds=float(times[1]) if times else None,
            cpu_seconds=float(cpu[1]) if cpu else None,peak_memory=float(mem[1]) if mem else None,peak_memory_unit=mem[2] if mem else None,
            allocated_slots=int(allocated[1]) if allocated else None,requested_slots=job['resources']['slots'],
            requested_wall_minutes=job['resources']['wall_minutes']))
    return rows


def source_audit(raw: Path,source: Path) -> dict:
    """Check pinned source content and review the executed validation-only path."""
    manifest=read(raw/'source-files.json');bad=[name for name,value in manifest.items() if not (source/name).is_file() or digest(source/name)!=value]
    require(not bad,'PINNED_SOURCE_FILE_HASH_MISMATCH')
    own=Path(__file__).resolve().parents[3]
    model_files=[name for name in manifest if name.startswith('src/gnn_pruning/models/') and name.endswith('.py')]
    require(all(digest(own/name)==manifest[name] for name in model_files),'LOCAL_MODEL_CODE_DIFFERS_FROM_PINNED_SOURCE')
    code=(source/'src/gnn_pruning/experiments/complete_campaign.py').read_text()
    search=code[code.index('def search('):code.index('\ndef verify_projection')]
    require(search.index("work.y[split['test']] = -1")<search.index('compute_scores('),'SOURCE_TEST_MASK_ORDER')
    require("candidate.eval()(work).argmax(-1)[split['val']]" in search and "work.y[split['val']]" in search,'SOURCE_VALIDATION_CALLBACK')
    require('compute_scores(model, criterion, work, torch.tensor(split[\'train\']))' in search,'SOURCE_SNIP_TRAIN_ONLY')
    prune=code[code.index('def prune('):code.index('\ndef pilot_gate')]
    require(prune.index('search(dense, data, split, run, protocol)')<prune.index("test=_metrics(final"),'SOURCE_TEST_REPORT_AFTER_SELECTION')
    return dict(status='PASS',pinned_files=len(manifest),local_model_files_checked=len(model_files),
        decision_data_access='test labels masked; train-only SNIP loss; validation-only adaptive callback; test metrics after search',
        inference_verification='final saved checkpoint forward only; no candidate/pruner/policy rerun')


def submission_anchors(root: Path,source: Path,submission: dict,protocol: dict,reference: dict) -> dict:
    """Tie collected records to the locally preserved pre-collection receipts."""
    require(protocol['source_commit']==reference['source_commit'],'REFERENCE_SOURCE_COMMIT')
    require(digest(root/'protocol.json')==reference['protocol_sha256'],'REFERENCE_PROTOCOL_HASH')
    archive=source.parent/('source-'+protocol['source_commit']+'.tar.gz')
    require(digest(archive)==reference['source_archive_sha256'],'REFERENCE_SOURCE_ARCHIVE')
    expected={j['node']:j for j in reference['jobs']}
    require(len(submission['jobs'])==len(expected),'REFERENCE_JOB_COUNT')
    seen=set()
    for item in submission['jobs']:
        node=item['job']['id'];ref=expected[node]
        require(item['status']=='ACCEPTED' and item['accepted'] and item['job_id']==ref['job_id'],'REFERENCE_JOB_ID_ACCEPTANCE')
        require(item['source_commit']==protocol['source_commit'] and item['protocol_sha256']==reference['protocol_sha256'],'REFERENCE_RECEIPT_SOURCE')
        require(item['dependency_job_ids']==ref['dependencies'] and item['resources']==ref['resources'],'REFERENCE_JOB_DEPENDENCIES_RESOURCES')
        require(item['job_id'] not in seen and set(item['dependency_job_ids'])<=seen,'REFERENCE_DAG_ORDER');seen.add(item['job_id'])
        directory=root/'submissions'/node
        require(read(directory/'receipt.json')==item,'RAW_RECEIPT_LEDGER_MISMATCH')
        intent=read(directory/'intent.json');response=read(directory/'response.json')
        require(intent['status']=='INTENT' and intent['created_unix']==item['created_unix'],'RAW_INTENT_MISMATCH')
        require(digest(directory/'launcher.sh')==item['launcher_sha256']==ref['launcher_sha256'],'REFERENCE_LAUNCHER_HASH')
        require(response['returncode']==0 and re.fullmatch(r'\s*Job <'+str(item['job_id'])+r'> is submitted to queue <normal>\.\s*',response['stdout']) is not None,'RAW_ACCEPTANCE_RESPONSE')
    return dict(status='PASS',accepted_receipts=len(seen),independent_reference='pre-collection committed submission.json',source_archive_sha256=reference['source_archive_sha256'])


def write_csv(path: Path | str,rows: list[dict]) -> None:
    """Stable flat tables; lists/maps are encoded as JSON rather than Python repr."""
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def summarize(rows: list[dict]) -> list[dict]:
    groups=defaultdict(list)
    for row in rows:groups[(row['dataset'],row['architecture'],row['method'],row['requested_compression'])].append(row)
    numeric=('achieved_compression','pruned_val_accuracy','pruned_val_macro_f1','pruned_test_accuracy','pruned_test_macro_f1',
             'delta_val_accuracy','delta_val_macro_f1','parameters_after','search_seconds','validation_evaluations','candidate_evaluations','inference_mean_ms')
    summaries=[]
    for key,cells in sorted(groups.items()):
        good=[r for r in cells if r['verified']]
        item=dict(zip(('dataset','architecture','method','requested_compression'),key))
        item.update(n_expected=len(cells),n_collected=sum(r['collected'] for r in cells),n_verified=len(good),
            n_fail=sum(r['audit_status']=='FAIL' for r in cells),n_warning=sum(r['audit_status']=='WARNING' for r in cells),
            training_seeds=json.dumps(sorted(r['dense_training_seed'] for r in cells)),
            n_target_band=sum(not r['outside_symmetric_band'] for r in good),n_physically_pruned=sum(r['physical_pruning'] for r in good),
            stopping_reasons=json.dumps(dict(Counter(r['terminal_reason'] for r in good)),sort_keys=True))
        for name in numeric:
            values=[r[name] for r in good if r.get(name) is not None]
            item[name+'_mean']=float(np.mean(values)) if values else None
            item[name+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summaries.append(item)
    return summaries


def audit(root: Path,source: Path,inventory: list[dict],accounting: str,output: Path,reference: dict | None = None) -> dict:
    """Produce all expected rows even on missing/inconsistent evidence; no repair."""
    root,source,output=map(Path,(root,source,output));output.mkdir(parents=True,exist_ok=False)
    spec=read(source/'research/experiments/polus_complete_pruning_20261031/manifest.json')
    protocol=read(root/'protocol.json');submission=read(root/'submission-result.json')
    require(digest(root/'protocol.json')==submission['protocol_sha256'],'SUBMITTED_PROTOCOL_HASH')
    require(protocol['source_commit']==submission['source_commit'],'SUBMITTED_SOURCE_COMMIT')
    if reference is None:reference=read(Path(__file__).resolve().parents[3]/'research/experiments/polus_complete_pruning_20261031/submission.json')
    anchors=submission_anchors(root,source,submission,protocol,reference)
    provenance=source_audit(root,source)
    transfer=inventory_audit(root,inventory,required_files(spec,submission));write_csv(output/'artifact_manifest.csv',transfer)
    bad_transfer=[r for r in transfer if r['status']!='VERIFIED']
    actual_reports=defaultdict(list);extras=[]
    for path in sorted((root/'runs').glob('*/report.json')):
        try:actual_reports[read(path)['run']['id']].append(path.parent.name)
        except (ValueError,KeyError):extras.append(dict(path=str(path.relative_to(root)),issue='MALFORMED_REPORT'))
    duplicates={key:paths for key,paths in actual_reports.items() if len(paths)>1}
    expected={run['id'] for c in spec['conditions'] for run in c['runs']}
    for name,paths in actual_reports.items():
        if name not in expected:extras.append(dict(path=paths,issue='UNEXPECTED_RESULT_ID',run_id=name))
    for path in (root/'runs').iterdir():
        if path.is_dir() and path.name not in expected:extras.append(dict(path=path.name,issue='UNEXPECTED_RUN_DIRECTORY'))
    graphs={};graph_errors={};graph_manifest=read(root/'inputs/manifest.json')
    require(digest(root/'inputs/manifest.json')==protocol['input_manifest_sha256'],'FROZEN_GRAPH_MANIFEST_HASH')
    for dataset,item in graph_manifest.items():
        try:
            path=root/'inputs'/item['file'];require(digest(path)==item['sha256'],'FROZEN_GRAPH_FILE_HASH')
            tensors=torch.load(path,map_location='cpu',weights_only=True)
            require(tensor_identity(tensors)==item['tensor_sha256'],'FROZEN_GRAPH_TENSOR_HASH')
            data=Data(**tensors)
            require(data.x.ndim==2 and data.x.shape[0]==data.y.numel() and bool(torch.isfinite(data.x).all()),'FROZEN_GRAPH_FEATURES')
            require(data.y.dtype==torch.long and data.y.ndim==1 and sorted(data.y.unique().tolist())==list(range(item['classes'])),'FROZEN_GRAPH_LABELS')
            require(data.edge_index.dtype==torch.long and data.edge_index.ndim==2 and data.edge_index.shape[0]==2,'FROZEN_GRAPH_EDGE_STRUCTURE')
            require(int(data.edge_index.min())>=0 and int(data.edge_index.max())<data.num_nodes,'FROZEN_GRAPH_EDGE_RANGE')
            require((int(data.num_nodes),int(data.num_features),int(data.num_edges))==(item['nodes'],item['features'],item['edges']),'FROZEN_GRAPH_DIMENSIONS')
            graphs[dataset]=data
        except Exception as exc:graph_errors[dataset]=dict(error=type(exc).__name__,issue=str(exc))
    rows=[];dense_rows=[];tensor_groups=defaultdict(list)
    for c in spec['conditions']:
        try:
            dense=dense_audit(root,c,protocol,graphs[c['dataset']],graph_manifest[c['dataset']]['tensor_sha256'])
            tensor_groups[(c['dataset'],c['model'])].append((c['seed'],dense['tensor_hash'],dense['split_hash']))
            dense_rows.append(dict(condition_id=c['id'],dataset=c['dataset'],architecture=c['model'],seed=c['seed'],status='PASS',
                checkpoint_sha256=dense['checkpoint_hash'],tensor_sha256=dense['tensor_hash'],split_sha256=dense['split_hash'],
                parameters=dense['parameters'],reused=dense['reused'],training_seconds=dense['training_seconds'],
                source_commit=protocol['source_commit'],origin_source_commit=dense['origin_source'],metrics=dense['metrics']))
        except Exception as exc:
            dense=None;dense_rows.append(dict(condition_id=c['id'],status='FAIL',issue=str(exc),error=type(exc).__name__))
        for run in c['runs']:
            if dense is None:
                row=basic_row(c,run);row.update(collected=(root/'runs'/run['id']/'report.json').is_file(),issues=['DENSE_INPUT_NOT_VERIFIED'])
            else:row=audit_run(root,c,run,protocol,graphs[c['dataset']],dense,duplicates)
            rows.append(row)
            print(json.dumps(dict(run=run['id'],status=row['audit_status'],issues=row['issues'])),flush=True)
    replication=[]
    for key,values in tensor_groups.items():
        good=len({x[0] for x in values})==len(values)==len({x[1] for x in values})==len({x[2] for x in values})
        replication.append(dict(dataset=key[0],architecture=key[1],dense_seeds=[x[0] for x in values],unique_tensor_and_split_hashes=good))
    jobs=parse_accounting(accounting,submission,root)
    totals=dict(expected_dense=len(spec['conditions']),expected_runs=len(expected),collected_runs=sum(r['collected'] for r in rows),
        verified_runs=sum(r['verified'] for r in rows),audit_status_counts=dict(Counter(r['audit_status'] for r in rows)),
        dense_status_counts=dict(Counter(r['status'] for r in dense_rows)),job_status_counts=dict(Counter(r['status'] for r in jobs)),
        transfer_status_counts=dict(Counter(r['status'] for r in transfer)),duplicate_result_ids=duplicates,extra_artifacts=extras,
        issue_counts=dict(Counter(issue for row in rows for issue in row['issues'])),
        source_commit=protocol['source_commit'],metric_absolute_tolerance=METRIC_TOL,
        compression_absolute_tolerance=COMPRESSION_TOL,trace_float32_absolute_tolerance=1.2e-7,
        source_review=provenance,submission_anchors=anchors,graph_errors=graph_errors,replication=replication,
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),threads=torch.get_num_threads()),
        unrecorded_evidence=['unselected Greedy candidate observations','original fitting wall for reused seed101 checkpoints','dense inference latency',
                            'raw per-pass latency samples; only mean/std were saved','per-run incremental peak RSS; saved RSS is worker-process high water'],
        new_training=False,new_pruning=False,new_timing=False)
    resource_issues=[]
    by_condition={j['condition']:j for j in jobs if j['stage']=='prune'}
    for c in spec['conditions']:
        job=by_condition.get(c['id'],{});wall=job.get('active_wall_seconds')
        measured=sum(r.get('search_seconds',0) for r in rows if r['condition_id']==c['id'] and r['verified'])
        if wall is not None and measured>wall+1:resource_issues.append(dict(condition=c['id'],code='SEARCH_EXCEEDS_JOB_WALL'))
    for job in jobs:
        wall=job.get('active_wall_seconds')
        if wall is not None and wall>job['requested_wall_minutes']*60+1:resource_issues.append(dict(node=job['node'],code='JOB_WALL_EXCEEDS_LIMIT'))
    totals['resource_issues']=resource_issues
    totals['go_for_descriptive_analysis']=(totals['verified_runs']==len(expected) and all(r['status']=='PASS' for r in dense_rows) and
        not bad_transfer and not graph_errors and not resource_issues and not duplicates and not extras and all(r['status']=='DONE' and r['stdout_success'] for r in jobs) and all(r['unique_tensor_and_split_hashes'] for r in replication))
    save(output/'audit.json',dict(totals=totals,runs=rows,dense=dense_rows,jobs=jobs))
    write_csv(output/'audit_runs.csv',rows);write_csv(output/'verified_results.csv',rows)
    write_csv(output/'summary.csv',summarize(rows));write_csv(output/'dense_audit.csv',dense_rows);write_csv(output/'job_status.csv',jobs)
    save(output/'unresolved_issues.json',dict(counts=totals['issue_counts'],unrecorded_evidence=totals['unrecorded_evidence'],extra_artifacts=extras,duplicate_result_ids=duplicates))
    print(json.dumps(totals),flush=True)
    return totals


def audit_safe(root: Path,source: Path,inventory: list[dict],accounting: str,output: Path,reference: dict | None = None) -> dict:
    """Keep essential input failures visible as NOT VERIFIED; never repair data."""
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new audit output directory; preserve earlier audit evidence.')
    try:
        return audit(root,source,inventory,accounting,output,reference)
    except Exception as exc:
        output.mkdir(parents=True,exist_ok=True)
        try:spec=read(Path(source)/'research/experiments/polus_complete_pruning_20261031/manifest.json')
        except Exception:spec={'conditions':[]}
        rows=[]
        for condition in spec['conditions']:
            for run in condition['runs']:
                row=basic_row(condition,run);row['collected']=(Path(root)/'runs'/run['id']/'report.json').is_file()
                row['issues']=['GLOBAL_ESSENTIAL_EVIDENCE_NOT_VERIFIED'];rows.append(row)
        reason=Path(exc.filename).name if isinstance(exc,FileNotFoundError) and exc.filename else str(exc)
        totals=dict(expected_runs=len(rows) if rows else None,collected_runs=sum(r['collected'] for r in rows),verified_runs=0,
            audit_status_counts={'NOT_VERIFIED':len(rows)},go_for_descriptive_analysis=False,
            global_error=dict(severity='CRITICAL',error=type(exc).__name__,issue=reason),new_training=False,new_pruning=False,new_timing=False)
        save(output/'audit.json',dict(totals=totals,runs=rows,dense=[],jobs=[]))
        save(output/'unresolved_issues.json',totals['global_error'])
        write_csv(output/'audit_runs.csv',rows);write_csv(output/'verified_results.csv',rows)
        write_csv(output/'summary.csv',summarize(rows))
        return totals


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','inventory','accounting','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();inv=read(a.inventory);text=a.accounting.read_text()
    torch.set_num_threads(2);audit_safe(a.root,a.source,inv,text,a.output)
