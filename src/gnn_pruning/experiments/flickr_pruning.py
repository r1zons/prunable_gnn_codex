"""Frozen Flickr structural-budget V5 campaign; never trains or reads test labels."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import resource
import time

import torch
from sklearn.metrics import f1_score

from gnn_pruning.models import build_model
from gnn_pruning.pruning.budget_v5 import Budget, METHODS, VERSION, run, static_plan
from gnn_pruning.pruning.static_v2 import apply_static_plan
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2
from .flickr_dense import digest, environment, load_inputs, metrics, sha, tensors_hash, verify_run, write
from .frozen_validation import FrozenAccuracy


ARCHITECTURES = ('flickr-graphsage-l3-h128-s101','flickr-graphsage-l4-h128-s101',
                 'flickr-graphsage-l4-h256-s101','flickr-gcn-l2-h256-s101')


def matrix(protocol):
    """Fixed56, separately named implementations and three seed identities."""
    if (protocol['architectures'] != list(ARCHITECTURES) or protocol['methods'] != list(METHODS)
        or protocol['targets'] != [.3,.5] or protocol['policy_seed'] != 101
        or protocol['dense_seed'] != 101 or protocol['split_seed'] != 101):
        raise ValueError('Authorized56-condition matrix differs.')
    return [dict(id=f'{arch}-{method}-t{round(target*100)}-budget-v5',architecture=arch,
                 method=method,implementation=method+'_'+VERSION,target=target,
                 dense_seed=101,split_seed=101,policy_seed=101)
            for arch in ARCHITECTURES for target in protocol['targets'] for method in METHODS]


def frozen(root, protocol, arch):
    """Strict original Dense reload and graph/split pin verification, without fitting."""
    item=protocol['dense'][arch];directory=root/'dense'/arch
    if sha(root/'inputs/manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Input manifest changed.')
    data,split,manifest=load_inputs(root/'inputs')
    report=verify_run(directory,item['condition'],root/'inputs')
    if sha(directory/'checkpoint.pt') != item['checkpoint_sha256'] or report['tensor_sha256'] != item['tensor_sha256']:
        raise ValueError('Exact archived Dense checkpoint identity differs.')
    config=json.loads((directory/'resolved_config.json').read_text())
    if config['architecture'] != item['model_config'] or config['seed'] != protocol['dense_seed']:
        raise ValueError('Dense architecture/seed differs.')
    if report['split_sha256'] != protocol['split_sha256'] or report['metrics']['val'] != item['validation']:
        raise ValueError('Dense split/recorded metrics differ.')
    cp=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=build_model(item['condition']['model'],**config['architecture']).eval()
    model.load_state_dict(cp['model_state_dict'],strict=True)
    return model,data,split,manifest,report


def per_class(model,data,split):
    """Validation labels only; all classes explicitly retained in F1 reporting."""
    with torch.no_grad():
        predictions=model.eval()(data)[split['val']].argmax(-1).numpy()
    return f1_score(data.y[split['val']].numpy(),predictions,
                    labels=list(range(model.out_channels)),average=None,zero_division=0).tolist()


def verify_projection(dense, final, kept):
    projected=dense
    for layer,ids in enumerate(kept):
        if len(ids)<projected.convs[layer].out_channels:
            projected=prune_hidden_channels_local_v2(projected,layer,ids,4)
    if tensors_hash(projected.state_dict()) != tensors_hash(final.state_dict()):
        raise ValueError('Retained original masks do not project to final tensors.')


def execute(root, protocol, condition, technical=False, execution_resources=None):
    """One independent run; completed markers require integrity and physical pruning."""
    if condition not in matrix(protocol):
        raise ValueError('Unknown run identity.')
    output=root/('smoke' if technical else 'runs')/condition['id']
    output.mkdir(parents=True,exist_ok=False)
    try:
        threads = 4 if execution_resources is None else execution_resources['threads']
        if threads not in (4, 12):
            raise ValueError('Unsupported Flickr execution thread count.')
        if execution_resources is not None and execution_resources.get('interop_threads') is not None:
            torch.set_num_interop_threads(execution_resources['interop_threads'])
        torch.set_num_threads(threads)
        dense,data,split,manifest,original=frozen(root,protocol,condition['architecture'])
        dense_hash=tensors_hash(dense.state_dict())
        budget=Budget(dense,condition['target'],**protocol['budget'])
        write(output/'environment.json',environment())
        write(output/'resolved_config.json',dict(condition=condition,protocol_sha256=digest(protocol),
                                               budget_horizon=budget.horizon,technical=technical,
                                               execution_resources=execution_resources))
        calls=0
        forward_times=[]
        def forward_accuracy(model):
            nonlocal calls
            if not bool((data.y[split['test']]==-1).all()):
                raise ValueError('Test labels exposed.')
            start=time.perf_counter()
            with torch.no_grad():
                pred=model.eval()(data)[split['val']].argmax(-1)
                value=float((pred==data.y[split['val']]).float().mean())
            forward_times.append(time.perf_counter()-start);calls+=1
            return value
        evaluate=FrozenAccuracy(forward_accuracy,manifest['training_tensor_sha256']+':'+manifest['split_sha256'])
        started=time.perf_counter()
        method=condition['method'];training=[];qtable=[];decisions=[]
        if method.startswith('static_'):
            criterion=method.removeprefix('static_')
            p=static_plan(dense,criterion,budget,condition['policy_seed'],data,torch.tensor(split['train']))
            final=apply_static_plan(dense,p);kept=p.kept_indices
            trace=[];widths=list(budget.original);ids=[list(range(w)) for w in widths]
            for layer,selected in enumerate(kept):
                if len(selected)==widths[layer]:continue
                before_widths=list(widths);deleted=widths[layer]-len(selected)
                widths[layer]=len(selected);ids[layer]=list(selected)
                trace.append(dict(action=f'static_prune:{layer}:{deleted}',layer=layer,deletion_size=deleted,
                       widths_before=before_widths,widths_after=list(widths),original_kept_ids=[list(x) for x in ids],
                       sparsity=1-budget.count(widths)/budget.p0,intermediate_validation_recorded=False))
            stats=dict(version=VERSION,complete=True,terminal_reason='closest_feasible' if p.fallback else 'target_band',
                       candidate_evaluations=0,validation_evaluations=0,static_plan=asdict(p),
                       scoring_gradient_forwards=int(criterion=='snip'),horizon=budget.horizon)
        else:
            settings=dict(protocol['q_learning'])
            if technical:settings['episodes']=protocol['technical_q_episodes']
            result=run(dense,evaluate,method,condition['target'],seed=condition['policy_seed'],
                       budget=budget,max_deployment_steps=2 if technical else None,**settings)
            final,kept,trace,stats=result.model,result.kept_indices,result.trace,result.statistics
            training=result.training_trace;decisions=result.decisions
            qtable=[dict(state=s,actions=v) for s,v in result.q_table.items()]
        search_seconds=time.perf_counter()-started
        if evaluate.requests != stats['validation_evaluations']:
            raise ValueError('Evaluation accounting differs.')
        if calls!=evaluate.forward_calls:
            raise ValueError('Physical forward accounting differs.')
        stats['validation_evaluation_requests']=stats['validation_evaluations']
        stats['validation_evaluations']=calls
        stats['evaluation_cache_hits']=evaluate.requests-calls
        stats['candidate_evaluation_requests']=stats['candidate_evaluations']
        if method in ('adaptive_greedy','lookahead_depth2'):
            stats['candidate_evaluations']=max(0,calls-1)
            offset=1
            for d in decisions:
                n=d['candidate_evaluations'];events=evaluate.events[offset:offset+n];offset+=n
                d['candidate_evaluation_requests']=n
                d['candidate_evaluations']=sum(not event['reused'] for event in events)
                d['exact_model_cache_hits']=sum(event['reused'] for event in events)
        else:
            for i,entry in enumerate(training):
                entry['validation_evaluation_reused']=evaluate.events[1+i]['reused']
            for i,d in enumerate(decisions):
                d['validation_evaluation_reused']=evaluate.events[1+len(training)+i]['reused']
        if not technical and not stats['complete']:
            raise ValueError('Technical prefix cannot masquerade as a completed experiment.')
        verify_projection(dense,final,kept)
        if tensors_hash(dense.state_dict()) != dense_hash:
            raise ValueError('Dense input mutated.')
        before=budget.p0;after=sum(p.numel() for p in final.parameters())
        if after >= before or after != budget.count(tuple(map(len,kept))):
            raise ValueError('Physical reduction/count gate failed.')
        cfg=final.export_architecture_config();cfg.pop('name')
        cp=dict(model_name=protocol['dense'][condition['architecture']]['condition']['model'],
                model_config=cfg,model_state_dict=final.state_dict(),original_kept_ids=kept,
                dense_checkpoint_sha256=original['checkpoint_sha256'],source_commit=protocol['source_commit'],
                protocol_sha256=digest(protocol),condition=condition)
        torch.save(cp,output/'checkpoint.pt')
        loaded=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True)
        restored=build_model(loaded['model_name'],**loaded['model_config']).eval()
        restored.load_state_dict(loaded['model_state_dict'],strict=True)
        if tensors_hash(restored.state_dict()) != tensors_hash(final.state_dict()):
            raise ValueError('Strict final reload failed.')
        dense_metrics=metrics(dense,data,split)['val'];final_metrics=metrics(restored,data,split)['val']
        for name in ('accuracy','macro_f1','loss'):
            if abs(dense_metrics[name]-original['metrics']['val'][name]) > (5e-5 if name=='loss' else 1e-12):
                raise ValueError('Frozen Dense metrics do not reproduce.')
        achieved=1-after/before
        for name,value in [('deployment',trace),('training',training),('decisions',decisions),('q_table',qtable)]:
            write(output/(name+'.json'),value)
        write(output/'evaluation_cache.json',evaluate.events)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report=dict(condition=condition,technical=technical,source_commit=protocol['source_commit'],
             protocol_sha256=digest(protocol),dense_checkpoint_sha256=original['checkpoint_sha256'],
             split_sha256=manifest['split_sha256'],graph_tensor_sha256=manifest['training_tensor_sha256'],
             parameters_before=before,parameters_after=after,achieved_compression=achieved,
             signed_target_deviation=achieved-condition['target'],undershoot=max(0.,condition['target']-achieved),
             overshoot=max(0.,achieved-condition['target']),outside_band=not budget.goal(tuple(map(len,kept))) or budget.fallback,
             band_feasible=not budget.fallback,final_widths=list(map(len,kept)),retained_original_ids=kept,
             structural_actions=(sum(len(ids)<w for ids,w in zip(kept,budget.original)) if method.startswith('static_') else len(trace)),
             dense_validation=dense_metrics,validation=final_metrics,
             delta_accuracy=final_metrics['accuracy']-dense_metrics['accuracy'],
             delta_macro_f1=final_metrics['macro_f1']-dense_metrics['macro_f1'],
             validation_per_class_f1=per_class(restored,data,split),statistics=stats,
             search_seconds=search_seconds,observed_validation_evaluations=calls,
             validation_evaluation_requests=evaluate.requests,
             evaluation_reuse='exact tensors/architecture and frozen input, within this run only',
             reporting_forward_evaluations=3,
             validation_forward_seconds=forward_times,actual_host=platform.node(),threads=threads,
             interop_threads=torch.get_num_interop_threads(),execution_resources=execution_resources,
             peak_process_rss_bytes=int(peak if platform.system()=='Darwin' else peak*1024),
             checkpoint_sha256=sha(output/'checkpoint.pt'),tensor_sha256=tensors_hash(final.state_dict()),
             strict_reload=True,retained_projection=True,test_labels_accessed=False)
        write(output/'report.json',report)
        write(output/'completed.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as exc:
        write(output/'failed.json',dict(error=repr(exc),condition=condition,technical=technical))
        raise


def smoke(root,protocol,arch):
    """Seven native checks at50%; Q three full episodes, deployment prefix2."""
    results=[]
    for condition in matrix(protocol):
        if condition['architecture']==arch and condition['target']==.5:
            r=execute(root,protocol,condition,technical=True)
            results.append(dict(condition=condition,search_seconds=r['search_seconds'],
                           validation_evaluations=r['observed_validation_evaluations'],
                           peak_process_rss_bytes=r['peak_process_rss_bytes'],statistics=r['statistics'],
                           maximum_forward_seconds=max(r['validation_forward_seconds'],default=0.)))
            print(json.dumps(results[-1]),flush=True)
    write(root/('smoke-'+arch+'.json'),results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--condition');parser.add_argument('--smoke-architecture',choices=ARCHITECTURES)
    args=parser.parse_args();p=json.loads(args.protocol.read_text());torch.set_num_threads(4)
    if args.smoke_architecture:smoke(args.root,p,args.smoke_architecture)
    else:execute(args.root,p,next(c for c in matrix(p) if c['id']==args.condition))
