"""Isolated CPU resource benchmark; calls unchanged budget-V5 scientific engines.

No dense fitting, test-label evaluation, CUDA, concurrent Q policies or continuation.
The scientific protocol remains30 episodes; exactly3 are executed as a technical
profile with the historical two-action deployment prefix, never research completion.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ARCH = 'flickr-graphsage-l4-h256-s101'
THREADS = (4, 8, 12, 20)
TARGETS = (.3, .5)
MODULE = 'gnn_pruning.experiments.flickr_thread_scaling'


def store(path, value):
    """Exclusive creation: never replace a prior attempt/evidence file."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def topology(sysroot=Path('/sys/devices/system/cpu'), affinity=None):
    """Read physical core/SMT/NUMA identities visible in the actual allocation."""
    affinity = sorted(os.sched_getaffinity(0)) if affinity is None else list(affinity)
    rows = []
    for cpu in affinity:
        path = sysroot / f'cpu{cpu}'
        def read(name):
            file = path / 'topology' / name
            return file.read_text().strip() if file.exists() else None
        nodes = sorted(p.name for p in path.glob('node[0-9]*'))
        rows.append(dict(cpu=cpu, socket=read('physical_package_id'), core=read('core_id'),
                         siblings=read('thread_siblings_list'), numa=nodes[0] if nodes else 'unknown'))
    return dict(host=socket.gethostname(), platform=platform.platform(), logical_cpus=os.cpu_count(),
                affinity=affinity, cpu_topology=rows,
                physical_cores_visible=len({(r['socket'], r['core']) for r in rows if r['core'] is not None}),
                lsf={k:v for k,v in os.environ.items() if k.startswith(('LSB_', 'LSF_'))},
                proc_status=Path('/proc/self/status').read_text() if Path('/proc/self/status').exists() else None)


def physical_affinity(rows, count=20):
    """Choose one permitted SMT sibling per physical core, round-robin over NUMA."""
    cores = {}
    for row in rows:
        if row['core'] is None or row['socket'] is None:
            raise ValueError('Physical core identity missing; refusing SMT-as-core assumptions.')
        cores.setdefault((row['socket'], row['core']), row)
    groups = defaultdict(list)
    for row in cores.values():
        groups[row['numa']].append(row['cpu'])
    for values in groups.values():
        values.sort()
    selected = []
    while any(groups.values()):
        for node in sorted(groups):
            if groups[node]:
                selected.append(groups[node].pop(0))
    if len(selected) < count:
        raise ValueError('Allocation exposes fewer than20 physical cores.')
    return selected[:count]


def thread_affinity():
    """Observe each live native thread's actual allowed CPUs, without changing it."""
    rows=[]
    for entry in Path('/proc/self/task').iterdir():
        try:rows.append(dict(tid=int(entry.name),cpus=sorted(os.sched_getaffinity(int(entry.name)))))
        except ProcessLookupError:continue
    return rows


def capture_topology(output):
    """Collect metadata inside an LSF job, without model computation."""
    output.mkdir(parents=True, exist_ok=False)
    store(output/'topology.json', topology())
    for name, args in [('lscpu', ['lscpu']), ('lscpu-extended', ['lscpu','-e']),
                       ('numa', ['numactl','--hardware']), ('allocation', ['bjobs','-l',os.environ.get('LSB_JOBID','0')])]:
        try:
            p = subprocess.run(args, text=True, capture_output=True, timeout=15)
            value = dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            value = dict(command=args, error=str(exc))
        store(output/(name+'.json'), value)
    store(output/'completed.json', dict(status='METADATA_COMPLETE', host=socket.gethostname()))


@contextmanager
def observe_engine(engine):
    """Timing-only wrappers; original inputs/returns and RNG consumption preserved.

    Categories are non-overlapping except inclusive step/cache totals, labelled
    explicitly. No profiler hook or observer is inserted into numerical kernels.
    """
    from types import SimpleNamespace
    observer = SimpleNamespace(seconds=defaultdict(float), counts=defaultdict(int),
                               episodes=[], initializations=0, episode_start=None)
    def timed(label, function):
        def call(*args, **kwargs):
            start=time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                observer.seconds[label]+=time.perf_counter()-start
                observer.counts[label]+=1
        return call
    original_init=engine.BudgetEnv.__init__
    def initialize(self, *args, **kwargs):
        number=observer.initializations
        observer.initializations+=1
        if number:
            observer.episode_start=time.perf_counter()
        return original_init(self, *args, **kwargs)
    original_update=engine.update_q
    def update(*args, **kwargs):
        value=timed('q_update',original_update)(*args,**kwargs)
        terminal=args[6] if len(args)>6 else kwargs['terminal']
        if terminal:
            observer.episodes.append(dict(episode=observer.initializations-2,
                         seconds=time.perf_counter()-observer.episode_start))
        return value
    with ExitStack() as stack:
        for name,label in [('prune_hidden_channels_local_v2','surgery'), ('compute_scores','magnitude_scores'),
                           ('channel_rankings','channel_ranking'), ('_clone','reset_clone')]:
            stack.enter_context(patch.object(engine,name,timed(label,getattr(engine,name))))
        for name,label in [('legal_actions','legal_action_planning'), ('state','state_encoding'), ('step','step_inclusive')]:
            stack.enter_context(patch.object(engine.BudgetEnv,name,timed(label,getattr(engine.BudgetEnv,name))))
        stack.enter_context(patch.object(engine.BudgetEnv,'__init__',initialize))
        stack.enter_context(patch.object(engine,'update_q',update))
        yield observer


def forecast(seconds):
    """Unchanged predeclared C4 Q forecast:31/3 times,1.75 safety,+180seconds."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive measured search time required.')
    safe=seconds*(31/3)*1.75+180
    return dict(observed_search_seconds=seconds, multiplier=31/3, time_safety=1.75,
                fixed_overhead_seconds=180, safe_seconds=safe, safe_minutes=safe/60,
                acceptance_minutes=170, absolute_lsf_minutes=180,
                passes_acceptance=safe<=170*60, passes_absolute_limit=safe<=180*60)


def thread_environment(count, cpus):
    """One process, one policy; all numeric pools bounded, no nested OpenMP teams."""
    if count not in THREADS or len(cpus)<20:
        raise ValueError('Expected declared thread count and fixed20-core placement.')
    return dict(OMP_NUM_THREADS=str(count), OPENBLAS_NUM_THREADS=str(count), MKL_NUM_THREADS=str(count),
                NUMEXPR_NUM_THREADS=str(count), OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1',
                OMP_PROC_BIND='FALSE', OMP_PLACES='threads',
                CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')


def forward_profile(model, data, warmup=2, repeats=7):
    """Actual uncached full-batch CPU forwards; validation extraction after forward."""
    import torch
    for _ in range(warmup):
        with torch.no_grad():
            model.eval()(data)
    start_cpu=time.process_time();samples=[]
    for _ in range(repeats):
        started=time.perf_counter()
        with torch.no_grad():
            model.eval()(data)
        samples.append(time.perf_counter()-started)
    elapsed=sum(samples);cpu=time.process_time()-start_cpu
    return dict(warmup=warmup,repeats=repeats,actual_forward_calls=warmup+repeats,
                cache_used=False,samples_seconds=samples,median_seconds=statistics.median(samples),
                mean_seconds=statistics.mean(samples),stdev_seconds=statistics.stdev(samples),
                minimum_seconds=min(samples),maximum_seconds=max(samples),
                process_cpu_seconds=cpu,mean_cpu_utilization_percent=100*cpu/elapsed)


def q_profile(model, data, split, manifest, protocol, target, instrument=True):
    """Run precisely the old three-episode/two-deployment-action technical call."""
    import torch
    from contextlib import nullcontext
    from gnn_pruning.pruning import budget_v5 as engine
    from .frozen_validation import FrozenAccuracy
    forward_times=[];evaluation_times=[]
    if protocol['q_learning']['episodes']!=30 or protocol['technical_q_episodes']!=3:
        raise ValueError('Scientific30 / technical3 identity changed.')
    def real_forward(candidate):
        if not bool((data.y[split['test']]==-1).all()):
            raise ValueError('Held-out labels exposed.')
        start=time.perf_counter()
        with torch.no_grad():
            pred=candidate.eval()(data)[split['val']].argmax(-1)
            accuracy=float((pred==data.y[split['val']]).float().mean())
        forward_times.append(time.perf_counter()-start)
        return accuracy
    cache=FrozenAccuracy(real_forward,manifest['training_tensor_sha256']+':'+manifest['split_sha256'])
    def evaluate(candidate):
        start=time.perf_counter()
        try:return cache(candidate)
        finally:evaluation_times.append(time.perf_counter()-start)
    budget=engine.Budget(model,target,**protocol['budget'])
    settings=dict(protocol['q_learning']);settings['episodes']=3
    start_cpu=time.process_time();started=time.perf_counter()
    context=observe_engine(engine) if instrument else nullcontext(None)
    with context as observer:
        result=engine.run(model,evaluate,'tabular_q',target,seed=protocol['policy_seed'],
                          budget=budget,max_deployment_steps=2,**settings)
    elapsed=time.perf_counter()-started;cpu=time.process_time()-start_cpu
    if cache.requests!=result.statistics['validation_evaluations'] or cache.forward_calls!=len(forward_times):
        raise ValueError('Evaluation accounting failed.')
    for episode in range(3):
        entries=[x for x in result.training_trace if x['episode']==episode]
        if not entries or entries[-1]['terminal_reason']!='target_band':
            raise ValueError('Three full training episodes did not reach their declared band.')
    for i,row in enumerate(result.training_trace):
        row['validation_evaluation_reused']=cache.events[1+i]['reused']
    for i,row in enumerate(result.decisions):
        row['validation_evaluation_reused']=cache.events[1+len(result.training_trace)+i]['reused']
    seconds=dict(observer.seconds) if observer else {}
    numeric_eval=sum(forward_times);all_eval=sum(evaluation_times)
    categories=['surgery','magnitude_scores','channel_ranking','reset_clone','legal_action_planning','state_encoding','q_update']
    categorized=sum(seconds.get(k,0.) for k in categories)+all_eval
    timing=dict(search_seconds=elapsed,native_engine_search_seconds=result.statistics['search_seconds'],
                process_cpu_seconds=cpu,mean_cpu_utilization_percent=100*cpu/elapsed,
                evaluation_seconds=all_eval,actual_forward_seconds=numeric_eval,
                cache_hash_lookup_overhead_seconds=max(0.,all_eval-numeric_eval),
                observed_components_seconds=seconds, component_calls=dict(observer.counts) if observer else {},
                episodes=observer.episodes if observer else [],
                residual_policy_trace_and_observer_seconds=elapsed-categorized,
                accounting_note='step_inclusive overlaps components; never sum it. Episode intervals end after terminal Q update, before trace append. Residual includes selection, Python traces and instrumentation overhead.')
    return result,cache,timing


def worker(root, inputs, protocol_path, config_path, threads, target, cpus):
    """Fresh process, frozen inputs, benchmark only; no training or test labels."""
    os.sched_setaffinity(0,cpus)  # before torch/BLAS import
    import torch
    from threadpoolctl import threadpool_info
    from .flickr_dense import sha,tensors_hash,metrics
    from .flickr_pruning import frozen,verify_projection
    from gnn_pruning.models import build_model
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    protocol=json.loads(protocol_path.read_text());config=json.loads(config_path.read_text())
    if threads not in THREADS or target not in TARGETS:
        raise ValueError('Unknown technical condition.')
    output=root/'runs'/f't{threads}-target{round(target*100)}';output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        if len({(r['socket'],r['core']) for r in topology()['cpu_topology']})!=20:
            raise ValueError('Fixed physical20-core affinity failed.')
        dense,data,split,manifest,original=frozen(inputs,protocol,ARCH)
        if any(p.device.type!='cpu' for p in dense.parameters()) or data.x.device.type!='cpu':
            raise ValueError('Only CPU tensors permitted.')
        initial=tensors_hash(dense.state_dict())
        store(output/'environment.json',dict(topology=topology(),torch=torch.__version__,
              intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
              environment={k:os.environ.get(k) for k in thread_environment(threads,cpus)},
              threadpools_before=threadpool_info(),torch_parallel_info=torch.__config__.parallel_info()))
        profile={'dense':dict(**forward_profile(dense,data),widths=dense.hidden_channel_dims,
                              parameters=sum(p.numel() for p in dense.parameters()))}
        pin=config['representative_pruned'];checkpoint=inputs/'smoke'/pin['id']/'checkpoint.pt'
        if sha(checkpoint)!=pin['checkpoint_sha256']:
            raise ValueError('Representative pruned checkpoint changed.')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        pruned=build_model(cp['model_name'],**cp['model_config']).eval()
        pruned.load_state_dict(cp['model_state_dict'],strict=True)
        if tensors_hash(pruned.state_dict())!=pin['tensor_sha256']:
            raise ValueError('Representative pruned tensor identity differs.')
        verify_projection(dense,pruned,cp['original_kept_ids'])
        profile['pruned']=dict(**forward_profile(pruned,data),widths=pruned.hidden_channel_dims,
                              parameters=sum(p.numel() for p in pruned.parameters()),
                              tensor_sha256=pin['tensor_sha256'])
        del pruned
        result,cache,timing=q_profile(dense,data,split,manifest,protocol,target)
        verify_projection(dense,result.model,result.kept_indices)
        if tensors_hash(dense.state_dict())!=initial:
            raise ValueError('Dense input mutated.')
        cfg=result.model.export_architecture_config();cfg.pop('name')
        torch.save(dict(model_name='graphsage',model_config=cfg,model_state_dict=result.model.state_dict(),
                        original_kept_ids=result.kept_indices),output/'checkpoint.pt')
        cp=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True)
        restored=build_model('graphsage',**cp['model_config']).eval()
        restored.load_state_dict(cp['model_state_dict'],strict=True)
        if tensors_hash(restored.state_dict())!=tensors_hash(result.model.state_dict()):
            raise ValueError('Strict reload changed tensors.')
        for name,value in [('training',result.training_trace),('deployment',result.trace),
                 ('decisions',result.decisions),('q_table',[dict(state=s,actions=v) for s,v in result.q_table.items()]),
                 ('evaluation_cache',cache.events),('forward_profile',profile),('q_timing',timing)]:
            store(output/(name+'.json'),value)
        pools=threadpool_info()
        if any(p.get('num_threads',0)>threads for p in pools):
            raise ValueError('A numeric thread pool exceeds declared setting.')
        validation=metrics(restored,data,split)['val']
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report=dict(technical=True,scientific_complete=False,threads=threads,target=target,
            source_commit=config['benchmark_source_commit'],scientific_source_commit=protocol['source_commit'],
            protocol_sha256=sha(protocol_path),dense_checkpoint_sha256=original['checkpoint_sha256'],
            dense_tensor_sha256=initial,split_sha256=manifest['split_sha256'],
            graph_tensor_sha256=manifest['training_tensor_sha256'],test_labels_accessed=False,
            actual_host=socket.gethostname(),affinity=sorted(os.sched_getaffinity(0)),
            interop=torch.get_num_interop_threads(),threadpools_after=pools,thread_affinity=thread_affinity(),peak_rss_bytes=peak,
            worker_total_seconds=time.perf_counter()-started,search_seconds=timing['search_seconds'],
            statistics=result.statistics,validation_evaluation_requests=cache.requests,
            actual_forward_calls=cache.forward_calls,cache_hits=cache.requests-cache.forward_calls,
            parameters_before=sum(p.numel() for p in dense.parameters()),
            parameters_after=sum(p.numel() for p in restored.parameters()),
            achieved_compression=1-sum(p.numel() for p in restored.parameters())/sum(p.numel() for p in dense.parameters()),
            final_widths=restored.hidden_channel_dims,retained_original_ids=result.kept_indices,
            tensor_sha256=tensors_hash(restored.state_dict()),checkpoint_sha256=sha(output/'checkpoint.pt'),
            validation=validation,strict_reload=True,retained_projection=True,
            scientific_episodes=30,technical_episodes=3,deployment_prefix_steps=2,
            forecast=forecast(timing['search_seconds']))
        store(output/'report.json',report)
        store(output/'completed.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    except Exception as exc:
        store(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise


def decision_differences(first, second):
    """Numerical comparisons only at an identical recorded state/action prefix."""
    matched=0;maximum={k:0. for k in ['reward','q_before','q_after','validation_accuracy']}
    divergence=None
    for i,(a,b) in enumerate(zip(first,second)):
        if a.get('state')!=b.get('state') or a.get('selected_action',a.get('action'))!=b.get('selected_action',b.get('action')):
            divergence=dict(index=i,reference_action=a.get('selected_action',a.get('action')),
                            candidate_action=b.get('selected_action',b.get('action')),
                            episode=a.get('episode'))
            break
        matched+=1
        for key in maximum:
            if key in a and key in b:maximum[key]=max(maximum[key],abs(a[key]-b[key]))
    if divergence is None and len(first)!=len(second):divergence=dict(index=matched,reason='length differs')
    return dict(matched_state_action_prefix=matched,first_divergence=divergence,max_absolute_difference_on_matched_prefix=maximum)


def q_differences(first,second):
    """Compare Q-values only for identical concrete state/action keys."""
    flatten=lambda rows:{(json.dumps(row['state'],sort_keys=True),action):value
                         for row in rows for action,value in row['actions'].items()}
    a,b=flatten(first),flatten(second);keys=a.keys()&b.keys()
    return dict(matched_entries=len(keys),reference_only=len(a.keys()-b.keys()),candidate_only=len(b.keys()-a.keys()),
                max_absolute_difference=max((abs(a[k]-b[k]) for k in keys),default=None))


def compare(root, reference, candidate):
    """Exact structural comparison plus recorded numeric drift, no tolerance masking."""
    from .flickr_dense import tensors_hash
    import torch
    a=json.loads((reference/'report.json').read_text());b=json.loads((candidate/'report.json').read_text())
    fields=['retained_original_ids','final_widths','parameters_before','parameters_after','achieved_compression',
            'validation_evaluation_requests','actual_forward_calls','cache_hits','tensor_sha256']
    equality={k:a[k]==b[k] for k in fields}
    traces={}
    for name in ['training','deployment','decisions','q_table','evaluation_cache']:
        x=json.loads((reference/(name+'.json')).read_text());y=json.loads((candidate/(name+'.json')).read_text())
        traces[name]=x==y
    def selected(path,name):
        return [r.get('selected_action',r.get('action')) for r in json.loads((path/(name+'.json')).read_text())]
    actions_equal=all(selected(reference,n)==selected(candidate,n) for n in ['training','deployment','decisions'])
    accuracy_diff=b['validation']['accuracy']-a['validation']['accuracy']
    f1_diff=b['validation']['macro_f1']-a['validation']['macro_f1']
    x=torch.load(reference/'checkpoint.pt',map_location='cpu',weights_only=True)['model_state_dict']
    y=torch.load(candidate/'checkpoint.pt',map_location='cpu',weights_only=True)['model_state_dict']
    tensor_equal=tensors_hash(x)==tensors_hash(y)
    return dict(reference_threads=a['threads'],candidate_threads=b['threads'],target=b['target'],
        exact_fields=equality,exact_traces=traces,selected_actions_equal=actions_equal,
        training_differences=decision_differences(json.loads((reference/'training.json').read_text()),json.loads((candidate/'training.json').read_text())),
        q_value_differences=q_differences(json.loads((reference/'q_table.json').read_text()),json.loads((candidate/'q_table.json').read_text())),
        final_tensors_exact=tensor_equal,validation_accuracy_difference=accuracy_diff,
        validation_macro_f1_difference=f1_diff,validation_loss_difference=b['validation']['loss']-a['validation']['loss'],
        scientific_equivalence=all(equality.values()) and all(traces.values()) and tensor_equal and accuracy_diff==0 and f1_diff==0)


def orchestrate(root, inputs, protocol, config_path):
    """One C4 allocation, eight serial fresh Python workers; no parallel search."""
    config=json.loads(config_path.read_text())
    if socket.gethostname()!=config['host'] or int(os.environ.get('LSB_DJOB_NUMPROC','0'))<20:
        raise ValueError('Verified C4 single-host20-slot allocation required.')
    capture_topology(root/'topology-c4')
    cpus=physical_affinity(topology()['cpu_topology'])
    store(root/'placement.json',dict(fixed20_physical_cpu_ids=cpus,policy='one SMT sibling/core; same20-core affinity for all cases; no OpenMP per-thread binding; default memory policy'))
    outcomes=[]
    # Predeclared order; no accuracy-dependent task selection or stopping.
    for threads in THREADS:
        for target in TARGETS:
            env=os.environ.copy();env.pop('GOMP_CPU_AFFINITY',None);env.update(thread_environment(threads,cpus))
            args=[sys.executable,'-u','-m',MODULE,'worker','--root',str(root),'--inputs',str(inputs),
                  '--protocol',str(protocol),'--config',str(config_path),'--threads',str(threads),
                  '--target',str(target),'--cpus',','.join(map(str,cpus))]
            log=root/f'worker-t{threads}-target{round(target*100)}.log'
            try:
                with log.open('x') as stream:
                    p=subprocess.run(args,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
                outcome=dict(threads=threads,target=target,returncode=p.returncode,status='DONE' if p.returncode==0 else 'FAILED')
            except subprocess.TimeoutExpired:
                outcome=dict(threads=threads,target=target,status='TIMEOUT',limit_seconds=1200)
            outcomes.append(outcome)
            store(root/f'worker-t{threads}-target{round(target*100)}.status.json',outcome)
    store(root/'worker_statuses.json',outcomes)
    analyze(root,config)


def analyze(root, config):
    """Publish no thread recommendation without all matched targets and exact checks."""
    from .flickr_dense import sha
    rows=[];equivalence=[];missing=[]
    for threads in THREADS:
        for target in TARGETS:
            directory=root/'runs'/f't{threads}-target{round(target*100)}'
            if not (directory/'completed.json').exists():
                missing.append(dict(threads=threads,target=target));continue
            for name,expected in json.loads((directory/'completed.json').read_text()).items():
                if sha(directory/name)!=expected:raise ValueError('Benchmark output changed.')
            row=json.loads((directory/'report.json').read_text())
            row={k:row[k] for k in ['threads','target','search_seconds','actual_forward_calls','cache_hits','peak_rss_bytes','forecast','actual_host']}
            profiles=json.loads((directory/'forward_profile.json').read_text())
            timing=json.loads((directory/'q_timing.json').read_text())
            row['forward_profiles']=profiles;row['q_timing']=timing
            rows.append(row)
            reference=root/'runs'/f't4-target{round(target*100)}'
            if (reference/'completed.json').exists():equivalence.append(compare(root,reference,directory))
    acceptable=[]
    for threads in THREADS:
        measured=[r for r in rows if r['threads']==threads]
        paired=[r for r in equivalence if r['candidate_threads']==threads]
        if len(measured)!=2 or len(paired)!=2:continue
        envelope=forecast(max(r['search_seconds'] for r in measured))
        ram=max(math.ceil((r['peak_rss_bytes']*3+2*1024**3)/1024**2) for r in measured)
        if all(r['scientific_equivalence'] for r in paired) and envelope['passes_acceptance'] and ram<=32768:
            acceptable.append(dict(threads=threads,worst_target_forecast=envelope,required_memory_mb=ram))
    # Exact historical 50% reference is also checked, separately from new4-thread baseline.
    historical=None
    old=config.get('historical_q_reference')
    if old and (root/'runs/t4-target50/completed.json').exists():
        # Old report has a different field schema; compare immutable scientific traces directly.
        historical={name:json.loads((Path(old)/(name+'.json')).read_text())==json.loads((root/'runs/t4-target50'/(name+'.json')).read_text()) for name in ['training','deployment','decisions','q_table','evaluation_cache']}
    if historical is not None:
        old_report=json.loads((Path(old)/'report.json').read_text())
        new_report=json.loads((root/'runs/t4-target50/report.json').read_text())
        for key in ['retained_original_ids','final_widths','parameters_before','parameters_after','tensor_sha256','validation_evaluation_requests']:
            historical[key]=old_report[key]==new_report[key]
        historical['actual_forward_calls']=old_report['observed_validation_evaluations']==new_report['actual_forward_calls']
        historical['cache_hits']=old_report['statistics']['evaluation_cache_hits']==new_report['cache_hits']
        historical['validation_accuracy']=old_report['validation']['accuracy']==new_report['validation']['accuracy']
        historical['validation_macro_f1']=old_report['validation']['macro_f1']==new_report['validation']['macro_f1']
    recommendation=min(acceptable,key=lambda x:x['worst_target_forecast']['safe_seconds']) if acceptable and not missing and historical is not None and all(historical.values()) else None
    with (root/'benchmark_summary.csv').open('x',newline='') as stream:
        fields=['threads','target','search_seconds','actual_forward_calls','cache_hits','peak_rss_bytes','forward_dense_median_seconds','forward_pruned_median_seconds','forecast_minutes','passes170min']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            flat={k:row[k] for k in fields[:6]}
            flat.update(forward_dense_median_seconds=row['forward_profiles']['dense']['median_seconds'],forward_pruned_median_seconds=row['forward_profiles']['pruned']['median_seconds'],forecast_minutes=row['forecast']['safe_minutes'],passes170min=row['forecast']['passes_acceptance'])
            writer.writerow(flat)
    store(root/'analysis.json',dict(status='INCOMPLETE' if missing else 'COMPLETE',missing=missing,runs=rows,
          scientific_equivalence=equivalence,historical_q50_trace_equality=historical,
          recommended_configuration=recommendation,
          q_4_256_resource_gate='GO' if recommendation else 'NO_GO',
          full56_status='NOT_SUBMITTED',full56_release='NO_GO_UNTIL_FRESH_FULL_MANIFEST_GATE',
          other54_policy='Keep prior4-thread allocations/evidence; no higher-thread assumption for unbenchmarked methods.',
          rule='Each target measured separately; campaign recommendation uses max(t30,t50) in unchanged31/3*1.75+180s formula; both exact equivalence and170min required.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['topology','orchestrate','worker'])
    parser.add_argument('--root',type=Path,required=True)
    for name in ['inputs','protocol','config']:parser.add_argument('--'+name,type=Path)
    parser.add_argument('--threads',type=int);parser.add_argument('--target',type=float);parser.add_argument('--cpus')
    a=parser.parse_args()
    if a.stage=='topology':capture_topology(a.root)
    elif a.stage=='orchestrate':orchestrate(a.root,a.inputs,a.protocol,a.config)
    else:worker(a.root,a.inputs,a.protocol,a.config,a.threads,a.target,[int(v) for v in a.cpus.split(',')])


if __name__=='__main__':main()
