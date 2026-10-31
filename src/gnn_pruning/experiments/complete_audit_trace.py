"""Independent arithmetic audit of stored transitions; never executes a policy.

Only recorded observations are replayed. No environment, surgery, pruning search,
training, candidate forward or unrecorded counterfactual score is invoked.
"""
from __future__ import annotations
from functools import lru_cache
from fractions import Fraction
import math
import random
import numpy as np
import torch

MENU=(1,2,4,8)
OBS_TOL=1.2e-7  # Recorded environment accuracies are float32 means.


def require(condition: bool, code: str) -> None:
    if not condition:raise ValueError(code)


def close(actual: float, expected: float, code: str, tolerance: float = 1e-12) -> None:
    require(math.isfinite(float(actual)) and abs(float(actual)-float(expected)) <= tolerance,code)


def count_parameters(features: int, classes: int, widths: list | tuple, kind: str) -> int:
    """Independent three-layer geometry, including one bias per convolution."""
    a,b=widths
    return (2 if kind=='graphsage' else 1)*(features*a+a*b+classes*b)+a+b+classes


def projected_state(state: dict, keeps: list[list[int]], features: int, classes: int) -> dict:
    """Direct coordinate selection; no production Local Surgery implementation."""
    dims=[list(range(features)),*keeps,list(range(classes))]
    output={}
    for name,tensor in state.items():
        bits=name.split('.')
        require(len(bits)>=3 and bits[0]=='convs','UNKNOWN_TENSOR_COORDINATES')
        layer=int(bits[1]);rows=torch.tensor(dims[layer+1],dtype=torch.long)
        if tensor.ndim==2:
            cols=torch.tensor(dims[layer],dtype=torch.long)
            output[name]=tensor.index_select(0,rows).index_select(1,cols).contiguous()
        elif tensor.ndim==1:
            output[name]=tensor.index_select(0,rows).contiguous()
        else:raise ValueError('UNKNOWN_PARAMETER_RANK')
    return output


def magnitude_keep(state: dict, keeps: list[list[int]], features: int, classes: int, layer: int, deletion: int) -> list[list[int]]:
    """Reconstruct incident absolute saliency and tie-by-current-ID ranking."""
    projected=projected_state(state,keeps,features,classes)
    width=len(keeps[layer]);scores=torch.zeros(width,dtype=torch.float64)
    for name,value in projected.items():
        index=int(name.split('.')[1]);value=value.abs()
        if index==layer:
            marginal=value.sum(1) if value.ndim==2 else value
            scores += marginal.double()
        elif index==layer+1 and value.ndim==2:
            scores += value.sum(0).double()
    order=sorted(range(width),key=lambda i:(-float(scores[i]),i))
    local=sorted(order[:width-deletion]);result=[list(ids) for ids in keeps]
    result[layer]=[keeps[layer][i] for i in local]
    return result


def actions(widths: list | tuple, steps: int, maximum: int, minimum: int, mode: str, can_reach=None) -> list[str]:
    if steps>=maximum:return []
    result=[f'prune:{layer}:{k}' for layer in (0,1) for k in MENU if widths[layer]-k>=minimum]
    if mode=='sequential_random':
        def reachable(key):
            _,layer,k=key.split(':');future=list(widths);future[int(layer)]-=int(k)
            return can_reach(tuple(future),maximum-steps-1)
        result=[a for a in result if reachable(a)]
    if mode=='tabular_q' and steps>0:result.append('stop')
    return result


def reachability(features: int,classes: int,dense_widths: tuple,kind: str,target: float,tolerance: float,minimum: int) -> tuple:
    """Independent finite width/horizon dynamic programming, integer geometry."""
    p0=count_parameters(features,classes,dense_widths,kind)
    nominal,eps=Fraction(str(target)),Fraction(str(tolerance))
    band=lambda w:abs(Fraction(p0-count_parameters(features,classes,w,kind),p0)-nominal)<=eps
    @lru_cache(None)
    def reachable(widths,remaining):
        if band(widths):return remaining>=0
        if remaining<=0:return False
        for layer in (0,1):
            for k in MENU:
                if widths[layer]-k>=minimum:
                    future=list(widths);future[layer]-=k
                    if reachable(tuple(future),remaining-1):return True
        return False
    return band,reachable


def context(dense: dict, data, report: dict, protocol: dict) -> dict:
    widths=[32,32]
    if 'model_config' in dense:
        hidden=dense['model_config']['hidden_channels'];widths=[hidden,hidden] if isinstance(hidden,int) else list(hidden)
    return dict(state=dense['model_state_dict'],features=int(data.x.shape[1]),classes=int(data.y.max())+1,
        kind=dense['model_name'],dense_widths=widths,target=report['run']['target'],
        tolerance=protocol['adaptive']['tolerance'],minimum=protocol['adaptive']['min_channels'],
        maximum=protocol['adaptive']['max_steps'],penalty=protocol['adaptive']['penalty_lambda'],
        dense_accuracy=float(np.float32(report['_dense_validation_accuracy'])))


def replay(trace: list[dict], ctx: dict, mode: str, start: dict | None = None, magnitude: bool = True) -> tuple[list[dict],dict]:
    """Validate recorded observations and return state/action records and final mask."""
    features,classes,kind=ctx['features'],ctx['classes'],ctx['kind']
    dense_widths=ctx['dense_widths'];p0=count_parameters(features,classes,dense_widths,kind)
    band,reachable=reachability(features,classes,tuple(dense_widths),kind,ctx['target'],ctx['tolerance'],ctx['minimum'])
    if start is None:
        keeps=[list(range(w)) for w in dense_widths];step=0;accuracy=ctx['dense_accuracy'];sparsity=0.;terminal=False
    else:
        keeps=[list(k) for k in start['keeps']];step=start['step'];accuracy=start['accuracy'];sparsity=start['sparsity'];terminal=False
    records=[]
    for entry in trace:
        require(not terminal,'TRANSITION_AFTER_TERMINATION')
        widths=list(map(len,keeps));key=entry['action']
        state=('local_v2',tuple(widths),ctx['maximum']-step,math.floor(accuracy/.01),False)
        legal=actions(widths,step,ctx['maximum'],ctx['minimum'],mode,reachable)
        previous=sparsity
        require(entry['widths_before']==widths,'TRACE_WIDTHS_BEFORE')
        if key in ('initial_target_infeasible','initial_target_band','no_valid_actions'):
            require(start is None and step==0 and len(trace)==1,'INITIAL_TERMINAL_POSITION')
            if key=='initial_target_infeasible':require(mode=='sequential_random' and not reachable(tuple(widths),ctx['maximum']),'FALSE_INFEASIBLE')
            elif key=='initial_target_band':require(mode=='sequential_random' and band(widths),'FALSE_INITIAL_BAND')
            else:require(not legal,'FALSE_NO_ACTIONS')
            reward=0.;reason=key
        else:
            require(key in legal,'ILLEGAL_ACTION')
            if key=='stop':reward=0.;reason='agent_stop'
            else:
                _,layer,k=key.split(':');layer,k=int(layer),int(k)
                if magnitude:
                    keeps=magnitude_keep(ctx['state'],keeps,features,classes,layer,k)
                else:
                    candidate=entry['original_kept_ids']
                    require(len(candidate[layer])==len(keeps[layer])-k,'DELETION_SIZE')
                    require(candidate[1-layer]==keeps[1-layer],'UNRELATED_LAYER_IDS_CHANGED')
                    require(set(candidate[layer])<=set(keeps[layer]),'RETAINED_ID_NOT_SUBSET');keeps=[list(x) for x in candidate]
                step+=1;accuracy=float(entry['validation_accuracy'])
                require(0<=accuracy<=1 and math.isfinite(accuracy),'INVALID_INTERMEDIATE_ACCURACY')
                sparsity=1-count_parameters(features,classes,list(map(len,keeps)),kind)/p0
                reward=.6*(sparsity-previous)-.4*max(0.,ctx['dense_accuracy']-accuracy)
                raw=actions(list(map(len,keeps)),step,ctx['maximum'],ctx['minimum'],'adaptive_greedy')
                if step>=ctx['maximum']:reason='max_steps'
                elif not raw:reason='no_valid_actions'
                elif mode in ('adaptive_greedy','lookahead_depth2') and sparsity>=ctx['target']-ctx['tolerance']:reason='target_reached'
                elif mode=='sequential_random' and band(list(map(len,keeps))):reason='target_band'
                else:reason=None
        penalty=-ctx['penalty']*max(0.,abs(sparsity-ctx['target'])-ctx['tolerance']) if reason else 0.
        reward+=penalty;terminal=reason is not None
        require(entry['terminal_reason']==reason,'TERMINAL_REASON')
        require(entry['original_kept_ids']==keeps,'MAGNITUDE_CHANNEL_IDENTITY' if magnitude else 'RETAINED_IDENTITIES')
        require(entry['widths_after']==list(map(len,keeps)),'TRACE_WIDTHS_AFTER')
        require(entry['successful_deletions']==step,'SUCCESSFUL_ACTION_COUNTER')
        close(entry['sparsity'],sparsity,'TRACE_COMPRESSION')
        close(entry['signed_target_deviation'],sparsity-ctx['target'],'TRACE_TARGET_DEVIATION')
        close(entry['accuracy_drop'],max(0.,ctx['dense_accuracy']-accuracy),'TRACE_DROP',OBS_TOL)
        close(entry['terminal_penalty'],penalty,'TERMINAL_PENALTY')
        close(entry['reward'],reward,'REWARD',OBS_TOL)
        after=('local_v2',tuple(map(len,keeps)),ctx['maximum']-step,math.floor(accuracy/.01),terminal)
        next_actions=[] if terminal else actions(list(map(len,keeps)),step,ctx['maximum'],ctx['minimum'],mode,reachable)
        records.append(dict(state=state,action=key,legal=legal,next_state=after,next_actions=next_actions,
            reward=entry['reward'],terminal=terminal,entry=entry,keeps=[list(x) for x in keeps],
            step=step,accuracy=accuracy,sparsity=sparsity))
    return records,dict(keeps=keeps,step=step,accuracy=accuracy,sparsity=sparsity,terminal=terminal)


def freeze(value):
    return tuple(freeze(x) for x in value) if isinstance(value,(list,tuple)) else value


def audit_q(report: dict,ctx: dict,protocol: dict) -> tuple:
    """Rebuild Q updates/seeded choices from stored episodes, never new rollouts."""
    training=report['training'];episodes=protocol['q_learning']['episodes'];settings=protocol['q_learning']
    rng=random.Random(report['run']['policy_seed']);table={};visited=set();slots={};calls=0;cursor=0
    for episode in range(episodes):
        require(cursor<len(training) and training[cursor].get('event')=='reset' and training[cursor]['episode']==episode,'Q_RESET_OR_EPISODE_ID')
        require(training[cursor]['widths']==ctx['dense_widths'],'Q_RESET_NOT_DENSE');cursor+=1;entries=[]
        while cursor<len(training) and training[cursor].get('event')!='reset':
            entry=training[cursor];require(entry['episode']==episode,'Q_EPISODE_ID');entries.append(entry);cursor+=1
        records,end=replay(entries,ctx,'tabular_q');require(end['terminal'],'Q_EPISODE_NOT_TERMINAL')
        calls+=1+sum(r['action'].startswith('prune:') for r in records)
        for rec in records:
            state,legal=rec['state'],rec['legal'];visited.add(state);slots[state]=len(legal)
            if rng.random()<settings['epsilon']:expected=rng.choice(legal)
            else:expected=max(legal,key=lambda key:table.get(state,{}).get(key,0.))
            require(rec['action']==expected,'Q_RECORDED_SEEDED_SELECTION')
            nxt=rec['next_state'];future=0. if rec['terminal'] else max((table.get(nxt,{}).get(key,0.) for key in rec['next_actions']),default=0.)
            row=table.setdefault(state,{});old=row.get(rec['action'],0.)
            row[rec['action']]=old+settings['alpha']*(rec['reward']+settings['gamma']*future-old)
            visited.add(nxt);slots[nxt]=len(rec['next_actions'])
    require(cursor==len(training),'EXTRA_Q_EPISODES')
    records,end=replay(report['trace'],ctx,'tabular_q');require(end['terminal'],'Q_DEPLOYMENT_NOT_TERMINAL')
    calls+=1+sum(r['action'].startswith('prune:') for r in records)
    for rec in records:
        state=rec['state'];visited.add(state);slots[state]=len(rec['legal'])
        expected=max(rec['legal'],key=lambda key:table.get(state,{}).get(key,0.))
        require(rec['action']==expected,'Q_DEPLOYMENT_NOT_FROZEN_ARGMAX')
    visited.add(records[-1]['next_state']);slots[records[-1]['next_state']]=0
    stored={freeze(row['state']):row['actions'] for row in report['q_table']}
    require(len(stored)==len(report['q_table']) and stored.keys()==table.keys(),'Q_STATE_COVERAGE')
    for state,row in table.items():
        require(row.keys()==stored[state].keys(),'Q_ACTION_COVERAGE')
        for key,value in row.items():close(stored[state][key],value,'Q_BELLMAN_VALUE')
    stats=report['statistics'];entries=sum(map(len,table.values()))
    require(stats['episodes']==episodes and stats['q_entries']==entries and stats['unique_states']==len(visited),'Q_COVERAGE_STATISTICS')
    require(stats['legal_state_action_slots']==sum(slots.values()),'Q_LEGAL_SLOT_COUNT')
    close(stats['q_table_sparsity'],1-entries/sum(slots.values()) if sum(slots.values()) else 1.,'Q_SPARSITY')
    return calls,end,dict(training_episodes=episodes,q_entries=entries,training_transitions=sum(e.get('event')!='reset' for e in training))


def audit_lookahead(report: dict,ctx: dict) -> tuple:
    """Verify all recorded tree arithmetic, legal extensions, ties and cache counts."""
    require(report['statistics']['depth']==2 and report['statistics']['gamma']==.9,'LOOKAHEAD_DEPTH_GAMMA')
    deployment,end=replay(report['trace'],ctx,'lookahead_depth2')
    decisions=report['decisions'];require(len(decisions)==len(deployment),'LOOKAHEAD_DECISION_COUNT')
    previous=dict(keeps=[list(range(w)) for w in ctx['dense_widths']],step=0,accuracy=ctx['dense_accuracy'],sparsity=0.)
    total_new=total_reused=total_transitions=total_paths=0;cached={}
    for selected,decision in zip(deployment,decisions):
        require(freeze(decision['state'])==selected['state'],'LOOKAHEAD_PARENT_STATE')
        require(decision['legal_actions']==selected['legal'],'LOOKAHEAD_LEGAL_ACTIONS')
        require(decision['selected_action']==selected['action'],'LOOKAHEAD_EXECUTED_FIRST_ACTION')
        require(decision['original_kept_ids']==previous['keeps'],'LOOKAHEAD_PARENT_MASK')
        nodes={};new=reused=0;ordered=[]
        for item in decision['candidate_transitions']:
            path=tuple(item['path']);require(path not in nodes and len(path) in (1,2),'LOOKAHEAD_DUPLICATE_PATH')
            parent=previous if len(path)==1 else nodes[path[:1]]['end']
            require(len(path)==1 or not parent['terminal'],'LOOKAHEAD_EXTENDS_TERMINAL')
            records,child=replay([item['transition']],ctx,'lookahead_depth2',start=parent)
            record=records[0];require(record['action']==path[-1],'LOOKAHEAD_PATH_ACTION')
            require(freeze(item['from_state'])==record['state'] and freeze(item['state'])==record['next_state'],'LOOKAHEAD_CANDIDATE_STATE')
            require(item['parameters_after']==count_parameters(ctx['features'],ctx['classes'],list(map(len,child['keeps'])),ctx['kind']),'LOOKAHEAD_CANDIDATE_PARAMETERS')
            if item['evaluation_reused']:
                require(len(path)==1 and path[0] in cached and item['transition']==cached[path[0]],'LOOKAHEAD_INVALID_REUSE');reused+=1
            else:new+=1
            nodes[path]=dict(record=record,end=child);ordered.append(path)
        require([p for p in ordered if len(p)==1]==[(key,) for key in selected['legal']],'LOOKAHEAD_FIRST_ENUMERATION')
        expected=[]
        for first in selected['legal']:
            node=nodes[(first,)];r1=node['record']['reward']
            if node['end']['terminal']:
                expected.append(dict(actions=[first],rewards=[r1],score=r1,terminal=True,terminal_reason=node['record']['entry']['terminal_reason']))
            else:
                continuations=node['record']['next_actions']
                require([p[1] for p in ordered if len(p)==2 and p[0]==first]==continuations,'LOOKAHEAD_NOT_EXHAUSTIVE')
                for second in continuations:
                    leaf=nodes[(first,second)];r2=leaf['record']['reward']
                    expected.append(dict(actions=[first,second],rewards=[r1,r2],score=r1+.9*r2,
                        terminal=leaf['end']['terminal'],terminal_reason=leaf['record']['entry']['terminal_reason']))
        require(len(expected)==len(decision['candidate_paths']),'LOOKAHEAD_PATH_COUNT')
        for actual,exp in zip(decision['candidate_paths'],expected):
            require(actual['actions']==exp['actions'] and actual['terminal']==exp['terminal'] and actual['terminal_reason']==exp['terminal_reason'],'LOOKAHEAD_LEAF_PATH')
            require(actual['rewards']==exp['rewards'],'LOOKAHEAD_PATH_REWARDS');close(actual['score'],exp['score'],'LOOKAHEAD_RETURN')
        best=max(expected,key=lambda x:x['score'])
        require(best['actions']==decision['best_planned_path'] and best['actions'][0]==selected['action'],'LOOKAHEAD_ARGMAX_TIE')
        close(decision['selected_score'],best['score'],'LOOKAHEAD_SELECTED_SCORE')
        require(decision['candidate_evaluations']==new and decision['reused_evaluations']==reused and decision['explored_transitions']==len(nodes),'LOOKAHEAD_EVALUATION_ACCOUNTING')
        require(decision['planning_seconds']>=0,'LOOKAHEAD_NEGATIVE_COST')
        require(nodes[(selected['action'],)]['record']['entry']==selected['entry'],'LOOKAHEAD_ADOPTED_CANDIDATE')
        cached={p[1]:node['record']['entry'] for p,node in nodes.items() if len(p)==2 and p[0]==selected['action']}
        previous=dict(keeps=selected['keeps'],step=selected['step'],accuracy=selected['accuracy'],sparsity=selected['sparsity'])
        total_new+=new;total_reused+=reused;total_transitions+=len(nodes);total_paths+=len(expected)
    stats=report['statistics']
    for key,value in [('candidate_evaluations',total_new),('reused_evaluations',total_reused),('explored_candidate_transitions',total_transitions),('candidate_paths',total_paths)]:
        require(stats[key]==value,'LOOKAHEAD_TOTAL_'+key.upper())
    return 1+total_new,end,dict(candidate_paths=total_paths,reused_evaluations=total_reused)


def audit_behavior(report: dict,ctx: dict,protocol: dict) -> dict:
    """Dispatch evidence audits; record scientific target misses separately."""
    method=report['run']['method'];stats=report['statistics'];extra={}
    if not method.startswith('static_'):
        require(bool(report['trace']) and stats['terminal_reason']==report['trace'][-1]['terminal_reason'],'TERMINAL_STATS_TRACE_MISMATCH')
    if method.startswith('static_'):
        require(not report['trace'] and not report['training'] and not report['q_table'] and not report['decisions'],'STATIC_HAS_ADAPTIVE_TRACE')
        plan=stats['static_plan']
        require(plan['criterion']=={'static_random':'random','static_magnitude':'magnitude','static_snip':'snip'}[method],'STATIC_CRITERION_IDENTITY')
        require(plan['kept_indices']==report['retained_original_ids'],'STATIC_PLAN_IDENTITIES')
        require(plan['widths']==report['final_widths'] and plan['parameters_after']==report['parameters_after'],'STATIC_PLAN_DIMENSIONS')
        close(plan['achieved_sparsity'],report['achieved_compression'],'STATIC_PLAN_COMPRESSION')
        require(plan['candidate_count']>=plan['eligible_candidate_count']>0,'STATIC_CANDIDATE_COUNTS')
        expected='static_nearest_fallback' if plan['fallback'] else 'static_target_band'
        require(stats['terminal_reason']==expected,'STATIC_STOP')
        band,_=reachability(ctx['features'],ctx['classes'],tuple(ctx['dense_widths']),ctx['kind'],ctx['target'],ctx['tolerance'],ctx['minimum'])
        exists=any(band((a,b)) for a in range(ctx['minimum'],ctx['dense_widths'][0]+1) for b in range(ctx['minimum'],ctx['dense_widths'][1]+1))
        require(plan['fallback']==(not exists),'STATIC_FALSE_FALLBACK')
        if exists:require(band(report['final_widths']),'STATIC_NOT_IN_BAND')
        calls=0;end=dict(keeps=report['retained_original_ids'],terminal=True)
    elif method=='tabular_q':calls,end,extra=audit_q(report,ctx,protocol)
    elif method=='lookahead_depth2':calls,end,extra=audit_lookahead(report,ctx)
    else:
        records,end=replay(report['trace'],ctx,method);require(end['terminal'],'NONTERMINAL_DEPLOYMENT')
        require(not report['training'] and not report['q_table'] and not report['decisions'],'CONTROL_HAS_LEARNING_TRACE')
        if method=='adaptive_greedy':
            candidates=sum(len(r['legal']) for r in records if r['action'].startswith('prune:'))
            require(stats['candidate_evaluations']==candidates,'GREEDY_CANDIDATE_ACCOUNTING');calls=1+candidates
            extra['argmax_verification']='NOT_VERIFIED: unselected Greedy candidate observations were not saved'
        elif method=='sequential_random':
            rng=random.Random(report['run']['policy_seed'])
            for r in records:
                if r['action'].startswith('prune:'):require(r['action']==rng.choice(r['legal']),'SEQUENTIAL_RANDOM_SEEDED_ACTION')
            calls=1+sum(r['action'].startswith('prune:') for r in records)
        else:raise ValueError('UNKNOWN_METHOD')
    require(end['keeps']==report['retained_original_ids'],'FINAL_TRACE_MASK')
    require(stats['validation_evaluations']==calls==report['observed_validation_evaluations'],'VALIDATION_EVALUATION_ACCOUNTING')
    require(report['search_seconds']>=0 and math.isfinite(report['search_seconds']),'INVALID_SEARCH_COST')
    return extra
