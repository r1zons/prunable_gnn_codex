"""Validation-only episode/decision diagnostics; no test metrics or labels are loaded."""
import collections,csv,json,math,random
from pathlib import Path
from fractions import Fraction
import numpy as np
from pilot_runner import ROOT


def dump(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def table(name,rows):
    fields=list(rows[0])
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def state(s):return(s[0],tuple(s[1]),s[2],s[3],s[4])
def legal(widths,steps):return [f'prune:{l}:{k}' for l,w in enumerate(widths) for k in (1,2,4,8) if w-k>=4]+(['stop'] if steps>0 else [])
def count(kind,widths):
    a,b=widths;return (2 if kind=='graphsage' else 1)*(1433*a+a*b+7*b)+a+b+7

def verify_run(result,tr,audit,q):
    """Replay all original RNG choices, Bellman updates, counters and rewards independently."""
    N=result['episodes'];kind=result['model'];p0=count(kind,[32,32]);target=result['target_sparsity'];penalty=result['penalty_lambda'];tableq={};visited=set();slots={};rng=random.Random(101);decision_index=0;evals=0
    for episode in range(N+1):
        entries=[e for e in tr['training'] if e.get('episode')==episode and e.get('event')!='reset'] if episode<N else tr['deployment']
        assert entries;reset=audit['resets'][episode];assert reset['widths']==[32,32]
        widths=[32,32];ids=[list(range(32)),list(range(32))];step=0;sparsity=0.;acc=float(np.float32(result['dense_validation']['accuracy']));a0=acc;evals+=1
        for e in entries:
            s=('local_v2',tuple(widths),20-step,math.floor(acc/.01),False);actions=legal(widths,step);visited.add(s);slots[s]=len(actions)
            observed=audit['decisions'][decision_index];decision_index+=1;assert state(observed['state'])==s and observed['original_kept_ids']==ids
            row=tableq.get(s,{})
            assert observed['legal_q']=={a:row.get(a,0.) for a in actions} and observed['visited_actions']==[a for a in actions if a in row]
            explore=episode<N and rng.random()<.2
            expected=rng.choice(actions) if explore else max(actions,key=lambda a:row.get(a,0.))
            assert e['action']==observed['selected_action']==expected and observed['exploration']==explore
            assert observed['selected_was_unvisited']==(expected not in row)
            before=sparsity
            if expected=='stop':
                reward=0.;reason='agent_stop';assert step>0 and e['original_kept_ids']==ids
            else:
                _,layer,k=expected.split(':');layer,k=int(layer),int(k);newids=e['original_kept_ids']
                assert newids[1-layer]==ids[1-layer] and len(newids[layer])==len(ids[layer])-k and set(newids[layer])<set(ids[layer])
                ids=newids;widths=list(map(len,ids));step+=1;evals+=1;acc=e['validation_accuracy'];sparsity=1-count(kind,widths)/p0
                reward=.6*(sparsity-before)-.4*max(0.,a0-acc)
                reason='max_steps' if step==20 else ('no_valid_actions' if not legal(widths,0) else None)
            term=-penalty*max(0.,abs(sparsity-target)-.01) if reason else 0.
            for a,b in [(e['reward'],reward+term),(e['terminal_penalty'],term),(e['sparsity'],sparsity),(e['accuracy_drop'],max(0.,a0-acc)),(e['signed_target_deviation'],sparsity-target)]:assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)
            assert e['widths_after']==widths and e['widths_before']==list(s[1]) and e['successful_deletions']==step and e['terminal_reason']==reason
            terminal=reason is not None;ns=('local_v2',tuple(widths),20-step,math.floor(acc/.01),terminal);nextactions=[] if terminal else legal(widths,step)
            if episode<N:
                visited.add(ns);slots[ns]=len(nextactions);future=0. if terminal else max(tableq.get(ns,{}).get(a,0.) for a in nextactions)
                old=tableq.get(s,{}).get(expected,0.);new=old+.1*(e['reward']+.9*future-old);tableq.setdefault(s,{})[expected]=new
                update=observed['update'];assert state(update['next_state'])==ns and update['next_actions']==nextactions
                for a,b in [(update['q_before'],old),(update['q_after'],new),(update['bootstrap'],future)]:assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
            assert terminal==(e is entries[-1])
        if episode==N:visited.add(ns);slots[ns]=0
    saved={state(r['state']):r['actions'] for r in q};assert saved==tableq
    stats=result['statistics'];assert stats['validation_evaluations']==evals and stats['unique_states']==len(visited) and stats['q_entries']==sum(map(len,tableq.values())) and stats['legal_state_action_slots']==sum(slots.values())
    assert decision_index==len(audit['decisions']);assert math.isclose(stats['q_table_sparsity'],1-stats['q_entries']/sum(slots.values()))


def q_summary():
    deployments=[];episode_rows=[];audits={};comparisons=[];commands={r['run_id']:r for r in json.loads((ROOT/'commands.json').read_text())}
    for p in sorted(ROOT.glob('q_runs/*/result.json')):
        d=json.loads(p.read_text());trace=json.loads((p.parent/'trace.json').read_text());a=json.loads((p.parent/'audit.json').read_text());q=json.loads((p.parent/'q_table.json').read_text());verify_run(d,trace,a,q)
        ep_rows=[];allactions=[]
        for episode in range(d['episodes']):
            entries=[e for e in trace['training'] if e.get('episode')==episode and e.get('event')!='reset'];last=entries[-1];allactions.extend(e['action'] for e in entries)
            decisions=[e for e in a['decisions'] if e['phase']=='training' and e['episode']==episode]
            row=dict(run_id=d['run_id'],model=d['model'],target_sparsity=d['target_sparsity'],episodes_budget=d['episodes'],penalty_lambda=d['penalty_lambda'],episode=episode,terminal_reason=last['terminal_reason'],successful_deletions=last['successful_deletions'],hidden_widths=last['widths_after'],achieved_sparsity=last['sparsity'],signed_target_deviation=last['signed_target_deviation'],validation_accuracy=last['validation_accuracy'],terminal_penalty=last['terminal_penalty'],discounted_return=sum(.9**i*e['reward'] for i,e in enumerate(entries)),ever_in_target_band=any(abs(e['sparsity']-d['target_sparsity'])<=.01 for e in entries),final_in_target_band=abs(last['sparsity']-d['target_sparsity'])<=.01,stop_was_exploration=decisions[-1]['exploration'] if last['action']=='stop' else None,actions=[e['action'] for e in entries],original_kept_ids=last['original_kept_ids'])
            episode_rows.append(row);ep_rows.append(row)
        dep=[e for e in a['decisions'] if e['phase']=='deployment'];reasons=collections.Counter(e['terminal_reason'] for e in ep_rows);stats=d['statistics']
        actions=collections.Counter(allactions);kcounts=collections.Counter(x.split(':')[2] for x in allactions if x!='stop');layers=collections.Counter(x.split(':')[1] for x in allactions if x!='stop')
        stateids=collections.defaultdict(set);stateacc=collections.defaultdict(set);sa_outcomes=collections.defaultdict(set)
        for e in a['decisions']:
            s=state(e['state']);sig=tuple(map(tuple,e['original_kept_ids']));stateids[s].add(sig);stateacc[s].add(e['exact_validation_accuracy'])
            if 'update' in e:
                u=e['update'];sa_outcomes[(s,e['selected_action'])].add((sig,tuple(u['next_state'][1]),u['reward']))
        collision_states=[s for s,v in stateids.items() if len(v)>1]
        reward_collision=[(s,act) for (s,act),v in sa_outcomes.items() if len({x[2] for x in v})>1]
        unvisited=[e for e in dep if e['selected_was_unvisited']];optimistic=[e for e in unvisited if any(v<0 for k,v in e['legal_q'].items() if k in e['visited_actions'])]
        updates=[e['update'] for e in a['decisions'] if 'update' in e];qvalues=[v for row in q for v in row['actions'].values()]
        audit_report=dict(deployment_actions=[e['selected_action'] for e in dep],deployment_selected_unvisited=len(unvisited),deployment_unvisited_preferred_with_negative_visited=len(optimistic),training_exploration_count=sum(e['exploration'] for e in a['decisions'] if e['phase']=='training'),training_action_counts=dict(actions),training_k_counts=dict(kcounts),training_layer_counts=dict(layers),deployment_k_counts=dict(collections.Counter(e['selected_action'].split(':')[2] for e in dep if e['selected_action']!='stop')),deployment_layer_counts=dict(collections.Counter(e['selected_action'].split(':')[1] for e in dep if e['selected_action']!='stop')),states_with_multiple_retained_id_signatures=len(collision_states),states_with_multiple_exact_accuracies=sum(len(v)>1 for v in stateacc.values()),state_actions_with_multiple_observed_rewards=len(reward_collision),collision_examples=[dict(state=s,retained_id_signatures=list(stateids[s])[:2],exact_accuracies=sorted(stateacc[s])) for s in collision_states[:3]],reward_collision_examples=[dict(state=s,action=act,outcomes=list(sa_outcomes[(s,act)])[:3]) for s,act in reward_collision[:3]],terminal_updates=sum(u['terminal'] for u in updates),nonterminal_bootstrap_zero=sum(not u['terminal'] and u['bootstrap']==0 for u in updates),nonterminal_updates=sum(not u['terminal'] for u in updates),negative_q_entries=sum(v<0 for v in qvalues),positive_q_entries=sum(v>0 for v in qvalues),q_min=min(qvalues),q_max=max(qvalues),off_target_stop_episodes=sum(e['terminal_reason']=='agent_stop' and not e['final_in_target_band'] for e in ep_rows),stop_exploration_episodes=sum(e['stop_was_exploration'] is True for e in ep_rows),stop_exploitation_episodes=sum(e['stop_was_exploration'] is False for e in ep_rows),episode_ever_hit_count=sum(e['ever_in_target_band'] for e in ep_rows),episode_final_hit_count=sum(e['final_in_target_band'] for e in ep_rows),deployment_ever_hit=any(abs(e['sparsity']-d['target_sparsity'])<=.01 for e in trace['deployment']))
        audits[d['run_id']]=audit_report
        row={k:d[k] for k in ('run_id','model','checkpoint_training_seed','policy_seed','target_sparsity','episodes','penalty_lambda','max_steps','hidden_widths','original_kept_ids','achieved_sparsity','signed_target_deviation','target_gap','target_violation','validation_accuracy','validation_macro_f1','terminal_reason','successful_deletions','candidate_evaluations','search_seconds','scoring_seconds','surgery_seconds','policy_other_seconds','runner_seconds','checkpoint','status')}
        row.update(command_seconds=commands[d['run_id']]['elapsed_seconds'],unique_states=stats['unique_states'],q_entries=stats['q_entries'],legal_state_action_slots=stats['legal_state_action_slots'],legal_state_action_coverage=1-stats['q_table_sparsity'],q_table_sparsity=stats['q_table_sparsity'],validation_evaluations=stats['validation_evaluations'],stop_episodes=reasons['agent_stop'],stop_proportion=reasons['agent_stop']/d['episodes'],max_steps_episodes=reasons['max_steps'],max_steps_proportion=reasons['max_steps']/d['episodes'],no_valid_actions_episodes=reasons['no_valid_actions'],no_valid_actions_proportion=reasons['no_valid_actions']/d['episodes'],episode_final_hits=audit_report['episode_final_hit_count'],deployment_unvisited_choices=len(unvisited),multiple_id_states=len(collision_states),training_k_counts=dict(kcounts),training_layer_counts=dict(layers))
        deployments.append(row)
    for kind in ('gcn','graphsage'):
        for target in (.3,.5):
            s=f'{kind}_target{int(target*100)}_episodes'
            p30=ROOT/'q_runs'/f'{s}30_lambda2/trace.json';p100=ROOT/'q_runs'/f'{s}100_lambda2/trace.json'
            if p100.exists():
                a=json.loads(p30.read_text());b=json.loads(p100.read_text());assert a['training']==[e for e in b['training'] if e['episode']<30]
    # Compare target-conditioned tables and greedy choices over shared visited states.
    for kind in ('gcn','graphsage'):
        for episodes in (30,100):
            paths=[ROOT/'q_runs'/f'{kind}_target{t}_episodes{episodes}_lambda2/q_table.json' for t in (30,50)]
            if not all(p.exists() for p in paths):continue
            tables=[{state(r['state']):r['actions'] for r in json.loads(p.read_text())} for p in paths];common=set(tables[0])&set(tables[1]);different=0;value_diffs=[]
            for s in common:
                acts=legal(s[1],20-s[2]);choices=[max(acts,key=lambda k:t[s].get(k,0.)) for t in tables];different+=choices[0]!=choices[1]
                value_diffs.extend(abs(tables[0][s].get(k,0.)-tables[1][s].get(k,0.)) for k in set(tables[0][s])|set(tables[1][s]))
            comparisons.append(dict(model=kind,episodes=episodes,common_q_states=len(common),different_greedy_choices=different,max_common_q_difference=max(value_diffs,default=0.),deployment_actions_equal=audits[f'{kind}_target30_episodes{episodes}_lambda2']['deployment_actions']==audits[f'{kind}_target50_episodes{episodes}_lambda2']['deployment_actions']))
    table('q_pilot.csv',deployments);table('q_episodes.csv',episode_rows);dump('q_diagnostics.json',audits);dump('target_policy_comparison.json',comparisons)
    return deployments


def random_summary():
    rows=[];summaries=[];matrix=list(csv.DictReader((Path.cwd()/'research/experiments/local_structured_v2_cora_matrix_20261027/summary.csv').open()))
    for p in sorted(ROOT.glob('random_runs/*/result.json')):
        d=json.loads(p.read_text());rows.append({k:d[k] for k in ('model','target_sparsity','mask_seed','checkpoint_training_seed','hidden_widths','original_kept_ids','parameters_after','achieved_sparsity','validation_accuracy','validation_macro_f1','seconds','checkpoint','status')})
    for kind in ('gcn','graphsage'):
        for target in (.3,.5):
            trials=[r for r in rows if r['model']==kind and r['target_sparsity']==target];assert len(trials)==20
            y=np.array([r['validation_accuracy'] for r in trials]);compression=np.array([r['achieved_sparsity'] for r in trials]);width2=np.array([r['hidden_widths'][1] for r in trials]);widths=collections.defaultdict(list)
            for r in trials:widths[tuple(r['hidden_widths'])].append(r['validation_accuracy'])
            baseline={r['method']:r for r in matrix if r['model']==kind and float(r['target_sparsity'])==target};mag=float(baseline['magnitude_local_v2']['validation_accuracy']);old=float(baseline['random_local_v2']['validation_accuracy']);greedy=float(baseline['adaptive_greedy_local_v2']['validation_accuracy'])
            total=float(((y-y.mean())**2).sum());within=sum(sum((v-np.mean(values))**2 for v in values) for values in widths.values());between=total-within
            rankx=np.argsort(np.argsort(width2)).astype(float) # Pearson also given; rank ties use mean ranks below.
            from scipy.stats import spearmanr
            rho=float(spearmanr(width2,y).statistic)
            eligible=[];p0=count(kind,[32,32])
            for a in range(4,33):
                for b in range(4,33):
                    comp=Fraction(p0-count(kind,[a,b]),p0)
                    if abs(comp-Fraction(str(target)))<=Fraction(1,100):eligible.append([a,b])
            summaries.append(dict(model=kind,target_sparsity=target,n_masks=20,accuracy_mean=float(y.mean()),accuracy_sd=float(y.std(ddof=1)),accuracy_median=float(np.median(y)),accuracy_min=float(y.min()),accuracy_max=float(y.max()),accuracy_p10=float(np.quantile(y,.1)),accuracy_p90=float(np.quantile(y,.9)),compression_mean=float(compression.mean()),compression_min=float(compression.min()),compression_max=float(compression.max()),width_counts={f'{a}x{b}':len(v) for (a,b),v in sorted(widths.items())},eligible_width_pairs=eligible,width_group_means={f'{a}x{b}':float(np.mean(v)) for (a,b),v in sorted(widths.items())},repeated_width_groups=sum(len(v)>1 for v in widths.values()),within_width_accuracy_ranges={f'{a}x{b}':max(v)-min(v) for (a,b),v in sorted(widths.items()) if len(v)>1},within_width_ss_fraction=within/total,between_width_ss_fraction=between/total,width2_accuracy_spearman=rho,original_random_accuracy=old,original_random_empirical_percentile=float(((y<old).sum()+.5*(y==old).sum())/len(y)),original_outside_p10_p90=bool(old<np.quantile(y,.1) or old>np.quantile(y,.9)),magnitude_accuracy=mag,magnitude_compression=float(baseline['magnitude_local_v2']['achieved_sparsity']),greedy_accuracy=greedy,greedy_compression=float(baseline['adaptive_greedy_local_v2']['achieved_sparsity']),strong_at_least_magnitude_count=int((y>=mag).sum()),weak_10pp_below_magnitude_count=int((y<=mag-.1).sum()),old_greedy_search_seconds=float(baseline['adaptive_greedy_local_v2']['search_seconds'])))
    table('random_trials.csv',rows);table('random_summary.csv',summaries);dump('random_summary.json',summaries)
    return summaries

if __name__=='__main__':
    q=q_summary();r=random_summary();print(json.dumps({'q_runs_summarized':len(q),'random_conditions':len(r)},indent=2))
