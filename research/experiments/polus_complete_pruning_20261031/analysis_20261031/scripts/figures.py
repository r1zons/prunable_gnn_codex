"""Publication figures from frozen analytical tables; validation outcomes only."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

METHODS=['static_random','static_magnitude','static_snip','sequential_random','adaptive_greedy','lookahead_depth2','tabular_q']
LABELS=['Static Random','Magnitude','SNIP-PT','Sequential Random','Greedy','Lookahead d2','Tabular Q (V2)']
COLORS=['#4477AA','#EE7733','#AA4499','#228833','#CC6677','#009988','#665544']
CELLS=[(d,m) for d in ('cora','citeseer','pubmed') for m in ('gcn','graphsage')]
STRATA=[(d,m,t) for d,m in CELLS for t in (.3,.5)]
TITLE={'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed','gcn':'GCN','graphsage':'GraphSAGE'}


def readcsv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=250,metadata={'Software':'Matplotlib audited POLUS analysis'})
    fig.savefig(path.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None,'Title':path.stem})
    plt.close(fig)


def key_label(key):
    d,m,t=key;return f'{TITLE[d]} / {TITLE[m]} / {int(t*100)}%'


def plot(collection: Path,analysis: Path):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.titleweight':'semibold'})
    rows=readcsv(collection/'verified_results.csv');pairs=readcsv(analysis/'PAIRED_COMPARISONS.csv');summary=readcsv(analysis/'METHOD_SUMMARY.csv')
    out=analysis/'FIGURES';out.mkdir(exist_ok=True)
    for filename,yname,cost in [('01_accuracy_achieved_compression','accuracy',False),('02_macro_f1_achieved_compression','macro_f1',False),('03_quality_change_search_cost','macro_f1',True)]:
        fig,axes=plt.subplots(3,2,figsize=(10.5,9.7),sharex=True,sharey=True)
        for ax,(dataset,model) in zip(axes.flat,CELLS):
            cell=[r for r in rows if (r['dataset'],r['architecture'])==(dataset,model)]
            for method,color in zip(METHODS,COLORS):
                for target,marker in ((.3,'o'),(.5,'^')):
                    group=[r for r in cell if r['method']==method and float(r['requested_compression'])==target]
                    x=np.array([float(r['search_seconds'] if cost else r['achieved_compression'])* (1 if cost else 100) for r in group])
                    y=np.array([100*float(r['delta_val_'+yname] if cost else r['pruned_val_'+yname]) for r in group])
                    ax.scatter(x,y,c=color,marker=marker,s=21,alpha=.48,linewidth=.3,edgecolors='white')
                    ax.errorbar(x.mean(),y.mean(),yerr=y.std(ddof=1),fmt=marker,color=color,markersize=5,capsize=2,elinewidth=.8)
            if cost:
                ax.set_xscale('log');ax.axhline(0,color='grey',lw=.6);ax.set_ylim(-82,17)
            else:
                dense={r['dense_training_seed']:float(r['dense_val_'+yname])*100 for r in cell}
                ax.scatter([0]*len(dense),list(dense.values()),c='black',s=22,marker='x');ax.set_xlim(-2,100);ax.set_ylim(0,106)
                ax.axvline(30,color='grey',ls=':',lw=.5);ax.axvline(50,color='grey',ls=':',lw=.5)
            ax.set_title(TITLE[dataset]+' / '+TITLE[model]);ax.grid(alpha=.17)
        for ax in axes[:,0]:ax.set_ylabel('Validation macro-F1 change (pp)' if cost else 'Validation '+('accuracy' if yname=='accuracy' else 'macro-F1')+' (%)')
        for ax in axes[-1]:ax.set_xlabel('Recorded search wall time (s, log scale)' if cost else 'Achieved parameter compression (%)')
        handles=[Line2D([],[],marker='o',lw=0,color=c,label=n) for n,c in zip(LABELS,COLORS)]
        handles += [Line2D([],[],marker='o',lw=0,color='grey',label='Requested 30%'),Line2D([],[],marker='^',lw=0,color='grey',label='Requested 50%')]
        if not cost:handles += [Line2D([],[],marker='x',lw=0,color='black',label='Dense checkpoints')]
        fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,fontsize=8)
        fig.text(.5,.008,'Faint points: paired dense seeds 101/202/303. Filled markers: means; vertical bars: quality sample SD (n=3), not confidence intervals.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.115,1,1));save(fig,out/filename)
    lg=[p for p in pairs if p['method_a']=='adaptive_greedy' and p['method_b']=='lookahead_depth2']
    fig,axes=plt.subplots(1,2,figsize=(11.3,6.5),sharey=True)
    for ax,metric,label in zip(axes,['delta_accuracy_pp','delta_macro_f1_pp'],['Validation accuracy difference (pp)','Validation macro-F1 difference (pp)']):
        for i,key in enumerate(STRATA):
            group=sorted([p for p in lg if (p['dataset'],p['architecture'],float(p['requested_compression']))==key],key=lambda p:int(p['dense_training_seed']))
            values=np.array([float(p[metric]) for p in group]);y=np.arange(3)*.12+i-.12
            ax.scatter(values,y,s=28,c=['#4477AA','#EE7733','#228833'],zorder=4)
            ax.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt='D',ms=4,color='black',capsize=3,zorder=3)
        ax.axvline(0,c='grey',lw=.7);ax.grid(axis='x',alpha=.2);ax.set_xlabel(label+'\nLookahead minus Greedy')
    axes[0].set_yticks(range(12),[key_label(k) for k in STRATA]);axes[0].invert_yaxis()
    handles=[Line2D([],[],marker='o',lw=0,color=c,label='Dense seed '+str(s)) for c,s in zip(['#4477AA','#EE7733','#228833'],[101,202,303])]
    handles.append(Line2D([],[],marker='D',lw=1,color='black',label='Mean ± sample SD, n=3'))
    fig.legend(handles=handles,ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.5,.005));fig.tight_layout(rect=(0,.07,1,1));save(fig,out/'04_greedy_lookahead_paired_quality')
    fig,axes=plt.subplots(2,1,figsize=(11.8,6.6),sharex=True)
    for ax,metric,label in zip(axes,['search_ratio_b_over_a','candidate_evaluation_ratio'],['Search-time ratio (Lookahead / Greedy)','New-candidate evaluation ratio']):
        for i,key in enumerate(STRATA):
            group=[p for p in lg if (p['dataset'],p['architecture'],float(p['requested_compression']))==key];values=np.array([float(p[metric]) for p in group])
            ax.scatter(np.arange(3)*.13+i-.13,values,c='#4477AA',s=22,alpha=.7)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='D',c='black',ms=4,capsize=3)
        ax.axhline(1,color='#aa4444',ls='--',lw=.8);ax.set_ylabel(label);ax.grid(axis='y',alpha=.18)
    axes[-1].set_xticks(range(12),[key_label(k) for k in STRATA],rotation=42,ha='right',fontsize=8)
    fig.text(.5,.01,'Individual paired seeds; diamonds and whiskers = mean ± sample SD (n=3). Search cost is not inference latency.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));save(fig,out/'05_lookahead_cost_ratios')
    fig,axes=plt.subplots(1,2,figsize=(13.4,7.9),gridspec_kw={'width_ratios':[1.25,1]})
    quality=np.zeros((12,7));comp=np.zeros((12,7));sd=np.zeros_like(quality)
    for i,key in enumerate(STRATA):
        for j,method in enumerate(METHODS):
            r=next(s for s in summary if (s['dataset'],s['architecture'],float(s['requested_compression']))==key and s['method']==method)
            quality[i,j]=100*float(r['delta_val_macro_f1_mean']);sd[i,j]=100*float(r['delta_val_macro_f1_sd']);comp[i,j]=100*float(r['achieved_compression_mean'])
    im=axes[0].imshow(quality,cmap='RdBu',norm=TwoSlopeNorm(vmin=-80,vcenter=0,vmax=5),aspect='auto')
    im2=axes[1].imshow(comp,cmap='Blues',vmin=0,vmax=90,aspect='auto')
    for i in range(12):
        for j in range(7):
            axes[0].text(j,i,f'{quality[i,j]:.1f}\n±{sd[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if quality[i,j]<-45 else 'black')
            axes[1].text(j,i,f'{comp[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if comp[i,j]>50 else 'black')
    axes[0].set_yticks(range(12),[key_label(k) for k in STRATA]);axes[1].set_yticks(range(12),[])
    for ax in axes:ax.set_xticks(range(7),LABELS,rotation=48,ha='right',fontsize=8)
    axes[0].set_title('Validation macro-F1 change vs dense (pp)\nmean ± sample SD, n=3');axes[1].set_title('Actual parameter compression (%)\nmean, n=3')
    fig.colorbar(im,ax=axes[0],shrink=.7,label='Macro-F1 change (pp)');fig.colorbar(im2,ax=axes[1],shrink=.7,label='Achieved compression (%)')
    fig.text(.5,.012,'All252 outcomes retained. Requested targets are not matched compression. Zero compression in infeasible Random cells is explicit.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1));save(fig,out/'06_method_performance_and_compression')
    qr=readcsv(analysis/'Q_RUN_DIAGNOSTICS.csv');qd=readcsv(analysis/'Q_DEPLOYMENT_DECISIONS.csv')
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.5))
    for d,m in CELLS:
        selected=[r for r in qr if (r['dataset'],r['architecture'])==(d,m)]
        for r in selected:axes[0].plot([30 if float(r['requested_compression'])==.3 else 50],[100*float(r['achieved_compression'])],marker='o',ms=4,color=COLORS[CELLS.index((d,m))],alpha=.65)
    axes[0].plot([25,55],[25,55],ls=':',c='grey');axes[0].set_xlim(25,55);axes[0].set_ylim(20,92);axes[0].set_xlabel('Requested compression (%)');axes[0].set_ylabel('Achieved compression (%)');axes[0].set_title('Q reaches 87.24–87.71%')
    axes[1].bar(range(6),[6]*6,color='#665544');axes[1].set_xticks(range(6),[TITLE[d]+'\n'+TITLE[m] for d,m in CELLS],rotation=45,ha='right',fontsize=8)
    axes[1].set_ylim(0,7);axes[1].set_ylabel('Deployments ending max_steps');axes[1].set_title('36/36 exhaust 20 actions')
    parts=[sum(r['selected_known']=='False' for r in qd),sum(r['selected_known']=='True' for r in qd)]
    axes[2].bar([0,1],parts,color=['#CC6677','#4477AA']);axes[2].set_xticks([0,1],['Unseen action\n(default Q=0)','Known action'])
    for i,v in enumerate(parts):axes[2].text(i,v+8,str(v),ha='center')
    axes[2].set_ylim(0,540);axes[2].set_ylabel('Recorded deployment decisions');axes[2].set_title('466/720 choices are unseen')
    fig.text(.5,.012,'Q process counts are diagnostic, not independent replicates. STOP is tied for maximum in 452/684 legal states.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,out/'07_q_compression_stopping_coverage')
    case=json.loads((analysis/'LOOKAHEAD_CASE_EVIDENCE.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10.7,4.7))
    dense=next(r for r in rows if r['run_id']=='pubmed-gcn-s202-adaptive_greedy-t50');base=float(dense['dense_val_accuracy'])*100
    for name,color in [('greedy','#CC6677'),('lookahead','#009988')]:
        trace=case[name+'_trace'];xs=[0]+[100*t['sparsity'] for t in trace];ys=[base]+[100*t['validation_accuracy'] for t in trace]
        axes[0].plot(xs,ys,'o-',color=color,label=name.capitalize(),ms=4,lw=1.2)
    annotations=[('Both: L0 −8',case['greedy_trace'][0],(-82,-22)),
                 ('Greedy: L0 −8',case['greedy_trace'][-1],(-100,18)),
                 ('L1 −8',case['lookahead_trace'][1],(20,7)),
                 ('Then L0 −8',case['lookahead_trace'][-1],(-92,8))]
    for label,t,offset in annotations:
        axes[0].annotate(label,(100*t['sparsity'],100*t['validation_accuracy']),xytext=offset,textcoords='offset points',fontsize=8,
                         arrowprops=dict(arrowstyle='-',color='#777777',lw=.6))
    axes[0].set_xlabel('Achieved compression (%)');axes[0].set_ylabel('Validation accuracy (%)');axes[0].legend(frameon=False);axes[0].grid(alpha=.2);axes[0].set_title('PubMed / GCN / seed202 / target50%')
    greedy=case['best_paths_by_first_action']['prune:0:8'];look=case['best_paths_by_first_action']['prune:1:8']
    r1=[greedy['rewards'][0],look['rewards'][0]];future=[0,.9*look['rewards'][1]]
    axes[1].bar([0,1],r1,color='#bb7777',label='Immediate reward');axes[1].bar([0,1],future,bottom=r1,color='#66aaaa',label='0.9 × next reward')
    axes[1].set_xticks([0,1],['L0 / k=8\nterminal','L1 / k=8 → L0 / k=8']);axes[1].set_ylabel('Recorded planning return');axes[1].set_title('Replan after shared first action')
    for i,value in enumerate([greedy['score'],look['score']]):axes[1].text(i,value+.002,f'{value:.6f}',ha='center',fontsize=9)
    axes[1].set_ylim(0,.17);axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    fig.text(.5,.012,'Exploratory case explicitly requested; +4.29pp accuracy with +0.93pp compression and11.69× search. Not a general causal gain.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,out/'08_lookahead_recorded_case')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    args=p.parse_args();plot(args.collection,args.analysis)
    (args.analysis/'FIGURES/ENVIRONMENT.json').write_text(json.dumps(dict(python=sys.version.split()[0],matplotlib=matplotlib.__version__,numpy=np.__version__,formats=['PDF vector','PNG250dpi'],uncertainty='sample SD n=3, not CI',new_model_execution=False),indent=2))
