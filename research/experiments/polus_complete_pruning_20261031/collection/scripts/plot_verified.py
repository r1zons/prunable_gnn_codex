"""Validation-only diagnostics from an audited dataset; no model inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS=['static_random','static_magnitude','static_snip','sequential_random','adaptive_greedy','lookahead_depth2','tabular_q']
LABELS=['Static Random','Static Magnitude','Static SNIP-PT','Sequential Random','Adaptive Greedy','Lookahead depth 2','Tabular Q (V2)']
COLORS=['#4285b4','#d77b24','#8c63a7','#55985b','#d34d52','#20a3a0','#795b42']
CELLS=[(d,m) for d in ('cora','citeseer','pubmed') for m in ('gcn','graphsage')]
NAMES={'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed','gcn':'GCN','graphsage':'GraphSAGE'}


def plot(audit,output):
    output.mkdir(parents=True,exist_ok=True)
    rows=audit['runs'];good=[r for r in rows if r['verified']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    for filename,ykey,ylabel,cost in [('accuracy_compression','pruned_val_accuracy','Validation accuracy',False),
                                      ('f1_compression','pruned_val_macro_f1','Validation macro-F1',False),
                                      ('quality_search_cost','pruned_val_macro_f1','Validation macro-F1',True)]:
        fig,axes=plt.subplots(3,2,figsize=(12.3,11.7),sharex=True,sharey=True)
        for ax,(dataset,model) in zip(axes.flat,CELLS):
            cell=[r for r in good if r['dataset']==dataset and r['architecture']==model]
            for method,color in zip(METHODS,COLORS):
                for target,marker in ((.3,'o'),(.5,'^')):
                    selected=[r for r in cell if r['method']==method and r['requested_compression']==target]
                    ax.scatter([r['search_seconds'] if cost else r['achieved_compression'] for r in selected],
                               [r[ykey] for r in selected],s=38,marker=marker,c=color,alpha=.8,edgecolors='white',linewidths=.4)
            if not cost:
                dense={r['dense_training_seed']:r['dense_val_'+('accuracy' if ykey.endswith('accuracy') else 'macro_f1')] for r in cell}
                ax.scatter([0]*len(dense),list(dense.values()),marker='x',s=35,c='#222222',linewidth=1)
                ax.axvline(.3,color='#aaaaaa',ls=':',lw=.6);ax.axvline(.5,color='#aaaaaa',ls=':',lw=.6)
                ax.set_xlim(-.025,1.)
            else:ax.set_xscale('log')
            ax.set_ylim(0,1.02);ax.grid(alpha=.18);ax.set_title(NAMES[dataset]+' / '+NAMES[model])
        for ax in axes[:,0]:ax.set_ylabel(ylabel)
        for ax in axes[-1]:ax.set_xlabel('Recorded search wall time (s, log scale)' if cost else 'Achieved physical parameter compression')
        handles=[Line2D([],[],marker='o',color=c,lw=0,label=l) for c,l in zip(COLORS,LABELS)]
        handles.extend([Line2D([],[],marker='o',color='#555555',lw=0,label='Requested 30%'),Line2D([],[],marker='^',color='#555555',lw=0,label='Requested 50%')])
        if not cost:handles.append(Line2D([],[],marker='x',color='#222222',lw=0,label='Frozen dense (3 seeds)'))
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.025),fontsize=9)
        fig.suptitle(('Quality and recorded search cost' if cost else ylabel+' and achieved compression')+' — verified POLUS outputs',fontsize=14,y=.986)
        fig.text(.5,.008,'Each point is a submitted condition; 3 independent dense seeds. Test results are not used in these figures.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.12,1,.968));fig.savefig(output/(filename+'.png'));plt.close(fig)
    counts=np.zeros((6,7));annotations={}
    for i,(dataset,model) in enumerate(CELLS):
        for j,method in enumerate(METHODS):
            cell=[r for r in rows if (r['dataset'],r['architecture'],r['method'])==(dataset,model,method)]
            verified=[r for r in cell if r['verified']];counts[i,j]=len(verified)
            annotations[i,j]=f'{len(verified)}/{len(cell)}\n{sum(r["physical_pruning"] for r in verified)} pruned'
    fig,ax=plt.subplots(figsize=(12.6,5.9));im=ax.imshow(counts,cmap='Blues',vmin=0,vmax=6,aspect='auto')
    for (i,j),text in annotations.items():ax.text(j,i,text,ha='center',va='center',color='white' if counts[i,j]>=4 else 'black',fontsize=10)
    ax.set_yticks(range(6),[NAMES[d]+' / '+NAMES[m] for d,m in CELLS]);ax.set_xticks(range(7),['Static\nRandom','Static\nMagnitude','Static\nSNIP-PT','Sequential\nRandom','Adaptive\nGreedy','Lookahead\ndepth 2','Tabular Q\n(V2)'])
    ax.set_title('Verified / expected artifact sets; physical pruning shown separately',pad=16)
    fig.colorbar(im,ax=ax,label='Verified artifact sets',shrink=.85)
    fig.text(.5,.02,'6 = 3 dense training seeds × 2 targets. Verified infeasible outcomes remain visible; verification is not target achievement.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(output/'coverage.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(json.loads(a.audit.read_text()),a.output)
