"""Matplotlib-only plots of validation CSVs; no dataset/checkpoint access."""
import collections,csv,json,sys,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
q=list(csv.DictReader((ROOT/'q_pilot.csv').open()));trials=list(csv.DictReader((ROOT/'random_trials.csv').open()));summary=list(csv.DictReader((ROOT/'random_summary.csv').open()))
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':160})
def save(fig,name):
    fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    svg=figdir/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
# 1. Actual compression and validation, without compression-matched superiority claims.
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,(kind,target) in zip(axes.flat,[(m,t) for m in ('gcn','graphsage') for t in (.3,.5)]):
    rs=[r for r in trials if r['model']==kind and float(r['target_sparsity'])==target];qs=[r for r in q if r['model']==kind and float(r['target_sparsity'])==target];s=next(r for r in summary if r['model']==kind and float(r['target_sparsity'])==target)
    ax.axvspan(100*(target-.01),100*(target+.01),color='grey',alpha=.16)
    ax.scatter([100*float(r['achieved_sparsity']) for r in rs],[100*float(r['validation_accuracy']) for r in rs],c='#3476a8',s=21,alpha=.7,label='Random masks')
    for r in qs:
        ax.scatter(100*float(r['achieved_sparsity']),100*float(r['validation_accuracy']),marker='X' if float(r['penalty_lambda'])!=2 else ('s' if int(r['episodes'])==30 else 'D'),s=55,color='#bd4f32' if float(r['penalty_lambda'])==2 else '#8759a8')
    ax.scatter(100*float(s['magnitude_compression']),100*float(s['magnitude_accuracy']),marker='^',s=75,c='#2c9149',edgecolors='black',linewidths=.4)
    ax.scatter(100*float(s['greedy_compression']),100*float(s['greedy_accuracy']),marker='*',s=130,c='#dfb237',edgecolors='black',linewidths=.4)
    ax.set_title(f'{kind.upper()} | target {target:.0%}');ax.set_xlim(26,91);ax.set_ylim(5,98);ax.grid(alpha=.18)
for ax in axes[-1]:ax.set_xlabel('Achieved parameter compression (%)')
for ax in axes[:,0]:ax.set_ylabel('Validation accuracy (%)')
handles=[Line2D([0],[0],marker=m,color='w',markerfacecolor=c,markersize=7,label=l) for m,c,l in [('o','#3476a8','Random, 20 masks'),('^','#2c9149','Magnitude'),('*','#dfb237','Greedy'),('s','#bd4f32','Q 30 episodes, lambda=2'),('D','#bd4f32','Q 100 episodes, lambda=2'),('X','#8759a8','Q targeted lambda checks')]]
fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.025));fig.suptitle('Frozen Cora seed101: validation only; outcomes at different actual compression');fig.tight_layout(rect=(0,.065,1,.96));save(fig,'compression_validation')
# 2. Budget/lambda and STOP behavior. Missing cells are not interpolated.
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
categories=[(30,2.),(100,2.),(100,.5),(100,5.)]
for col,kind in enumerate(('gcn','graphsage')):
    for target,color in ((.3,'#3476a8'),(.5,'#bd4f32')):
        xs=[];comp=[];stop=[]
        for x,(episodes,lam) in enumerate(categories):
            rows=[r for r in q if r['model']==kind and float(r['target_sparsity'])==target and int(r['episodes'])==episodes and float(r['penalty_lambda'])==lam]
            if rows:xs.append(x);comp.append(100*float(rows[0]['achieved_sparsity']));stop.append(100*float(rows[0]['stop_proportion']))
        axes[0,col].plot(xs,comp,'o-',color=color,label=f'target {target:.0%}');axes[1,col].plot(xs,stop,'o-',color=color)
        axes[0,col].axhspan((target-.01)*100,(target+.01)*100,color=color,alpha=.09)
    axes[0,col].set_title(kind.upper());axes[0,col].set_ylim(20,95);axes[1,col].set_ylim(-2,102);axes[0,col].legend()
    axes[1,col].set_xticks(range(4),['30 ep\nlambda 2','100 ep\nlambda 2','100 ep\nlambda 0.5','100 ep\nlambda 5'])
    for ax in axes[:,col]:ax.grid(alpha=.18)
axes[0,0].set_ylabel('Deployed compression (%)');axes[1,0].set_ylabel('Training episodes ending STOP (%)');fig.suptitle('Budget comparison and targeted lambda sensitivity (one policy RNG seed)');fig.tight_layout(rect=(0,0,1,.95));save(fig,'q_budget_lambda')
# 3. All masks, with hidden second-layer width shown by color.
fig,ax=plt.subplots(figsize=(10,5.8));conditions=[(m,t) for m in ('gcn','graphsage') for t in (.3,.5)]
values=[[100*float(r['validation_accuracy']) for r in trials if r['model']==m and float(r['target_sparsity'])==t] for m,t in conditions]
ax.boxplot(values,positions=range(4),widths=.52,showfliers=False,medianprops={'color':'black','linewidth':1.5})
for x,(m,t) in enumerate(conditions):
    rs=[r for r in trials if r['model']==m and float(r['target_sparsity'])==t];s=next(r for r in summary if r['model']==m and float(r['target_sparsity'])==t);jitter=np.linspace(-.19,.19,len(rs))
    sc=ax.scatter(x+jitter,[100*float(r['validation_accuracy']) for r in rs],c=[json.loads(r['hidden_widths'])[1] for r in rs],vmin=4,vmax=32,cmap='viridis',s=31,zorder=3,alpha=.9)
    ax.scatter(x+.29,100*float(s['original_random_accuracy']),c='#be3c36',marker='D',s=45,zorder=4)
    ax.scatter(x-.29,100*float(s['magnitude_accuracy']),c='#2c9149',marker='^',s=55,zorder=4)
    ax.scatter(x+.38,100*float(s['greedy_accuracy']),c='#cf9b18',marker='*',s=95,zorder=4)
    ax.scatter(x,100*float(s['accuracy_mean']),c='#3476a8',marker='x',s=50,zorder=5)
ax.set_xticks(range(4),['GCN 30%','GCN 50%','GraphSAGE 30%','GraphSAGE 50%']);ax.set_ylabel('Validation accuracy (%)');ax.set_ylim(5,98);ax.grid(axis='y',alpha=.18)
ax.set_title('Random mask-selection variability: 20 masks per condition, one frozen dense seed')
fig.colorbar(sc,ax=ax,label='Retained second hidden width',pad=.025)
handles=[Line2D([0],[0],marker=m,color='w',markerfacecolor=c,markeredgecolor=c,markersize=7,label=l) for m,c,l in [('D','#be3c36','Original Random mask'),('^','#2c9149','Magnitude'),('*','#cf9b18','Greedy (compression differs at 30%)'),('x','#3476a8','Random mean')]]
fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.47,-.02));fig.tight_layout(rect=(0,.07,1,1));save(fig,'random_distributions')
(ROOT/'plot_runtime.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'q_rows':len(q),'random_rows':len(trials),'purpose':'Plot validation CSVs only; existing environment, no dataset/model access'},indent=2))
print('3 plots exported as PNG and SVG')
