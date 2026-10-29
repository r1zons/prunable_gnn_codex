"""Plot verified derived files only; compatible with existing Python3.8/Matplotlib."""
import argparse,csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ORDER=['q_v4_menu12','q_v4_percent','q_v4_300','q_v4_decay300','q_v4_random_ties','q_v4_incremental_drop','q_v4_gamma0']
LABELS=['A: {1,2}','A: width %','B: 300 eps','C: decay','D: random ties','E: incremental drop','F: gamma 0']
SHORT=['A12','A%','B300','C decay','D ties','E drop','F gamma0']
COLORS=['#0072B2','#56B4E9','#009E73','#E69F00','#CC79A7','#D55E00','#332288']
def read(p):return json.loads(Path(p).read_text())
def mean_sd(x):return float(np.mean(x)),float(np.std(x,ddof=1)) if len(x)>1 else 0.
def select(rows,model,target,variant):return [r for r in rows if r['model']==model and r['target']==target and r['variant']==variant]
def finish(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
 fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white')
 svg=out/(name+'.svg')
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)

def plot(data,out):
 out.mkdir(exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':11,'figure.titlesize':15,
   'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
 rows=read(data/'runs.json');baselines=read(data/'baseline_runs.json');comparisons=read(data/'comparisons.json')
 native=[b for b in baselines if b['platform']=='polus'];cells=[('gcn',.3),('gcn',.5),('graphsage',.3),('graphsage',.5)]
 fig,axes=plt.subplots(2,2,figsize=(14,10))
 for ax,(model,target) in zip(axes.flat,cells):
  for variant,color in zip(ORDER,COLORS):
   group=select(rows,model,target,variant);x=[r['search_seconds'] for r in group];y=[100*r['validation_accuracy'] for r in group]
   ax.scatter(x,y,c=color,s=18,alpha=.35)
   ax.errorbar(np.mean(x),np.mean(y),yerr=mean_sd(y)[1],fmt='o',color=color,capsize=3,markersize=6,zorder=3)
  for variant,marker in [('greedy_v4_menu12','P'),('greedy_v4_percent','*')]:
   r=select(rows,model,target,variant)[0];ax.scatter(r['search_seconds'],100*r['validation_accuracy'],marker=marker,s=115,c='black',zorder=4)
  for variant,marker in [('q_reachable_v3','D'),('greedy_reachable_v3','^')]:
   r=select(native,model,target,variant)[0];ax.scatter(r['search_seconds'],100*r['validation_accuracy'],marker=marker,s=75,facecolors='none',edgecolors='black',zorder=5)
  ax.set_xscale('log');ax.set_xlabel('Recorded search cost (s, log scale)');ax.set_ylabel('Validation accuracy (%)')
  ax.set_title('{} | target {:.0%}'.format(model.upper(),target));ax.grid(True,alpha=.2)
 handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in zip(COLORS,LABELS)]
 handles += [Line2D([],[],marker=m,linestyle='',color='black',markerfacecolor='black' if v else 'none',label=l)
  for m,v,l in [('P',True,'Greedy A12 (n=1)'),('*',True,'Greedy A% (n=1)'),('D',False,'V3 Q / POLUS101'),('^',False,'V3 Greedy / POLUS101')]]
 fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,.025),frameon=False)
 fig.suptitle('V4 quality versus recorded search cost | one frozen training checkpoint per model')
 fig.text(.5,.005,'Faint points: policy seeds; large Q points: mean of 3; vertical bars: sample SD. Width/compression differences remain; see runs.csv.',ha='center',fontsize=10)
 fig.subplots_adjust(bottom=.23,top=.9,hspace=.3,wspace=.23);finish(fig,out,'quality_cost')

 fig,axes=plt.subplots(2,2,figsize=(14,10))
 for ax,(model,target) in zip(axes.flat,cells):
  for i,variant in enumerate(ORDER):
   group=[c for c in comparisons if c['scope']=='POLUS_matched_seed' and c['model']==model and c['target']==target and c['treatment']==variant]
   for metric,offset,color in [('accuracy_delta_pp',-.13,'#0072B2'),('f1_delta_pp',.13,'#D55E00')]:
    mu,sd=mean_sd([r[metric] for r in group]);ax.errorbar(mu,i+offset,xerr=sd,fmt='o',color=color,capsize=3)
  ax.axvline(0,color='black',lw=1);ax.set_yticks(range(7));ax.set_yticklabels(LABELS);ax.invert_yaxis()
  ax.set_xlim(-12,12);ax.set_xlabel('Treatment minus correct control (percentage points)');ax.grid(axis='x',alpha=.2)
  ax.set_title('{} | target {:.0%}'.format(model.upper(),target))
 fig.legend(handles=[Line2D([],[],marker='o',color=c,linestyle='',label=l) for c,l in [('#0072B2','Accuracy'),('#D55E00','Macro-F1')]],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.025))
 fig.suptitle('Controlled POLUS comparisons: C versus B; other factors versus V3 constrained Q')
 fig.text(.5,.005,'C: mean paired difference +/- sample SD across 3 seeds. All other contrasts: seed101 only (n=1), no uncertainty estimate.',ha='center',fontsize=10)
 fig.subplots_adjust(bottom=.14,top=.9,hspace=.32,wspace=.48);finish(fig,out,'controlled_effects')

 fig,axes=plt.subplots(2,2,figsize=(14,9))
 for ax,(model,target) in zip(axes.flat,cells):
  cover=[];cover_sd=[];unseen=[]
  for variant in ORDER:
   group=select(rows,model,target,variant);mu,sd=mean_sd([100*r['q_coverage'] for r in group]);cover.append(mu);cover_sd.append(sd)
   unseen.append(np.mean([100*r['deployment_unseen_fraction'] for r in group]))
  ax.bar(range(7),cover,yerr=cover_sd,capsize=2,color='#4C78A8',alpha=.8)
  ax.plot(range(7),unseen,'D-',color='#D55E00',markersize=4,label='Unseen choices')
  baseline=select(native,model,target,'q_reachable_v3')[0];ax.axhline(100*baseline['q_coverage'],color='black',ls='--',lw=1)
  ax.set_ylim(0,100);ax.set_xticks(range(7));ax.set_xticklabels(SHORT,rotation=25,ha='right')
  ax.set_ylabel('Coverage / unseen choices (%)');ax.set_title('{} | target {:.0%}'.format(model.upper(),target));ax.grid(axis='y',alpha=.2)
 fig.legend(handles=[plt.Rectangle((0,0),1,1,color='#4C78A8',label='Visited-cell Q coverage'),
  Line2D([],[],marker='D',color='#D55E00',label='Unseen deployment action fraction'),Line2D([],[],color='black',ls='--',label='V3 Q101 coverage (n=1)')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.035))
 fig.suptitle('Coverage and unseen choices | V4 means of 3 policy seeds')
 fig.text(.5,.005,'Coverage denominator: legal cells in visited aggregate states; varies by menu. Unseen denominator: deployment decisions. Coverage bars +/- sample SD.',ha='center',fontsize=10)
 fig.subplots_adjust(bottom=.2,top=.89,hspace=.36,wspace=.24);finish(fig,out,'coverage_and_unseen')

 with (data/'learning_blocks.csv').open() as f:blocks=list(csv.DictReader(f))
 fig,axes=plt.subplots(2,2,figsize=(14,9))
 for ax,(model,target) in zip(axes.flat,cells):
  for variant,color,label in [('q_v4_300','#0072B2','B: constant epsilon .2'),('q_v4_decay300','#E69F00','C: epsilon .4 -> .05')]:
   xs=[];ys=[];sds=[]
   for start in range(0,300,25):
    group=[float(b['validation_accuracy_mean'])*100 for b in blocks if b['model']==model and float(b['target'])==target and b['variant']==variant and int(b['episode_start'])==start]
    assert len(group)==3;mu,sd=mean_sd(group);xs.append(start+12);ys.append(mu);sds.append(sd)
   xs=np.array(xs);ys=np.array(ys);sds=np.array(sds);ax.plot(xs,ys,color=color,label=label);ax.fill_between(xs,ys-sds,ys+sds,color=color,alpha=.15)
  ax.axvline(100,color='gray',ls='--',lw=1);ax.set_xlabel('Training episode');ax.set_ylabel('Training-end validation accuracy (%)')
  ax.set_title('{} | target {:.0%}'.format(model.upper(),target));ax.grid(alpha=.2)
 axes[0,0].legend(frameon=False,fontsize=9)
 fig.suptitle('Exploring training endpoints are not greedy deployment evaluations')
 fig.text(.5,.005,'25-episode blocks; band = sample SD of 3 policy-seed block averages, not confidence intervals. No extra policy rollout was performed.',ha='center',fontsize=10)
 fig.subplots_adjust(bottom=.12,top=.9,hspace=.3,wspace=.24);finish(fig,out,'learning_endpoints')
 (out/'plot_runtime.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,matplotlib=matplotlib.__version__,numpy=np.__version__,purpose='Read verified derived data only; no model/dataset/search access.'),indent=2))
 print('Rendered4 figures in PNG and SVG.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();plot(a.data,a.output)
