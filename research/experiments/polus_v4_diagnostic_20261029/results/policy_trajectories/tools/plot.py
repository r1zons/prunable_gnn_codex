"""Plot recorded observations and derived diagnostics; no ML imports/execution."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'figures'
OUT.mkdir(exist_ok=True)
case = json.loads((ROOT/'case_study.json').read_text())
q, g, r = case['q_steps'], case['greedy_steps'], case['run']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.hashsalt': 'v4-recorded-trajectories'})
colors = ['#1764ab', '#d36b13']


def finish(fig, name):
    fig.savefig(str(OUT/(name+'.png')), dpi=180, bbox_inches='tight')
    path = OUT/(name+'.svg')
    fig.savefig(str(path), bbox_inches='tight', metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), constrained_layout=True)
for events, label, color, isq in [(q, 'Q, seed 5002', colors[0], True), (g, 'Greedy', colors[1], False)]:
    x = list(range(len(events)+1))
    acc = [100*r['dense_validation_accuracy']]+[100*e['validation_accuracy_after' if isq else 'validation_accuracy'] for e in events]
    widths = [[32,32]]+[e['widths_after'] for e in events]
    compression = [0]+[100*e['achieved_compression' if isq else 'sparsity'] for e in events]
    axes[0,0].plot(x, acc, 'o-', color=color, label=label, ms=4)
    for layer, style in [(0,'-'),(1,'--')]:
        axes[0,1].step(x, [w[layer] for w in widths], where='post', color=color,
                       linestyle=style, label=label+f', layer {layer}')
    axes[1,0].plot(x, compression, 'o-', color=color, ms=4, label=label)
axes[0,0].set(ylabel='Validation accuracy (%)', title='Recorded intermediate accuracy')
axes[0,0].legend(frameon=False)
axes[0,1].set(ylabel='Retained width', title='Architectures along each path')
axes[0,1].legend(frameon=False, fontsize=8)
axes[1,0].axhspan(29,31,color='#999999',alpha=.18,label='Exact target band')
axes[1,0].set(ylabel='Physical parameter reduction (%)', title='Exact achieved compression')
axes[1,0].legend(frameon=False, fontsize=8)
for ax in [axes[0,0],axes[0,1],axes[1,0]]:
    ax.set(xlabel='Successful pruning action count', xticks=range(0,11,2))
    ax.grid(alpha=.15)
ax = axes[1,1]
assert q[-1]['kept_ids_before'] == g[-2]['original_kept_ids']
g_removed = sorted(set(g[-2]['original_kept_ids'][1])-set(g[-1]['original_kept_ids'][1]))
names = ['Delete 1\n(channel '+str(g_removed[0])+')',
         'Delete 2\n(channels '+', '.join(map(str,q[-1]['removed_ids'][1]))+')']
qvals = [q[-1]['legal_q_values']['prune:1:1'], q[-1]['legal_q_values']['prune:1:2']]
rewards = [g[-1]['reward'], q[-1]['reward']]
positions = np.arange(2)
ax.barh(positions+.15, qvals, .28, color=colors[0],label='Stored Q estimate')
ax.barh(positions-.15, rewards, .28, color='#777777',label='Observed terminal reward')
for i, action in enumerate(['prune:1:1','prune:1:2']):
    visits = q[-1]['legal_action_training'][action]['observations']
    ax.text(qvals[i]+.0003, i+.15, f'{visits} updates', va='center', fontsize=8)
ax.set(yticks=positions,yticklabels=names, xlabel='Reward / Q units',
       title='Final shared state: (23,17), 11 actions left')
ax.axvline(0,color='black',lw=.7)
ax.legend(loc='lower left',frameon=False,fontsize=8)
fig.suptitle('GraphSAGE 30%: width-percent menu, frozen training checkpoint 101\n'
             'First actions differ; both paths return to the same physical state before action 10',fontsize=13)
finish(fig,'paired_trajectory')

rows = list(csv.DictReader((ROOT/'run_diagnostics.csv').open()))
variants = ['q_v4_menu12','q_v4_percent','q_v4_300','q_v4_decay300',
            'q_v4_random_ties','q_v4_incremental_drop','q_v4_gamma0']
labels = ['A {1,2}','A width %','B 300','C decay300','D random ties','E incremental*','F gamma0']
fig, axes = plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
norm = TwoSlopeNorm(vmin=-11,vcenter=0,vmax=3)
for ax, (model,target) in zip(axes.flat, [('gcn',.3),('gcn',.5),('graphsage',.3),('graphsage',.5)]):
    grid = np.zeros((7,3))
    for i, variant in enumerate(variants):
        for j, seed in enumerate([101,5001,5002]):
            row = next(r for r in rows if r['model']==model and float(r['target'])==target and r['variant']==variant and int(r['policy_seed'])==seed)
            grid[i,j] = float(row['accuracy_delta_pp'])
            value = f"{row['deployment_training_alias_exposure']}/{row['deployment_steps']}"
            if row['deployment_cross_mask_value_transfer'] != '0':
                value += ' +'
            ax.text(j,i,value,ha='center',va='center',fontsize=10,color='white' if grid[i,j]<-6 else 'black')
    im = ax.imshow(grid,cmap='RdBu',norm=norm,aspect='auto')
    ax.set(xticks=range(3),xticklabels=['101','5001','5002'],yticks=range(7),yticklabels=labels,
           title=f'{model.upper()} target {int(100*target)}%',xlabel='POLUS policy seed')
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which='minor',color='white',lw=1)
    ax.tick_params(which='minor',bottom=False,left=False)
fig.colorbar(im,ax=axes,shrink=.85,label='Validation accuracy difference from menu-matched Greedy (pp)')
fig.suptitle('All 84 policies: training-aliased deployment steps / total steps\n'
             'Color: paired accuracy difference; +: value learned only from other masks\n'
             '* E shares menu and inputs with Greedy, but has a different reward',fontsize=12)
finish(fig,'cross_run_diagnostics')
(ROOT/'figure_runtime.json').write_text(json.dumps(dict(python=__import__('sys').version.split()[0],
    matplotlib=matplotlib.__version__,numpy=np.__version__,scope='Recorded data only'),indent=2)+'\n')
