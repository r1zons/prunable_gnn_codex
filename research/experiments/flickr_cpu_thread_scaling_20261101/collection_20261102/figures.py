"""Render stored native measurements only; no experiment/model imports."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(csv_path,output):
    with csv_path.open() as f:rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'flickr-c4-threads-20261102'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for target,color,marker in [(.3,'#2876ac','o'),(.5,'#b04b59','s')]:
        selected=[r for r in rows if float(r['target'])==target]
        x=[int(r['threads']) for r in selected]
        axes[0].plot(x,[float(r['search_seconds']) for r in selected],color=color,marker=marker,label=f'Target {target:.0%}')
        axes[1].plot(x,[float(r['forecast_minutes']) for r in selected],color=color,marker=marker,label=f'Target {target:.0%}')
    axes[0].set(title='Native technical Q search',ylabel='Seconds (3 episodes + deployment prefix)')
    axes[1].axhline(170,color='#536944',ls='--',lw=1,label='Acceptance: 170 min')
    axes[1].axhline(180,color='#986a32',ls=':',lw=1.4,label='LSF limit: 180 min')
    axes[1].set(title='Conservative 30-episode forecast',ylabel='Predicted minutes (not a full run)')
    for ax in axes:
        ax.set_xticks([4,8,12,20]);ax.set_xlabel('CPU threads');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('C4 · frozen Flickr GraphSAGE 4/256 · identical recorded policies across thread counts',fontsize=11)
    output.mkdir(exist_ok=True)
    fig.savefig(output/'thread_scaling.png',dpi=180,metadata={'Software':'matplotlib'})
    fig.savefig(output/'thread_scaling.svg',metadata={'Date':None});plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for target,color,marker in [(.3,'#2876ac','o'),(.5,'#b04b59','s')]:
        selected=[r for r in rows if float(r['target'])==target];x=[int(r['threads']) for r in selected]
        for ax,label in zip(axes,['dense','pruned']):
            ax.errorbar(x,[float(r[f'forward_{label}_median']) for r in selected],
                yerr=[float(r[f'forward_{label}_sd']) for r in selected],color=color,marker=marker,
                capsize=3,label=f'Process for target {target:.0%}')
    for ax,title in zip(axes,['Dense widths [256, 256, 256]','Fixed pruned widths [111, 230, 219]']):
        ax.set(title=title,ylabel='Median full-batch forward (seconds)',xlabel='CPU threads',xticks=[4,8,12,20])
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Uncached forwards · 2 warmups + 7 measurements · bars show within-process sample SD',fontsize=11)
    fig.savefig(output/'forward_scaling.png',dpi=180,metadata={'Software':'matplotlib'})
    fig.savefig(output/'forward_scaling.svg',metadata={'Date':None});plt.close(fig)
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.csv,a.output)
