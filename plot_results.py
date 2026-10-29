from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('research/results/benchmark.json').read_text());out=Path('docs/assets');out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f7f3','axes.facecolor':'#f6f7f3'})
models=['centroid','knn','mlp'];colors=['#a6b5a1','#69a88a','#096b59']
fig,ax=plt.subplots(figsize=(9,4.5));base=[x for x in r['summary'] if x['variant']=='full'];ax.bar(models,[x['accuracy_mean']*100 for x in base],yerr=[x['accuracy_sd']*100 for x in base],color=colors,capsize=6)
for i,x in enumerate(base):ax.text(i,x['accuracy_mean']*100+2,f"{x['accuracy_mean']*100:.1f}%",ha='center',fontweight='bold')
ax.set(ylim=(0,105),ylabel='Accuracy (%)',title='UCI hand postures · held-out user-ID evaluation');fig.text(.1,.01,'Mean ± SD across 14 IDs. Public Vicon data; not webcam or QOLDA measurements.',fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'benchmark.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5));variants=['full','no_count','no_pair_distances'];xx=np.arange(3)
for i,m in enumerate(models):ax.bar(xx+(i-1)*.23,[next(s for s in r['summary'] if s['model']==m and s['variant']==v)['macro_f1_mean'] for v in variants],width=.23,label=m,color=colors[i])
ax.set_xticks(xx,['All features','Without marker count','Without pair distances']);ax.set(ylabel='Macro F1',ylim=(0,1),title='Feature ablation · identical user-ID splits');ax.legend();fig.tight_layout();fig.savefig(out/'ablation.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
for m,c in zip(models,colors):
 values=[next(s for s in base if s['model']==m)['accuracy_mean']]+[np.mean([s['accuracy'] for s in r['synthetic_marker_dropout'] if s['model']==m and s['drop_markers']==d]) for d in [1,2]]
 ax.plot([0,1,2],np.array(values)*100,'o-',label=m,color=c)
ax.set(xticks=[0,1,2],xlabel='Artificially removed markers (minimum 3 retained)',ylabel='Accuracy (%)',title='Synthetic occlusion stress test · not measured camera failure');ax.legend();fig.tight_layout();fig.savefig(out/'robustness.png',dpi=170);plt.close(fig)
lines=['| Features | Model | Accuracy, mean ± SD | Macro F1 |','|---|---|---:|---:|']
for s in r['summary']:lines.append(f"| {s['variant']} | {s['model']} | {s['accuracy_mean']*100:.2f} ± {s['accuracy_sd']*100:.2f}% | {s['macro_f1_mean']:.3f} |")
Path('research/results/TABLES.md').write_text('\n'.join(lines)+'\n')
