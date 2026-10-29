"""Plot actual per-recording camera reliability by recorded condition."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('analysis');p.add_argument('--output',default='research/results/camera-conditions.png');a=p.parse_args()
rows=json.loads(Path(a.analysis).read_text())
if not rows:raise ValueError('No actual camera analyses supplied')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for col,key in enumerate(['lighting','view']):
 for row,(metric,title) in enumerate([('missing_hand_rate','Missing-hand rate'),('static_frame_displacement_screen_rms','Static displacement (screen units/frame)')]):
  groups=defaultdict(list)
  for r in rows:
   if r.get(metric) is not None:groups[r['meta'][key]].append(r[metric])
  ax=axes[row,col]
  if groups:
   keys=sorted(groups);ax.bar(keys,[np.mean(groups[k]) for k in keys],color='#218970')
   for i,k in enumerate(keys):
    ax.scatter(np.full(len(groups[k]),i),groups[k],color='black',s=12);ax.text(i,np.mean(groups[k]),f' n={len(groups[k])}',ha='center',va='bottom')
  else:ax.text(.5,.5,'No eligible recordings',ha='center',transform=ax.transAxes)
  ax.set_title(title+' by '+key);ax.set_ylim(bottom=0)
fig.suptitle('Actual camera recordings • dots = recordings, n ≠ independent participants\nStatic displacement contains real movement; not pure tracking noise',fontsize=11);fig.tight_layout();Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);print(a.output)
