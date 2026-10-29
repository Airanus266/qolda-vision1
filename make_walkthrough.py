"""Generate a 60-second, explicitly synthetic walkthrough (requires ffmpeg)."""
import json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
raw=subprocess.check_output(['node','--input-type=module','-e',"import {syntheticHand,features} from './dist/core.js'; console.log(JSON.stringify(Array.from({length:360},(_,i)=>{const l=syntheticHand(i/6);return {l,f:features(l).extension}})))"])
frames=json.loads(raw);r=json.loads(Path('research/results/benchmark.json').read_text());fig=plt.figure(figsize=(10,5.625),dpi=96,facecolor='#101d24')
connections=[(0,1),(0,5),(5,9),(9,13),(13,17),(17,0)]+[(i+j,i+j+1) for i in [1,5,9,13,17] for j in range(3)]
def draw(i):
 fig.clear();t=i/6;section=int(t//15);fig.text(.06,.91,'QOLDA VISION  /  RESEARCH EDITION',color='#88e5c6',fontsize=19,weight='bold');fig.text(.06,.045,'SYNTHETIC VISUALIZATION + MEASURED PUBLIC-DATA RESULTS | No patient or glove study',color='#c7d5dc',fontsize=8)
 fig.text(.91,.91,f'{int(t):02d}/60',color='white',fontsize=10)
 if section<2:
  fig.text(.06,.82,['01  From landmarks to movement','02  Instrumentation for real experiments'][section],color='white',fontsize=17)
  ax=fig.add_axes([.05,.16,.43,.57],facecolor='#152b35');p=np.array([[v['x'],v['y']] for v in frames[i]['l']]);
  for a,b in connections:ax.plot(p[[a,b],0],p[[a,b],1],color='#75dfbd',lw=3)
  ax.scatter(p[:,0],p[:,1],c='white',s=14);ax.set(xlim=(.1,.95),ylim=(1,.05));ax.axis('off')
  if section==0:
   ax=fig.add_axes([.56,.28,.36,.4],facecolor='#101d24');ax.barh(['Thumb','Index','Middle','Ring','Little'],frames[i]['f'],color='#75dfbd');ax.set_xlim(0,1);ax.tick_params(colors='white');ax.set_xlabel('Synthetic extension proxy',color='white');ax.spines[['top','right','left']].set_visible(False)
  else:
   for j,txt in enumerate(['Record numerical landmarks','Include missing-hand frames','Tag light, view and distance','Analyze speed and RMS jerk','Pair camera + glove timestamps']):fig.text(.52,.67-j*.085,txt,color='white',fontsize=12)
   fig.text(.52,.19,'Real recordings still required',color='#ffcc88',fontsize=12)
 elif section==2:
  fig.text(.06,.82,'03  Trained models, identical held-user-ID splits',color='white',fontsize=16)
  ax=fig.add_axes([.13,.29,.75,.4],facecolor='#101d24');s=r['summary'][:3];ax.barh(['Centroid','kNN','PyTorch MLP'],[v['accuracy_mean']*100 for v in s],color=['#788b8e','#5ea792','#75dfbd']);ax.set_xlim(0,100);ax.tick_params(colors='white');ax.set_xlabel('Mean held-ID accuracy (%)',color='white');ax.spines[['top','right']].set_visible(False)
  for j,v in enumerate(s):ax.text(v['accuracy_mean']*100+1,j,f"{v['accuracy_mean']*100:.2f}%",va='center',color='white')
  fig.text(.13,.17,'UCI motion capture | 5,050 sampled records | 14 ID folds | seed 42',color='white',fontsize=11);fig.text(.13,.12,'Not webcam accuracy. Between-ID SD: 16.21 / 14.87 / 15.28 percentage points.',color='#ffcc88',fontsize=9)
 else:
  fig.text(.06,.82,'04  Reproduce, inspect, challenge',color='white',fontsize=18)
  lines=['MLP checkpoints + predictions + split manifests','Feature ablations and unknown-class rejection','Colab notebook and three-page technical report','Next evidence: camera/glove data and real reliability','Public benchmark does not establish clinical benefit']
  for j,txt in enumerate(lines):fig.text(.09,.69-j*.095,txt,color='white' if j<3 else '#ffcc88',fontsize=13)
 fig.add_axes([.06,.085,.88*(i+1)/360,.006],facecolor='#75dfbd').set(xticks=[],yticks=[])
anim=FuncAnimation(fig,draw,frames=360,interval=1000/6);anim.save('docs/assets/walkthrough-60s.mp4',writer=FFMpegWriter(fps=6,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','23','-movflags','+faststart']));plt.close(fig)
