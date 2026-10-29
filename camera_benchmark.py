"""Evaluate actual research captures when supplied. No results without recordings.
Camera six-feature ablations; world vs screen uses the same complete cases.
Personal calibration is a separate condition using explicitly recorded reference
poses from each participant, including test participant (not zero-shot).
"""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.metrics import accuracy_score,f1_score
from benchmark import mlp_fit,infer
LABELS=['Open palm','Fist','Pinch','Point']
def main():
 p=argparse.ArgumentParser();p.add_argument('captures',nargs='+');p.add_argument('--output',default='research/results/camera-benchmark.json');a=p.parse_args();records=[];refs={};rng=np.random.default_rng(42)
 for path in a.captures:
  d=json.loads(Path(path).read_text())
  if d.get('schema')!='qolda-vision.capture.v1' or d.get('source')!='camera':raise ValueError('Expected real camera captures')
  user=d['meta']['participant'];task=d['meta']['task']
  for row in d['rows']:
   if not row['detected'] or row.get('features_world') is None:continue
   if task in ['calibration-open','calibration-closed']:refs.setdefault((user,task),[]).append(row['features_world']);continue
   if task in LABELS:records.append((user,LABELS.index(task)+1,row['features_world'],row['features_screen']))
 if not records:raise ValueError('No labeled camera observations with both world and screen features')
 users=sorted(set(r[0] for r in records))
 if len(users)<3:raise ValueError('Need at least three independent participants')
 sample=[]
 for u in users:
  for label in range(1,5):
   subset=[r for r in records if r[0]==u and r[1]==label]
   if not subset:raise ValueError(f'{u} lacks class {LABELS[label-1]}')
   sample.extend(subset[i] for i in rng.choice(len(subset),min(75,len(subset)),replace=False))
 u=np.array([r[0] for r in sample]);y=np.array([r[1] for r in sample]);world=np.array([r[2] for r in sample]);screen=np.array([r[3] for r in sample]);variants={'world':world,'screen':screen}
 for f in range(6):variants['world_without_feature_'+str(f)]=np.delete(world,f,axis=1)
 all_refs=all((user,t) in refs for user in users for t in ['calibration-open','calibration-closed'])
 if all_refs:
  calibrated=world.copy()
  for user in users:
   lo=np.median(refs[(user,'calibration-closed')],axis=0);hi=np.median(refs[(user,'calibration-open')],axis=0)
   if np.any(hi[:5]-lo[:5]<.1):raise ValueError(f'{user}: insufficient calibration range')
   calibrated[u==user,:5]=np.clip((world[u==user,:5]-lo[:5])/(hi[:5]-lo[:5]),0,1)
  variants['world_personal_calibration']=calibrated
 results=[]
 for user in users:
  tr=u!=user;te=~tr
  for variant,x in variants.items():
   scaler=StandardScaler().fit(x[tr]);train=scaler.transform(x[tr]);test=scaler.transform(x[te])
   models={'centroid':NearestCentroid().fit(train,y[tr]),'knn':KNeighborsClassifier(5).fit(train,y[tr]),'mlp':mlp_fit(train,y[tr],42,50)}
   for name,model in models.items():
    pred=infer(model,test);results.append({'test_participant':user,'variant':variant,'model':name,'accuracy':float(accuracy_score(y[te],pred)),'macro_f1':float(f1_score(y[te],pred,labels=[1,2,3,4],average='macro',zero_division=0))})
 out={'schema':'qolda.camera-benchmark.v1','sampled_rows':len(sample),'participants':users,'split':'leave-one-participant-out','fixed_seed':42,'complete_case_world_screen':True,'personal_calibration_available':all_refs,'calibration_caveat':'Uses separate reference poses from test participant; personalized, not zero-shot.','folds':results}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2));print(a.output)
if __name__=='__main__':main()
