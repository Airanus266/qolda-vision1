"""Verify published metrics and checkpoints against recorded predictions."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from benchmark import fetch,load_data,infer,metrics
root=Path('research/results');r=json.loads((root/'benchmark.json').read_text());preds=json.loads((root/'predictions.json').read_text());path=fetch(Path('research/data/Postures.csv'))
assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
x,y,u,clouds,indices,total=load_data(path,r['cap_per_user_class'],r['seed']);assert indices.tolist()==json.loads((root/'sample_indices.json').read_text());assert len(set(indices))==len(indices)==r['sampled_rows']
for split in r['splits']:
 test=split['test_user'];val=split['validation_user'];assert test!=val and not {test,val}&set(split['train_users']);mask=u==test
 ck=torch.load(root/f'mlp_holdout_{test}.pt',map_location='cpu',weights_only=True);assert ck['test_user']==test and ck['train_users']==split['train_users']
 model=nn.Sequential(nn.Linear(18,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,5));model.load_state_dict(ck['state_dict']);model.eval()
 output=infer(model,(x[mask]-np.array(ck['scaler_mean']))/np.array(ck['scaler_scale']))
 for name in ['centroid','knn','mlp']:
  rows=[p for p in preds if p['test_user']==test and p['model']==name];assert [p['source_index'] for p in rows]==indices[mask].tolist();assert [p['truth'] for p in rows]==y[mask].tolist()
  if name=='mlp':assert [p['prediction'] for p in rows]==output.tolist()
  result=metrics([p['truth'] for p in rows],[p['prediction'] for p in rows]);stored=next(f for f in r['folds'] if f['test_user']==test and f['model']==name and f['variant']=='full')
  for key,value in result.items():assert abs(stored[key]-value)<1e-12
for s in r['summary']:
 rows=[f for f in r['folds'] if f['model']==s['model'] and f['variant']==s['variant']]
 assert np.isclose(s['accuracy_mean'],np.mean([f['accuracy'] for f in rows]));assert np.isclose(s['accuracy_sd'],np.std([f['accuracy'] for f in rows],ddof=1));assert np.isclose(s['macro_f1_mean'],np.mean([f['macro_f1'] for f in rows]))
print('Verified source hash, sample indices, disjoint splits, all saved predictions, summary metrics and 14 MLP checkpoints.')
