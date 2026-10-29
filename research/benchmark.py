"""Real UCI405 subject-disjoint experiment. Run from repository root."""
from pathlib import Path
import argparse, csv, hashlib, json, sys, platform, time, urllib.request, zipfile
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import torch
from torch import nn

URL='https://archive.ics.uci.edu/static/public/405/motion+capture+hand+postures.zip'
FEATURES=['count','mean_x','mean_y','mean_z','std_x','std_y','std_z','radius_q25','radius_q50','radius_q75','pair_q10','pair_q25','pair_q50','pair_q75','pair_q90','eig_1','eig_2','eig_3']

def cloud_features(points):
    """Permutation-invariant features: marker indices do not identify fingers."""
    p=np.asarray(points,float); p=p[np.isfinite(p).all(axis=1)]
    if len(p)<3: raise ValueError('At least three visible markers required')
    radii=np.linalg.norm(p,axis=1); ev=np.linalg.eigvalsh(np.cov(p.T))
    return np.r_[len(p),p.mean(0),p.std(0),np.quantile(radii,[.25,.5,.75]),np.quantile(pdist(p),[.1,.25,.5,.75,.9]),ev]

def fetch(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        archive=path.parent/'uci405.zip';urllib.request.urlretrieve(URL,archive)
        with zipfile.ZipFile(archive) as z: path.write_bytes(z.read('Postures.csv'))
    return path

def load_data(path,cap,seed):
    a=np.genfromtxt(path,delimiter=',',skip_header=1,missing_values='?',filling_values=np.nan)
    original=a[a[:,0]>0]; rng=np.random.default_rng(seed); indices=[]
    for user in np.unique(original[:,1]):
        for cls in range(1,6):
            idx=np.flatnonzero((original[:,1]==user)&(original[:,0]==cls))
            indices.extend(rng.choice(idx,min(cap,len(idx)),replace=False).tolist())
    indices=np.array(sorted(indices));a=original[indices]
    clouds=a[:,2:].reshape(-1,12,3)
    return np.stack([cloud_features(p) for p in clouds]),a[:,0].astype(int),a[:,1].astype(int),clouds,indices,len(original)

def mlp_fit(x,y,seed,epochs):
    torch.manual_seed(seed);torch.set_num_threads(1)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,5))
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(y-1,dtype=torch.long)
    for _ in range(epochs):
        for ids in torch.randperm(len(tx)).split(128):
            opt.zero_grad();loss=nn.functional.cross_entropy(model(tx[ids]),ty[ids]);loss.backward();opt.step()
    model.eval();return model

def infer(model,x):
    if isinstance(model,nn.Module):
        with torch.no_grad():return model(torch.tensor(x,dtype=torch.float32)).argmax(1).numpy()+1
    return model.predict(x)

def metrics(y,p):return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=[1,2,3,4,5],average='macro',zero_division=0))}

def run(args):
    start=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    path=fetch(Path(args.data));x,y,u,clouds,indices,total=load_data(path,args.cap,args.seed)
    users=np.unique(u);variants={'full':np.arange(18),'no_count':np.arange(1,18),'no_pair_distances':np.r_[0:10,15:18]}
    folds=[];opens=[];robust=[];predictions=[];splits=[]
    for j,test_user in enumerate(users):
        val_user=users[(j+1)%len(users)]
        tr=(u!=test_user)&(u!=val_user);va=u==val_user;te=u==test_user
        splits.append({'test_user':int(test_user),'validation_user':int(val_user),'train_users':users[~np.isin(users,[test_user,val_user])].tolist()})
        for variant,cols in variants.items():
            scaler=StandardScaler().fit(x[tr][:,cols]);a=scaler.transform(x[tr][:,cols]);b=scaler.transform(x[te][:,cols])
            models={'centroid':NearestCentroid().fit(a,y[tr]),'knn':KNeighborsClassifier(5).fit(a,y[tr]),'mlp':mlp_fit(a,y[tr],args.seed+j,args.epochs)}
            for name,model in models.items():
                p=infer(model,b);folds.append({'test_user':int(test_user),'variant':variant,'model':name,'n_test':int(te.sum()),**metrics(y[te],p)})
                if variant=='full':
                    predictions.extend({'test_user':int(test_user),'source_index':int(i),'truth':int(t),'prediction':int(g),'model':name} for i,t,g in zip(indices[te],y[te],p))
                if name=='mlp' and variant=='full':
                    torch.save({'state_dict':model.state_dict(),'input_dim':18,'classes':[1,2,3,4,5],'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'test_user':int(test_user),'train_users':splits[-1]['train_users']},out/f'mlp_holdout_{test_user}.pt')
            if variant=='full':
                rng=np.random.default_rng(args.seed+j)
                for drop in [1,2]:
                    degraded=[]
                    for cloud in clouds[te]:
                        visible=cloud[np.isfinite(cloud).all(1)];n=max(3,len(visible)-drop)
                        degraded.append(cloud_features(visible[rng.choice(len(visible),n,replace=False)]))
                    degraded=scaler.transform(np.asarray(degraded))
                    for name,model in models.items():robust.append({'test_user':int(test_user),'drop_markers':drop,'model':name,**metrics(y[te],infer(model,degraded))})
        # Unknown class 5 never enters training or threshold calibration.
        known_tr=tr&(y!=5);known_va=va&(y!=5)
        sc=StandardScaler().fit(x[known_tr]);kn=KNeighborsClassifier(5).fit(sc.transform(x[known_tr]),y[known_tr])
        val_dist=kn.kneighbors(sc.transform(x[known_va]))[0].mean(1);threshold=float(np.quantile(val_dist,.95))
        test_dist=kn.kneighbors(sc.transform(x[te]))[0].mean(1);pred=kn.predict(sc.transform(x[te]));unknown=y[te]==5;reject=test_dist>threshold;known=~unknown;accepted=known&~reject
        opens.append({'test_user':int(test_user),'threshold':threshold,'unknown_class':5,'unknown_recall':float(reject[unknown].mean()) if unknown.any() else None,'known_false_reject':float(reject[known].mean()),'unknown_auroc':float(roc_auc_score(unknown,test_dist)) if unknown.any() and known.any() else None,'known_accuracy_no_rejection':float((pred[known]==y[te][known]).mean()),'accepted_known_accuracy':float((pred[accepted]==y[te][accepted]).mean()) if accepted.any() else None,'accepted_known_n':int(accepted.sum())})
        print(f'Finished held-out user ID {test_user}',flush=True)
    summary=[]
    for variant in variants:
        for model in ['centroid','knn','mlp']:
            subset=[r for r in folds if r['variant']==variant and r['model']==model]
            summary.append({'variant':variant,'model':model,'accuracy_mean':float(np.mean([r['accuracy'] for r in subset])),'accuracy_sd':float(np.std([r['accuracy'] for r in subset],ddof=1)),'macro_f1_mean':float(np.mean([r['macro_f1'] for r in subset]))})
    report={'dataset':'UCI Motion Capture Hand Postures (405)','doi':'10.24432/C5TG86','license':'CC BY 4.0','source_url':URL,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'original_non_dummy_rows':total,'sampled_rows':len(y),'user_ids':users.tolist(),'source_description_participants':12,'metadata_discrepancy':'UCI description states 12 users; downloaded CSV contains 14 unique user IDs. Identity mapping unresolved.','cap_per_user_class':args.cap,'seed':args.seed,'epochs':args.epochs,'features':FEATURES,'split':'one test user ID, next user ID validation, all others training; identical split for all models','summary':summary,'folds':folds,'open_set':opens,'synthetic_marker_dropout':robust,'splits':splits,'runtime_seconds':time.time()-start,'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},'limitations':['Public Vicon marker data, not MediaPipe or QOLDA recordings.','No original participant recruitment.','Marker dropout is synthetic, not measured lighting or camera failure.','No chronological timestamps; no temporal claims from this dataset.','Fixed hyperparameters and one seed; no confidence claims from repeated tuning.']}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False));(out/'predictions.json').write_text(json.dumps(predictions));(out/'sample_indices.json').write_text(json.dumps(indices.tolist()))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='research/data/Postures.csv');p.add_argument('--output',default='research/results');p.add_argument('--cap',type=int,default=75);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=50);run(p.parse_args())
