"""Movement descriptors, condition reliability, and receipt-time alignment.
These functions quantify signals; they do not establish clinical validity.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter,find_peaks


def motion_metrics(time_ms,values):
    t=np.asarray(time_ms,float)/1000;x=np.asarray(values,float)
    if len(t)<11 or len(t)!=len(x) or not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Need 11 finite observations with strictly increasing timestamps')
    dt=float(np.median(np.diff(t)))
    if np.max(np.diff(t))>max(.3,dt*3):raise ValueError('Tracking gaps too large: split recording into contiguous segments')
    grid=np.arange(t[0],t[-1]+dt/2,dt);a=np.interp(grid,t,x);window=min(11,len(a) if len(a)%2 else len(a)-1)
    smooth=savgol_filter(a,window,3);v=np.gradient(smooth,dt);acc=np.gradient(v,dt);jerk=np.gradient(acc,dt)
    peaks,_=find_peaks(smooth,prominence=.1,distance=max(1,int(.5/dt)))
    periods=np.diff(grid[peaks]);peak_values=smooth[peaks]
    return {'unit':'normalized flexion proxy','duration_s':float(grid[-1]-grid[0]),'amplitude_p95_p05':float(np.quantile(smooth,.95)-np.quantile(smooth,.05)),'mean_absolute_speed_per_s':float(np.mean(np.abs(v))),'rms_jerk_per_s3':float(np.sqrt(np.mean(jerk**2))),'peak_count':int(len(peaks)),'cycle_period_cv':float(np.std(periods)/np.mean(periods)) if len(periods)>=2 else None,'peak_amplitude_sd':float(np.std(peak_values,ddof=1)) if len(peaks)>=2 else None,'smoothing':'uniform resample at median interval; Savitzky-Golay window<=11 order3; finite differences','limitation':'Jerk depends strongly on smoothing and sampling. These are experimental descriptors, not clinical scores.'}


def align_pairs(camera,glove,tolerance_ms=50,offset_ms=0):
    """Greedy nearest timestamp pairing, without reuse or interpolation over gaps."""
    c=sorted(camera,key=lambda r:r['time_ms']);g=sorted(glove,key=lambda r:r['time_ms']);used=set();pairs=[]
    times=np.array([r['time_ms']+offset_ms for r in g])
    for row in c:
        if not len(times):break
        at=np.searchsorted(times,row['time_ms']);candidates=[j for j in (at-1,at) if 0<=j<len(g) and j not in used]
        if not candidates:continue
        j=min(candidates,key=lambda j:abs(times[j]-row['time_ms']))
        delta=float(times[j]-row['time_ms'])
        if abs(delta)<=tolerance_ms:used.add(j);pairs.append((row,g[j],delta))
    return pairs


def agreement(pairs,unit='proxy'):
    if unit not in ('proxy','degrees'):raise ValueError('unit must be proxy or degrees')
    result={}
    for finger in ['thumb','index','ring','little']:
        xy=[(a['values'][finger],b['values'][finger]) for a,b,_ in pairs if finger in a['values'] and finger in b['values']]
        xy=np.array(xy,float)
        if len(xy)<3:continue
        if not np.isfinite(xy).all():raise ValueError('Nonfinite measurements')
        error=xy[:,0]-xy[:,1]
        result[finger]={'n':len(xy),'mae':float(np.abs(error).mean()),'bias':float(error.mean()),'rmse':float(np.sqrt(np.mean(error**2))),'correlation':float(np.corrcoef(xy.T)[0,1]) if np.std(xy[:,0])>0 and np.std(xy[:,1])>0 else None}
    return {'unit':unit,'paired_frames':len(pairs),'median_abs_time_offset_ms':float(np.median([abs(p[2]) for p in pairs])) if pairs else None,'fingers':result,'limitation':'Receipt-time alignment. A flex sensor and camera joint angle are not equivalent without joint-specific physical calibration.'}


def capture_analysis(doc):
    if doc.get('schema')!='qolda-vision.capture.v1' or doc.get('source')!='camera':raise ValueError('Expected camera research capture')
    rows=doc['rows'];valid=[r for r in rows if r['detected']];out={'meta':doc['meta'],'observations':len(rows),'missing_hand_rate':sum(not r['detected'] for r in rows)/len(rows) if rows else None,'inference_ms_median':float(np.median([r['inference_ms'] for r in rows])) if rows else None}
    # Jitter is meaningful only for instructed static holds; real motion is not noise.
    if doc['meta']['task']=='static-hold':
        delta=[]
        for a,b in zip(rows,rows[1:]):
            if a['detected'] and b['detected'] and 0<b['time_ms']-a['time_ms']<300:
                aa=np.array([[p['x'],p['y']] for p in a['screen']]);bb=np.array([[p['x'],p['y']] for p in b['screen']]);delta.append(np.mean(np.linalg.norm(bb-aa,axis=1)))
        out['static_frame_displacement_screen_rms']=float(np.sqrt(np.mean(np.square(delta)))) if delta else None
        out['jitter_caveat']='Contains actual involuntary movement; not pure model noise. Screen-coordinate fraction per observed frame.'
    if doc['meta']['task']=='open-close' and len(valid)>=11:
        try:out['motion']=motion_metrics([r['time_ms'] for r in valid],[np.mean(r['flexion_proxy'][1:]) for r in valid])
        except ValueError as e:out['motion_unavailable']=str(e)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--output',default='research/results/capture-analysis.json');a=p.parse_args()
    reports=[capture_analysis(json.loads(Path(f).read_text())) for f in a.files];Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(reports,indent=2));print(a.output)
