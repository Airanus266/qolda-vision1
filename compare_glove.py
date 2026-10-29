import argparse,json
from pathlib import Path
from analysis import align_pairs,agreement
p=argparse.ArgumentParser();p.add_argument('camera');p.add_argument('glove');p.add_argument('--offset-ms',type=float,default=0);p.add_argument('--tolerance-ms',type=float,default=50);a=p.parse_args()
c=json.loads(Path(a.camera).read_text());g=json.loads(Path(a.glove).read_text())
if c.get('schema')!='qolda-vision.capture.v1' or c.get('source')!='camera' or g.get('schema')!='qolda.glove.v1' or g.get('unit')!='proxy':raise ValueError('Expected camera capture and normalized-proxy glove recording')
rows=[{'time_ms':r['time_ms'],'values':dict(zip(['thumb','index','ring','little'],[r['flexion_proxy'][i] for i in [0,1,3,4]]))} for r in c['rows'] if r['detected']]
pairs=align_pairs(rows,g['rows'],a.tolerance_ms,a.offset_ms);print(json.dumps(agreement(pairs),indent=2))
