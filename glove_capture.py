"""Record newline-delimited serial JSON using the same computer as the camera.
Input contract: {"values":{"thumb":0.2,"index":0.3,"ring":0.1,"little":0.4}}.
Values are normalized flexion proxies by default, not degrees.
Existing firmware needs an explicit adapter; raw ADC values are not angles.
"""
import argparse,json,time
from pathlib import Path

def validate_values(message):
    values=message.get('values',{})
    if not values or any(k not in ['thumb','index','ring','little'] for k in values):raise ValueError('Expected named finger values')
    if any(not isinstance(v,(int,float)) or not 0<=v<=1 for v in values.values()):raise ValueError('Expected normalized flexion proxies in [0,1]')
    return values

def main():
    import serial
    p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--baud',type=int,default=115200);p.add_argument('--seconds',type=float,default=60);p.add_argument('--output',default='glove.json');a=p.parse_args();rows=[];rejected=0
    with serial.Serial(a.port,a.baud,timeout=.2) as connection:
        end=time.monotonic()+a.seconds
        while time.monotonic()<end:
            line=connection.readline();now=time.time_ns()/1e6
            if not line:continue
            try:rows.append({'time_ms':now,'values':validate_values(json.loads(line))})
            except (ValueError,TypeError):rejected+=1
    Path(a.output).write_text(json.dumps({'schema':'qolda.glove.v1','unit':'proxy','clock':'host epoch ms at serial receipt','rejected_lines':rejected,'rows':rows},indent=2));print(len(rows),'observations;',rejected,'rejected lines')
if __name__=='__main__':main()
