"""Pha A: trich cac frame trong DOAN ton thuong (lan nhan) tu video, luu lam positive bo sung.
Lan nhan: moi doan = tu frame benh dau den frame benh cuoi (gop khe ho <=GAP) deu = benh."""
import csv, glob, os, numpy as np, h5py, cv2
from collections import defaultdict
LAB='data/kvasir_video_fulllabels.h5'; VDIR='data/kvasir/labelled_videos'
OUTDIR='data/kvasir_curated_pos'; GAP=45
os.makedirs(OUTDIR,exist_ok=True)
fl=h5py.File(LAB,'r')
def segments(g):
    idx=np.where(g>0)[0]
    if len(idx)==0: return []
    segs=[[idx[0],idx[0]]]
    for i in idx[1:]:
        if i-segs[-1][1]<=GAP: segs[-1][1]=i
        else: segs.append([i,i])
    return segs
vids=[v for v in fl.keys() if fl[v]['gtscore'][...].sum()>0]
tot=0
for vi,v in enumerate(vids):
    vd=os.path.join(OUTDIR,v)
    if os.path.isdir(vd) and len(os.listdir(vd))>0:
        print('skip',v); continue
    os.makedirs(vd,exist_ok=True)
    g=fl[v]['gtscore'][...]
    want=set()
    for s in segments(g):
        for i in range(s[0],s[1]+1): want.add(i)   # lan day trong doan
    cap=cv2.VideoCapture(os.path.join(VDIR,v+'.mp4')); idx=0; saved=0
    mx=max(want) if want else -1
    while idx<=mx:
        ok,fr=cap.read()
        if not ok: break
        if idx in want:
            cv2.imwrite(os.path.join(vd,'f%06d.jpg'%idx),fr); saved+=1
        idx+=1
    cap.release(); tot+=saved
    print('[%d/%d] %s: %d frame benh (lan nhan)'%(vi+1,len(vids),v,saved),flush=True)
print('TONG',tot,'positive frame lan nhan ->',OUTDIR)
