"""Pha A HEP: lan nhan chi +-3 frame quanh moi frame benh GOC (gan chac chan con ton thuong).
Khac ban GAP=45: KHONG lap day khe ho lon -> tranh dinh nen."""
import glob, os, numpy as np, h5py, cv2
LAB='data/kvasir_video_fulllabels.h5'; VDIR='data/kvasir/labelled_videos'
OUTDIR='data/kvasir_curated_narrow'; W=3
os.makedirs(OUTDIR,exist_ok=True); fl=h5py.File(LAB,'r')
vids=[v for v in fl.keys() if fl[v]['gtscore'][...].sum()>0]
tot=0
for vi,v in enumerate(vids):
    vd=os.path.join(OUTDIR,v)
    if os.path.isdir(vd) and len(os.listdir(vd))>0: print('skip',v); continue
    os.makedirs(vd,exist_ok=True)
    g=fl[v]['gtscore'][...]; pos=np.where(g>0)[0]
    want=set()
    for f in pos:
        for d in range(-W,W+1):
            j=f+d
            if 0<=j<len(g) and g[j]==0: want.add(int(j))   # chi lan sang frame CHUA co nhan
    if not want: print(v,'0'); continue
    cap=cv2.VideoCapture(os.path.join(VDIR,v+'.mp4')); idx=0; saved=0; mx=max(want)
    while idx<=mx:
        ok,fr=cap.read()
        if not ok: break
        if idx in want: cv2.imwrite(os.path.join(vd,'f%06d.jpg'%idx),fr); saved+=1
        idx+=1
    cap.release(); tot+=saved
    print('[%d/%d] %s: +%d frame (lan hep +-%d)'%(vi+1,len(vids),v,saved,W),flush=True)
print('TONG',tot,'frame lan hep ->',OUTDIR)
