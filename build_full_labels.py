"""Tao nhan cho MOI frame cua 43 video (khong decode, chi lay frame count).
Xuat data/kvasir_video_fulllabels.h5: frame_idx=0..n-1, gtscore theo metadata."""
import csv, glob, os, numpy as np, h5py, cv2
from collections import defaultdict
META='data/kvasir/metadata.csv'; VDIR='data/kvasir/labelled_videos'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
lab=defaultdict(dict)
for r in csv.DictReader(open(META),delimiter=';'):
    lab[r['video_id']][int(r['frame_number'])]=1 if r['finding_class'] in PATH else 0
vids=sorted(glob.glob(VDIR+'/*.mp4'))
fout=h5py.File('data/kvasir_video_fulllabels.h5','w')
tot=0; totp=0
for vp in vids:
    vid=os.path.splitext(os.path.basename(vp))[0]
    n=int(cv2.VideoCapture(vp).get(cv2.CAP_PROP_FRAME_COUNT))
    vl=lab.get(vid,{})
    g=np.array([vl.get(i,0) for i in range(n)],dtype=np.float32)
    grp=fout.create_group(vid)
    grp.create_dataset('frame_idx',data=np.arange(n,dtype=np.int32))
    grp.create_dataset('gtscore',data=g)
    tot+=n; totp+=int(g.sum())
    print(vid,n,'frame,',int(g.sum()),'benh',flush=True)
fout.close()
print('TONG',tot,'frame |',totp,'benh (%.2f%%)'%(totp/tot*100))
