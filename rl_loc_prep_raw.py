"""End-to-end (KHÔNG detector, KHÔNG feature pretrain) — Bước 1: load ảnh THÔ 84x84 + GT (thang 84).
CNN của agent sẽ học từ số 0 chỉ bằng reward. Lưu img(uint8 84x84x3), gt(4), vid."""
import csv, glob, os, numpy as np, h5py
from PIL import Image
SRC=336; W=84
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob('data/kvasir/labelled_images/**/*.jpg',recursive=True)}
imgs=[];gts=[];vids=[]
for r in csv.DictReader(open('data/kvasir/metadata.csv'),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in('x1','x2','x3','x4')];ys=[int(r[k]) for k in('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(SRC,max(xs)),min(SRC,max(ys))]
        if b[2]-b[0]<4 or b[3]-b[1]<4: continue
        im=Image.open(paths[r['filename']]).convert('RGB').resize((W,W))
        imgs.append(np.asarray(im,np.uint8)); gts.append([c*W/SRC for c in b]); vids.append(r['video_id'])
imgs=np.stack(imgs); gts=np.array(gts,np.float32)
f=h5py.File('data/loc_raw.h5','w'); f.create_dataset('img',data=imgs); f.create_dataset('gt',data=gts)
f.create_dataset('vid',data=np.array(vids,dtype='S20')); f.close()
print('[*] xong',len(imgs),'ảnh 84x84 ->',imgs.shape,'| data/loc_raw.h5')
