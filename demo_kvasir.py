"""DEMO: chon 5 video, xuat SET OF FRAMES cua tung video (selector = diem fine-tune + lam muot),
kem NHAN (benh/binh thuong + ten lop). Luu anh + bang tom tat."""
import csv, os, numpy as np, h5py, cv2
from collections import defaultdict

SCORE='data/kvasir_video_score_full.h5'; LAB='data/kvasir_video_fulllabels.h5'
META='data/kvasir/metadata.csv'; VDIR='data/kvasir/labelled_videos'; OUT='demo_setofframes'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
TOPN=20          # so frame luu anh de xem
BUDGET=0.01      # set of frames = top 1% (diem van hanh dat LesRecall 95%)
W=9              # cua so lam muot

# ten lop benh theo frame
cls=defaultdict(dict)
for r in csv.DictReader(open(META),delimiter=';'):
    cls[r['video_id']][int(r['frame_number'])]=r['finding_class']

fs=h5py.File(SCORE,'r'); fl=h5py.File(LAB,'r')
# chon 5 video co benh, uu tien nhieu loai
pv=[v for v in fs.keys() if fl[v]['gtscore'][...].sum()>0]
pv=sorted(pv, key=lambda v: -int(fl[v]['gtscore'][...].sum()))[:5]

def smooth(p,w):
    k=np.ones(w)/w; return np.convolve(p,k,mode='same')

os.makedirs(OUT,exist_ok=True)
summary=[]
for v in pv:
    sc=smooth(fs[v]['score'][...],W); g=fl[v]['gtscore'][...]; n=len(g)
    npos=int(g.sum())
    lim=max(1,int(n*BUDGET))
    sel=np.argsort(sc)[-lim:]                 # SET OF FRAMES (top 1%)
    sel=np.sort(sel)
    n_benh=int(g[sel].sum())
    frec=n_benh/npos if npos else 0
    les=1 if n_benh>0 else 0
    summary.append((v,n,npos,lim,n_benh,frec,les))
    # luu TOP-N anh diem cao nhat de xem
    top=np.argsort(sc)[-TOPN:][::-1]
    vd=os.path.join(OUT,v); os.makedirs(vd,exist_ok=True)
    cap=cv2.VideoCapture(os.path.join(VDIR,v+'.mp4'))
    wantmap={int(fi):rank for rank,fi in enumerate(top)}
    idx=0; saved={}
    while idx<=max(top) and len(saved)<TOPN:
        ok,fr=cap.read()
        if not ok: break
        if idx in wantmap:
            lab=int(g[idx]); cname=cls[v].get(idx,'Normal/unlabeled' if lab==0 else 'benh')
            tag='BENH' if lab==1 else 'binhthuong'
            # ve nhan len anh
            col=(0,0,255) if lab==1 else (0,180,0)
            cv2.putText(fr,'%s | %s | score=%.2f'%(tag,cname,sc[idx]),(8,24),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
            fn='%02d_frame%06d_%s.jpg'%(wantmap[idx],idx,tag)
            cv2.imwrite(os.path.join(vd,fn),fr); saved[idx]=1
        idx+=1
    cap.release()
    # ghi bang chi tiet top-N
    with open(os.path.join(vd,'_labels.csv'),'w') as f:
        f.write('rank,frame_idx,score,label,finding_class\n')
        for rank,fi in enumerate(top):
            fi=int(fi); lab=int(g[fi])
            f.write('%d,%d,%.4f,%d,%s\n'%(rank,fi,sc[fi],lab,cls[v].get(fi,'-')))

print('=== DEMO: set of frames = top 1%% (lam muot w=%d), luu top %d anh/video ==='%(W,TOPN))
print('%-20s %8s %6s %8s %8s %8s %6s'%('video','n_frame','#benh','set_size','#benh_set','FrameRec','co_benh'))
for v,n,npos,lim,nb,fr,les in summary:
    print('%-20s %8d %6d %8d %8d %7.0f%% %6s'%(v[:20],n,npos,lim,nb,fr*100,'CO' if les else 'khong'))
print('\n[*] Anh + nhan luu tai: %s/<video>/  (ten file co BENH/binhthuong, kem _labels.csv)'%OUT)
