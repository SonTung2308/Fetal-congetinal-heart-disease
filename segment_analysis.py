"""Phan tich da-doan: video co nhieu doan benh thi selector bat duoc bao nhieu doan?
- Segment = run lien tuc frame benh (gop khe ho <= GAP).
- Quet nguong tau: do SEGMENT-RECALL (% doan bat duoc) va SET-PRECISION (% frame chon la benh)."""
import numpy as np, h5py
fs=h5py.File('data/kvasir_video_score_full.h5','r')
fl=h5py.File('data/kvasir_video_fulllabels.h5','r')
GAP=45   # gop 2 doan benh cach nhau <=45 frame (~1.5s) thanh 1 doan
vids=[v for v in fs.keys() if fl[v]['gtscore'][...].sum()>0]
def smooth(p,w):
    k=np.ones(w)/w; return np.convolve(p,k,mode='same')
def segments(g):
    idx=np.where(g>0)[0]
    if len(idx)==0: return []
    segs=[[idx[0],idx[0]]]
    for i in idx[1:]:
        if i-segs[-1][1]<=GAP: segs[-1][1]=i
        else: segs.append([i,i])
    return segs
# cau truc da-doan
nseg=[len(segments(fl[v]['gtscore'][...])) for v in vids]
print('So video benh:',len(vids),'| so DOAN benh: TB %.1f, max %d'%(np.mean(nseg),max(nseg)))
print('video >=2 doan:',sum(1 for x in nseg if x>=2),'| >=3 doan:',sum(1 for x in nseg if x>=3))
print()
# quet nguong: segment-recall + set-precision
allseg=[(v,s) for v in vids for s in segments(fl[v]['gtscore'][...])]
print('%6s %10s %10s %12s'%('tau','SegRecall','SetPrec','set_size_TB'))
for tau in [0.5,0.7,0.8,0.9,0.95,0.99]:
    hit=0; precs=[]; sizes=[]
    for v in vids:
        sc=smooth(fs[v]['score'][...],9); g=fl[v]['gtscore'][...]
        sel=sc>=tau
        sizes.append(int(sel.sum()))
        if sel.sum()>0: precs.append(g[sel].mean())
        for s in segments(g):
            if sel[s[0]:s[1]+1].any(): hit+=1
    print('%6.2f %9.0f%% %9.0f%% %12.0f'%(tau,hit/len(allseg)*100,np.mean(precs)*100 if precs else 0,np.mean(sizes)))
