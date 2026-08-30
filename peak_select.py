"""So chon TOP-K% (global) vs chon theo DINH CUC BO (moi vung 1 dinh).
Muc tieu: bat nhieu DOAN hon o ngan sach thap."""
import numpy as np, h5py
from scipy.signal import find_peaks
fs=h5py.File('data/kvasir_video_score_full.h5','r'); fl=h5py.File('data/kvasir_video_fulllabels.h5','r')
GAP=45; vids=[v for v in fs.keys() if fl[v]['gtscore'][...].sum()>0]
def smooth(p,w): k=np.ones(w)/w; return np.convolve(p,k,mode='same')
def segments(g):
    idx=np.where(g>0)[0]
    if len(idx)==0: return []
    segs=[[idx[0],idx[0]]]
    for i in idx[1:]:
        if i-segs[-1][1]<=GAP: segs[-1][1]=i
        else: segs.append([i,i])
    return segs
allseg=[(v,s) for v in vids for s in segments(fl[v]['gtscore'][...])]
def segrecall(sel_of):
    hit=0
    for v in vids:
        sel=sel_of(v); g=fl[v]['gtscore'][...]
        for s in segments(g):
            if sel[s[0]:s[1]+1].any(): hit+=1
    return hit/len(allseg)*100
print('%18s %10s %12s'%('phuong phap','SegRecall','set_TB'))
# global top-k
for kb in [5,10,20]:
    def sel_of(v,kb=kb):
        sc=smooth(fs[v]['score'][...],9); n=len(sc); lim=int(n*kb/100)
        s=np.zeros(n,bool); s[np.argsort(sc)[-lim:]]=True; return s
    sizes=[int(sel_of(v).sum()) for v in vids]
    print('%18s %9.0f%% %12.0f'%('global top-%d%%'%kb,segrecall(sel_of),np.mean(sizes)))
# peak-based: lay dinh cuc bo + cua so quanh dinh, so dinh = tang dan
for npk in [20,50,100]:
    def sel_of(v,npk=npk):
        sc=smooth(fs[v]['score'][...],9); n=len(sc)
        pk,_=find_peaks(sc,distance=GAP)              # dinh cach nhau >=GAP
        if len(pk)==0: pk=np.array([np.argmax(sc)])
        top=pk[np.argsort(sc[pk])[-npk:]]             # npk dinh cao nhat
        s=np.zeros(n,bool)
        for p in top: s[max(0,p-GAP//2):min(n,p+GAP//2)]=True   # cua so quanh dinh
        return s
    sizes=[int(sel_of(v).sum()) for v in vids]
    print('%18s %9.0f%% %12.0f'%('peak x%d'%npk,segrecall(sel_of),np.mean(sizes)))
