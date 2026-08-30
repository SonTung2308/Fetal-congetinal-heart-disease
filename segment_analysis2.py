"""Do lai voi nhan LAP DAY trong doan (frame giua 2 frame benh cung doan = benh).
Tim ngan sach de SEGMENT-RECALL = 100% (bat het moi doan benh)."""
import numpy as np, h5py
fs=h5py.File('data/kvasir_video_score_full.h5','r'); fl=h5py.File('data/kvasir_video_fulllabels.h5','r')
GAP=45
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
def fill(g):  # lap day nhan trong moi doan
    gg=g.copy()
    for s in segments(g): gg[s[0]:s[1]+1]=1
    return gg

allseg=[(v,s) for v in vids for s in segments(fl[v]['gtscore'][...])]
print('=== nhan LAP DAY trong doan (khu nhieu) ===')
print('%8s %10s %10s %12s'%('budget','SegRecall','SetPrec','set_TB'))
for kb in [1,2,5,10,15,20,30]:
    hit=0; precs=[]; sizes=[]
    for v in vids:
        sc=smooth(fs[v]['score'][...],9); g=fl[v]['gtscore'][...]; gf=fill(g); n=len(g)
        lim=max(1,int(n*kb/100)); sel=np.zeros(n,bool); sel[np.argsort(sc)[-lim:]]=True
        sizes.append(lim); precs.append(gf[sel].mean())
        for s in segments(g):
            if sel[s[0]:s[1]+1].any(): hit+=1
    print('%7d%% %9.0f%% %9.0f%% %12.0f'%(kb,hit/len(allseg)*100,np.mean(precs)*100,np.mean(sizes)))
# tim ngan sach dat 100% segment recall
print('\nNgan sach nho nhat de bat HET moi doan (SegRecall=100%):')
for kb in [20,30,40,50,60,70,80]:
    hit=0
    for v in vids:
        sc=smooth(fs[v]['score'][...],9); g=fl[v]['gtscore'][...]; n=len(g)
        lim=max(1,int(n*kb/100)); sel=np.zeros(n,bool); sel[np.argsort(sc)[-lim:]]=True
        for s in segments(g):
            if sel[s[0]:s[1]+1].any(): hit+=1
    r=hit/len(allseg)*100
    print('  budget %2d%%: SegRecall %.0f%%'%(kb,r))
    if r>=100: break
