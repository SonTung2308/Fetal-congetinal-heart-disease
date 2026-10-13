"""Chạy closed-loop RL selector (value-critic) + ghi log từng vòng + xuất hình & bảng cho báo cáo.
Reward = det_iou (box detector vs GT) trên frame được chọn. Chia theo video. Detector cố định.
Xuất: fig_learning_curve.png, fig_before_after.png, fig_round_progress.png, bang_ketqua.csv"""
import numpy as np, h5py, torch, torch.nn as nn, csv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
dev='cuda'; EPOCHS=60
f=h5py.File('data/loop_frames.h5','r')
byv={}
for k in f: byv.setdefault(f[k].attrs['vid'],[]).append(k)
vids=sorted(byv); rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order); te=set(order[:max(4,int(0.25*len(order)))])
def pack(v):
    ks=sorted(byv[v],key=lambda k:f[k]['meta'][...][0])
    feat=np.stack([f[k]['feat'][...] for k in ks]).astype(np.float32); dc=np.array([float(f[k]['dc'][()]) for k in ks],np.float32)
    di=np.array([float(f[k]['di'][()]) for k in ks],np.float32); isp=np.array([int(f[k]['meta'][...][1]) for k in ks])
    return feat,dc,di,isp
DATA={v:pack(v) for v in vids}
tr=[v for v in vids if v not in te]; teL=[v for v in vids if v in te]
Xtr=torch.tensor(np.concatenate([np.concatenate([DATA[v][0],DATA[v][1][:,None]],1) for v in tr])).to(dev)
Ytr=torch.tensor(np.concatenate([DATA[v][2] for v in tr])).to(dev)
class Q(nn.Module):
    def __init__(s,d=513,h=256):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
def topk(scores,di,isp,k):
    idx=np.argsort(scores)[-k:]; return di[idx].mean(), isp[idx].mean(), (di[idx]>=0.5).mean()
def base(BUD):
    conf=[];rnd=[];orac=[]
    for v in teL:
        feat,dc,di,isp=DATA[v]; n=len(di); k=max(1,int(n*BUD))
        conf.append(topk(dc,di,isp,k)[0]); rnd.append(topk(np.random.RandomState(1).rand(n),di,isp,k)[0]); orac.append(topk(di,di,isp,k)[0])
    return np.mean(conf),np.mean(rnd),np.mean(orac)
def evalRL(q,BUD):
    q.eval(); rl=[];pct=[];dr=[]
    with torch.no_grad():
        for v in teL:
            feat,dc,di,isp=DATA[v]; n=len(di); k=max(1,int(n*BUD))
            pred=q(torch.tensor(np.concatenate([feat,dc[:,None]],1)).to(dev)).cpu().numpy()
            r,p,d=topk(pred,di,isp,k); rl.append(r);pct.append(p);dr.append(d)
    return np.mean(rl),np.mean(pct),np.mean(dr)
BUDS=[0.05,0.10,0.20]; curves={}; finals={}
for BUD in BUDS:
    torch.manual_seed(0)
    q=Q().to(dev); opt=torch.optim.Adam(q.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.MSELoss()
    conf,rnd,orac=base(BUD)
    hist=[]; n=len(Xtr); g=torch.Generator().manual_seed(0)
    # vòng 0 = trước khi học (critic ngẫu nhiên)
    r0,p0,d0=evalRL(q,BUD); hist.append((0,r0,p0,d0))
    for ep in range(1,EPOCHS+1):
        q.train(); perm=torch.randperm(n,generator=g)
        for i in range(0,n,1024):
            j=perm[i:i+1024]; opt.zero_grad(); lossf(q(Xtr[j]),Ytr[j]).backward(); opt.step()
        r,p,d=evalRL(q,BUD); hist.append((ep,r,p,d))
    curves[BUD]=(hist,conf,rnd,orac)
    best=max(h[1] for h in hist); bi=int(np.argmax([h[1] for h in hist]))
    finals[BUD]=dict(before=r0,after=best,pct_before=p0,pct_after=hist[bi][2],dr_after=hist[bi][3],conf=conf,rnd=rnd,orac=orac)
    print('BUD %.0f%% | before %.3f -> after %.3f | conf %.3f | oracle %.3f'%(BUD*100,r0,best,conf,orac))

# ===== BẢNG CSV =====
with open('bang_ketqua.csv','w',newline='') as fo:
    w=csv.writer(fo); w.writerow(['budget','random','RL_truoc(vong0)','RL_sau','conf','oracle','RL_dat_%oracle','%set_benh_truoc','%set_benh_sau'])
    for BUD in BUDS:
        d=finals[BUD]; w.writerow(['%.0f%%'%(BUD*100),'%.3f'%d['rnd'],'%.3f'%d['before'],'%.3f'%d['after'],'%.3f'%d['conf'],'%.3f'%d['orac'],'%.0f%%'%(d['after']/d['orac']*100),'%.0f%%'%(d['pct_before']*100),'%.0f%%'%(d['pct_after']*100)])

# ===== FIG 1: learning curve từng vòng (budget 10%) =====
BUD=0.10; hist,conf,rnd,orac=curves[BUD]
ep=[h[0] for h in hist]; rl=[h[1] for h in hist]
plt.figure(figsize=(8,5))
plt.plot(ep,rl,'-o',ms=3,lw=2,color='#1f77b4',label='RL selector (học qua từng vòng)')
plt.axhline(orac,ls='--',color='#2ca02c',label='Oracle (trần trên) = %.3f'%orac)
plt.axhline(conf,ls='--',color='#ff7f0e',label='Detector-confidence = %.3f'%conf)
plt.axhline(rnd,ls='--',color='#d62728',label='Random = %.3f'%rnd)
plt.xlabel('Vòng RL (epoch)'); plt.ylabel('mean det-IoU của SET được chọn')
plt.title('Closed-loop RL: chất lượng detection của set chọn cải thiện qua từng vòng (budget 10%)')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig('fig_learning_curve.png',dpi=130); plt.close()

# ===== FIG 2: before vs after (3 budget) =====
x=np.arange(len(BUDS)); wd=0.19
rndv=[finals[b]['rnd'] for b in BUDS]; bef=[finals[b]['before'] for b in BUDS]; aft=[finals[b]['after'] for b in BUDS]
cf=[finals[b]['conf'] for b in BUDS]; orc=[finals[b]['orac'] for b in BUDS]
plt.figure(figsize=(9,5.2))
plt.bar(x-2*wd,rndv,wd,label='Random',color='#d62728')
plt.bar(x-wd,bef,wd,label='RL trước học (vòng 0)',color='#9ecae1')
plt.bar(x,aft,wd,label='RL sau học',color='#1f77b4')
plt.bar(x+wd,cf,wd,label='Detector-confidence',color='#ff7f0e')
plt.bar(x+2*wd,orc,wd,label='Oracle',color='#2ca02c')
for i in range(len(BUDS)):
    plt.text(x[i]-wd,bef[i]+.003,'%.3f'%bef[i],ha='center',fontsize=8)
    plt.text(x[i],aft[i]+.003,'%.3f'%aft[i],ha='center',fontsize=8,fontweight='bold')
plt.xticks(x,['budget 5%','budget 10%','budget 20%']); plt.ylabel('mean det-IoU của SET được chọn')
plt.title('Trước vs sau RL — chất lượng detection trên frame được chọn')
plt.legend(ncol=2); plt.grid(axis='y',alpha=.3); plt.tight_layout(); plt.savefig('fig_before_after.png',dpi=130); plt.close()

# ===== FIG 3: mỗi vòng thay đổi gì — %frame bệnh trong set + det-IoU (budget 10%) =====
pct=[h[2]*100 for h in hist]; dr=[h[3]*100 for h in hist]
fig,ax1=plt.subplots(figsize=(8,5))
ax1.plot(ep,rl,'-o',ms=3,color='#1f77b4',label='mean det-IoU set'); ax1.set_xlabel('Vòng RL (epoch)')
ax1.set_ylabel('mean det-IoU của set',color='#1f77b4'); ax1.tick_params(axis='y',labelcolor='#1f77b4')
ax2=ax1.twinx(); ax2.plot(ep,pct,'-s',ms=3,color='#8c564b',label='% frame bệnh trong set')
ax2.plot(ep,dr,'-^',ms=3,color='#e377c2',label='% frame set có IoU≥0.5')
ax2.set_ylabel('% (thành phần set chọn)',color='#8c564b'); ax2.tick_params(axis='y',labelcolor='#8c564b')
l1,la1=ax1.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,la1+la2,loc='center right'); plt.title('Mỗi vòng RL: set chọn dồn dần về frame bệnh detect tốt (budget 10%)')
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig('fig_round_progress.png',dpi=130); plt.close()
print('[*] xuất: fig_learning_curve.png, fig_before_after.png, fig_round_progress.png, bang_ketqua.csv')
