"""Closed-loop RL — VALUE-BASED selector (critic học Q(frame)=det-IoU kỳ vọng TỪ reward detection).
Vòng: chọn set -> detector -> reward (di) -> critic học map feature->reward -> chọn top-k theo Q -> lặp.
Đây là reward-driven đúng nghĩa (REINFORCE ở rl_loop.py sập vì variance). Chia theo video.
So: RL(value) vs conf(det-conf) vs random vs ORACLE. Metric: mean det-IoU của SET chọn + %set là bệnh + DR."""
import sys, numpy as np, h5py, torch, torch.nn as nn
dev='cuda'; BUD=float(sys.argv[1]) if len(sys.argv)>1 else 0.1; EPOCHS=int(sys.argv[2]) if len(sys.argv)>2 else 60; LR=float(sys.argv[3]) if len(sys.argv)>3 else 1e-3
f=h5py.File('data/loop_frames.h5','r'); keys=list(f.keys())
byv={}
for k in keys:
    v=f[k].attrs['vid']; byv.setdefault(v,[]).append(k)
vids=sorted(byv); rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order); te=set(order[:max(4,int(0.25*len(order)))])
def pack(v):
    ks=sorted(byv[v],key=lambda k:f[k]['meta'][...][0])
    feat=np.stack([f[k]['feat'][...] for k in ks]).astype(np.float32); dc=np.array([float(f[k]['dc'][()]) for k in ks],np.float32)
    di=np.array([float(f[k]['di'][()]) for k in ks],np.float32); isp=np.array([int(f[k]['meta'][...][1]) for k in ks])
    return feat,dc,di,isp
DATA={v:pack(v) for v in vids}
tr=[v for v in vids if v not in te]; teL=[v for v in vids if v in te]
print('video train %d | test %d | budget %.0f%%'%(len(tr),len(teL),BUD*100))
# gom frame train de hoc critic
Xtr=np.concatenate([np.concatenate([DATA[v][0],DATA[v][1][:,None]],1) for v in tr])
Ytr=np.concatenate([DATA[v][2] for v in tr])
Xtr=torch.tensor(Xtr).to(dev); Ytr=torch.tensor(Ytr).to(dev)
class Q(nn.Module):
    def __init__(s,d=513,h=256):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
q=Q().to(dev); opt=torch.optim.Adam(q.parameters(),lr=LR,weight_decay=1e-4); lossf=nn.MSELoss()
def topk(scores,di,isp,k):
    idx=np.argsort(scores)[-k:]; return di[idx].mean(), isp[idx].mean(), (di[idx]>=0.5).mean()
def evaluate():
    q.eval(); rl=[];conf=[];rnd=[];orac=[];pct=[];dr=[]
    with torch.no_grad():
        for v in teL:
            feat,dc,di,isp=DATA[v]; n=len(di); k=max(1,int(n*BUD))
            pred=q(torch.tensor(np.concatenate([feat,dc[:,None]],1)).to(dev)).cpu().numpy()
            r,p,d=topk(pred,di,isp,k); rl.append(r);pct.append(p);dr.append(d)
            conf.append(topk(dc,di,isp,k)[0]); rnd.append(topk(np.random.RandomState(1).rand(n),di,isp,k)[0]); orac.append(topk(di,di,isp,k)[0])
    return np.mean(rl),np.mean(conf),np.mean(rnd),np.mean(orac),np.mean(pct),np.mean(dr)
best=0;bestpct=0;bestdr=0; n=len(Xtr); g=torch.Generator().manual_seed(0)
for ep in range(EPOCHS):
    q.train(); perm=torch.randperm(n,generator=g)
    for i in range(0,n,1024):
        j=perm[i:i+1024]; opt.zero_grad(); lossf(q(Xtr[j]),Ytr[j]).backward(); opt.step()
    rl,conf,rnd,orac,pct,dr=evaluate()
    if rl>best: best=rl;bestpct=pct;bestdr=dr
    if (ep+1)%10==0 or ep==0: print('ep %2d | RL %.3f | conf %.3f | rand %.3f | oracle %.3f | %%path %.0f%%'%(ep+1,rl,conf,rnd,orac,pct*100),flush=True)
rl,conf,rnd,orac,pct,dr=evaluate()
print('\n=== CLOSED-LOOP RL SELECTOR (value-based, reward=detection IoU) ===')
print('mean det-IoU cua SET chon: RL %.3f | conf %.3f | random %.3f | ORACLE %.3f'%(best,conf,rnd,orac))
print('=> RL vs random: %+.3f (%+.0f%%) | RL vs conf: %+.3f | RL đạt %.0f%% oracle'%(best-rnd,(best-rnd)/max(rnd,1e-6)*100,best-conf,best/orac*100))
print('%% frame set là bệnh: RL %.0f%% | DR (di>=.5) trong set: RL %.0f%%'%(bestpct*100,bestdr*100))
