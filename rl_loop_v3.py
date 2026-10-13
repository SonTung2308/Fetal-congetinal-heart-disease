"""Closed-loop RL v3: critic phân loại 'frame bệnh detect tốt' (di>=0.5) bằng BCE (tín hiệu sạch hơn regression),
rank theo prob. Thêm chế độ BLEND: score = prob_critic * (0.5+0.5*conf) để kết hợp tín hiệu học được + conf detector.
So RL / RL+blend / conf / random / oracle. Chia video."""
import sys, numpy as np, h5py, torch, torch.nn as nn
dev='cuda'; BUD=float(sys.argv[1]) if len(sys.argv)>1 else 0.1; EPOCHS=int(sys.argv[2]) if len(sys.argv)>2 else 60
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
Xtr=np.concatenate([np.concatenate([DATA[v][0],DATA[v][1][:,None]],1) for v in tr]); Ytr=(np.concatenate([DATA[v][2] for v in tr])>=0.5).astype(np.float32)
Xtr=torch.tensor(Xtr).to(dev); Ytr=torch.tensor(Ytr).to(dev)
print('train frame %d | pos(di>=.5) %.1f%%'%(len(Ytr),Ytr.mean().item()*100))
class C(nn.Module):
    def __init__(s,d=513,h=256):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
c=C().to(dev); opt=torch.optim.Adam(c.parameters(),lr=1e-3,weight_decay=1e-4)
pw=torch.tensor([(1-Ytr.mean()).item()/max(Ytr.mean().item(),1e-4)],device=dev); lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
def topk(scores,di,isp,k):
    idx=np.argsort(scores)[-k:]; return di[idx].mean(), isp[idx].mean(), (di[idx]>=0.5).mean()
def evaluate():
    c.eval(); rl=[];bl=[];conf=[];rnd=[];orac=[];pct=[]
    with torch.no_grad():
        for v in teL:
            feat,dc,di,isp=DATA[v]; n=len(di); k=max(1,int(n*BUD))
            prob=torch.sigmoid(c(torch.tensor(np.concatenate([feat,dc[:,None]],1)).to(dev))).cpu().numpy()
            rl.append(topk(prob,di,isp,k)[0]); bl.append(topk(prob*(0.5+0.5*dc),di,isp,k)[0])
            r,p,_=topk(prob,di,isp,k); pct.append(p)
            conf.append(topk(dc,di,isp,k)[0]); rnd.append(topk(np.random.RandomState(1).rand(n),di,isp,k)[0]); orac.append(topk(di,di,isp,k)[0])
    return np.mean(rl),np.mean(bl),np.mean(conf),np.mean(rnd),np.mean(orac),np.mean(pct)
best=0;bestbl=0; n=len(Xtr); g=torch.Generator().manual_seed(0)
for ep in range(EPOCHS):
    c.train(); perm=torch.randperm(n,generator=g)
    for i in range(0,n,1024):
        j=perm[i:i+1024]; opt.zero_grad(); lossf(c(Xtr[j]),Ytr[j]).backward(); opt.step()
    rl,bl,conf,rnd,orac,pct=evaluate()
    if rl>best: best=rl
    if bl>bestbl: bestbl=bl
    if (ep+1)%10==0 or ep==0: print('ep %2d | RL %.3f | blend %.3f | conf %.3f | oracle %.3f'%(ep+1,rl,bl,conf,orac),flush=True)
rl,bl,conf,rnd,orac,pct=evaluate()
print('\n=== CLOSED-LOOP RL v3 (BCE classify + blend) ===')
print('mean det-IoU SET: RL %.3f | RL+blend %.3f | conf %.3f | random %.3f | ORACLE %.3f'%(best,bestbl,conf,rnd,orac))
print('=> best(RL/blend) vs conf: %+.3f | đạt %.0f%% oracle | %%set bệnh %.0f%%'%(max(best,bestbl)-conf,max(best,bestbl)/orac*100,pct*100))
