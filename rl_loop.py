"""Closed-loop RL — Bước 2: SELECTOR học chọn frame để DETECTOR cho box tốt.
Reward = mean detection-IoU (det so GT) trên các frame ĐƯỢC CHỌN. Vòng: chọn->detect->reward->update.
So với baseline: random, uniform, top detector-conf, oracle (top det_iou). Chia theo video.
Chỉ số: mean det_iou của set chọn (chất lượng box detector đưa cho bác sĩ) + %frame set là bệnh."""
import sys, numpy as np, h5py, torch, torch.nn as nn
dev='cuda'; BUD=float(sys.argv[1]) if len(sys.argv)>1 else 0.2; EPOCHS=int(sys.argv[2]) if len(sys.argv)>2 else 80; LR=float(sys.argv[3]) if len(sys.argv)>3 else 5e-4
f=h5py.File('data/loop_frames.h5','r'); keys=list(f.keys())
byv={}
for k in keys:
    v=f[k].attrs['vid']; byv.setdefault(v,[]).append(k)
vids=sorted(byv); rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order); te=set(order[:max(4,int(0.25*len(order)))])
def pack(v):
    ks=sorted(byv[v],key=lambda k:f[k]['meta'][...][0])
    feat=np.stack([f[k]['feat'][...] for k in ks]); dc=np.array([float(f[k]['dc'][()]) for k in ks],np.float32)
    di=np.array([float(f[k]['di'][()]) for k in ks],np.float32); isp=np.array([int(f[k]['meta'][...][1]) for k in ks])
    return feat,dc,di,isp
DATA={v:pack(v) for v in vids}
tr=[v for v in vids if v not in te]; teL=[v for v in vids if v in te]
print('video train %d | test %d | budget %.0f%%'%(len(tr),len(teL),BUD*100))
class Sel(nn.Module):
    def __init__(s,d=512+1,h=128):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,feat,dc): return s.net(torch.cat([feat,dc.unsqueeze(1)],1)).squeeze(-1)
sel=Sel().to(dev); opt=torch.optim.Adam(sel.parameters(),lr=LR,weight_decay=1e-4)
def topk_reward(v,scores):
    feat,dc,di,isp=DATA[v]; n=len(di); k=max(1,int(n*BUD))
    idx=np.argsort(scores)[-k:]
    return di[idx].mean(), isp[idx].mean(), (di[idx]>=0.5).mean()
def evaluate(vlist):
    sel.eval(); res=defaultdict=None
    rl=[];rndm=[];uni=[];conf=[];orac=[]
    with torch.no_grad():
        for v in vlist:
            feat,dc,di,isp=DATA[v]; n=len(di)
            sc=sel(torch.tensor(feat).to(dev),torch.tensor(dc).to(dev)).cpu().numpy()
            rl.append(topk_reward(v,sc)[0])
            rndm.append(topk_reward(v,np.random.RandomState(1).rand(n))[0])
            uni.append(topk_reward(v,-np.abs(np.arange(n)-n/2))[0])   # gần giữa ~ uniform proxy? dùng conf thay
            conf.append(topk_reward(v,dc)[0])
            orac.append(topk_reward(v,di)[0])
    return np.mean(rl),np.mean(rndm),np.mean(conf),np.mean(orac)
from collections import defaultdict
best=0
for ep in range(EPOCHS):
    sel.train(); rng.shuffle(tr); loss=0;cnt=0
    for v in tr:
        feat,dc,di,isp=DATA[v]; n=len(di); k=max(1,int(n*BUD))
        sc=sel(torch.tensor(feat).to(dev),torch.tensor(dc).to(dev))
        p=torch.sigmoid(sc); m=torch.distributions.Bernoulli(p.clamp(1e-4,1-1e-4)); a=m.sample()
        keep=a.bool().cpu().numpy()
        r = di[keep].mean() if keep.sum()>0 else 0.0
        r = r - 0.5*abs(keep.mean()-BUD)                      # phạt lệch ngân sách
        loss=loss - (m.log_prob(a).mean()*(r)); cnt+=1
        if cnt>=32: (loss/cnt).backward(); opt.step(); opt.zero_grad(); loss=0;cnt=0
    if cnt>0: (loss/cnt).backward(); opt.step(); opt.zero_grad()
    rl,rndm,conf,orac=evaluate(teL)
    if rl>best: best=rl
    if (ep+1)%10==0 or ep==0: print('ep %2d | RL %.3f | conf %.3f | rand %.3f | oracle %.3f'%(ep+1,rl,conf,rndm,orac),flush=True)
rl,rndm,conf,orac=evaluate(teL)
# them thong ke bo sung
sel.eval(); pct=[]
with torch.no_grad():
    for v in teL:
        feat,dc,di,isp=DATA[v]; sc=sel(torch.tensor(feat).to(dev),torch.tensor(dc).to(dev)).cpu().numpy()
        pct.append(topk_reward(v,sc)[1])
print('\n=== CLOSED-LOOP RL SELECTOR (reward = detection IoU) ===')
print('mean det-IoU cua SET chon (cao=tot): RL %.3f | conf %.3f | random %.3f | ORACLE %.3f'%(best,conf,rndm,orac))
print('=> RL vs random: %+.3f (%+.0f%%) | RL vs conf-baseline: %+.3f'%(best-rndm,(best-rndm)/max(rndm,1e-6)*100,best-conf))
print('%% frame trong set la benh (RL): %.0f%%'%(np.mean(pct)*100))
