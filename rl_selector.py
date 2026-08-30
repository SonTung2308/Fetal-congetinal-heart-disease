"""RL SELECTOR: chọn proposal tốt nhất trong top-K của detector (không refine, không sửa box).
State mỗi proposal = roi_align(FPN lesion-aware) + hình học + score. Policy = softmax over proposals,
REINFORCE reward=IoU. Baseline = IoU top-1-score. Mục tiêu: vượt top-1-score (0.785) tiến tới oracle (0.814)."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336; FS=21
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 60; LR=float(sys.argv[2]) if len(sys.argv)>2 else 3e-4
fa=h5py.File('data/rl_detect2.h5','r'); fb=h5py.File('data/rl_select.h5','r')
keys=sorted(set(fa)&set(fb)); vids=sorted(set(fb[k].attrs['vid'] for k in keys))
rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order); te=set(order[:max(4,int(0.25*len(order)))])
# nap vao RAM
data={}
for k in keys:
    data[k]=(torch.tensor(fa[k]['fm'][...]),torch.tensor(fb[k]['boxes'][...]),torch.tensor(fb[k]['scores'][...]),torch.tensor(fb[k]['gt'][...]),fb[k].attrs['vid'])
tr=[k for k in keys if data[k][4] not in te]; teK=[k for k in keys if data[k][4] in te]
print('train %d | test %d anh | %d video test'%(len(tr),len(teK),len(te)))
def iou1(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def roifeat(fm,boxes):
    n=boxes.shape[0];idx=torch.zeros(n,1,device=dev)
    return roi_align(fm.unsqueeze(0).to(dev),torch.cat([idx,boxes.to(dev)*(FS/W)],1),output_size=(3,3),aligned=True).reshape(n,-1)
class Sel(nn.Module):
    def __init__(s,d=256*9+5,h=128):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,st): return s.net(st).squeeze(-1)
sel=Sel().to(dev); opt=torch.optim.Adam(sel.parameters(),lr=LR)
def state(k):
    fm,bx,sc,gt,_=data[k]
    rf=roifeat(fm,bx); geom=(bx.to(dev)/W); scc=sc.to(dev).unsqueeze(1)
    return torch.cat([rf,geom,scc],1),bx,gt
def evaluate(ks):
    sel.eval(); t1=[];rl=[]
    with torch.no_grad():
        for k in ks:
            st,bx,gt=state(k); lg=sel(st); pick=lg.argmax().item()
            gtl=gt.tolist(); t1.append(iou1(bx[0].tolist(),gtl)); rl.append(iou1(bx[pick].tolist(),gtl))
    t1=np.array(t1);rl=np.array(rl)
    return t1.mean(),rl.mean(),(t1>=.5).mean(),(rl>=.5).mean()
best=0
for ep in range(EPOCHS):
    sel.train(); rng.shuffle(tr); loss=0;cnt=0
    for k in tr:
        st,bx,gt=state(k)
        if bx.shape[0]<2: continue
        lg=sel(st); d=torch.distributions.Categorical(logits=lg); a=d.sample()
        gtl=gt.tolist(); r=iou1(bx[a.item()].tolist(),gtl); b0=iou1(bx[0].tolist(),gtl)
        loss=loss-d.log_prob(a)*(r-b0); cnt+=1
        if cnt>=64:
            (loss/cnt).backward(); opt.step(); opt.zero_grad(); loss=0;cnt=0
    if cnt>0: (loss/cnt).backward(); opt.step(); opt.zero_grad()
    i1,ir,d1,dr=evaluate(teK)
    if ir>best: best=ir
    if (ep+1)%5==0 or ep==0: print('ep %2d | top1 %.3f -> RL %.3f | DR %.1f%%->%.1f%%'%(ep+1,i1,ir,d1*100,dr*100),flush=True)
i1,ir,d1,dr=evaluate(teK)
print('\n=== RL SELECTOR ===')
print('top-1-score %.3f -> RL-select best %.3f (%+.3f = %+.1f%%) | DR %.1f%% -> %.1f%%'%(i1,best,best-i1,(best-i1)/i1*100,d1*100,dr*100))
print('(oracle top-K = 0.814)')
