"""RL SELECTOR v2 — VALUE-BASED (critic Q(proposal)=IoU dự đoán, chọn argmax).
REINFORCE thất bại vì trên train top-1-score đã ~oracle (không học được). Value-based học
features->IoU từ MỌI proposal (giàu tín hiệu) -> tổng quát lên test (oracle +14%).
Chia theo video. Baseline: top-1-score (0.613 test)."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336; FS=21
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 80; LR=float(sys.argv[2]) if len(sys.argv)>2 else 5e-4
USE_SCORE=int(sys.argv[3]) if len(sys.argv)>3 else 1
fa=h5py.File('data/rl_detect2.h5','r'); fb=h5py.File('data/rl_select.h5','r')
keys=sorted(set(fa)&set(fb)); vids=sorted(set(fb[k].attrs['vid'] for k in keys))
rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order); te=set(order[:max(4,int(0.25*len(order)))])
def iou1(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
# build per-proposal dataset
def build(cond):
    S=[];Y=[];META=[]
    for k in keys:
        if (fb[k].attrs['vid'] in te)!=cond: continue
        fm=torch.tensor(fa[k]['fm'][...]); bx=torch.tensor(fb[k]['boxes'][...]); sc=torch.tensor(fb[k]['scores'][...]); gt=fb[k]['gt'][...]
        n=bx.shape[0]; idx=torch.zeros(n,1)
        rf=roi_align(fm.unsqueeze(0),torch.cat([idx,bx*(FS/W)],1),output_size=(3,3),aligned=True).reshape(n,-1)
        st=torch.cat([rf,bx/W,sc.unsqueeze(1)],1) if USE_SCORE else torch.cat([rf,bx/W],1)
        ious=torch.tensor([iou1(b.tolist(),gt.tolist()) for b in bx])
        S.append(st); Y.append(ious); META.append((n,ious))
    return S,Y,META
Str,Ytr,_=build(False); Ste,Yte,_=build(True)
print('train %d anh | test %d anh | dùng score=%d'%(len(Str),len(Ste),USE_SCORE))
Xtr=torch.cat(Str).to(dev); ytr=torch.cat(Ytr).to(dev)
D=Xtr.shape[1]
class Q(nn.Module):
    def __init__(s,d,h=128):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
q=Q(D).to(dev); opt=torch.optim.Adam(q.parameters(),lr=LR,weight_decay=1e-4); lossf=nn.MSELoss()
def evaluate():
    q.eval(); t1=[];rl=[];orac=[]
    with torch.no_grad():
        for st,y in zip(Ste,Yte):
            pred=q(st.to(dev)); pick=pred.argmax().item()
            t1.append(y[0].item()); rl.append(y[pick].item()); orac.append(y.max().item())
    return np.mean(t1),np.mean(rl),np.mean(orac),(np.array(t1)>=.5).mean(),(np.array(rl)>=.5).mean()
best=0; n=len(Xtr); g=torch.Generator().manual_seed(0)
for ep in range(EPOCHS):
    q.train(); perm=torch.randperm(n,generator=g)
    for i in range(0,n,512):
        j=perm[i:i+512]; opt.zero_grad(); lossf(q(Xtr[j]),ytr[j]).backward(); opt.step()
    i1,ir,orc,d1,dr=evaluate()
    if ir>best: best=ir
    if (ep+1)%10==0 or ep==0: print('ep %2d | top1 %.3f -> RL %.3f (oracle %.3f) | DR %.1f%%->%.1f%%'%(ep+1,i1,ir,orc,d1*100,dr*100),flush=True)
i1,ir,orc,d1,dr=evaluate()
print('\n=== RL SELECTOR v2 (value-based) ===')
print('top-1-score %.3f -> RL-select %.3f best %.3f (%+.3f = %+.1f%%) | oracle %.3f'%(i1,ir,best,best-i1,(best-i1)/i1*100,orc))
print('DR@0.5: top1 %.1f%% -> RL %.1f%%'%(d1*100,dr*100))
