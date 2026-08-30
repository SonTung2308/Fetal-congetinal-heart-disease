"""RL refine v2: đặc trưng LESION-AWARE (FPN detector v3) + policy LIÊN TỤC tối ưu trực tiếp IoU.
State = roi_align 3x3 (giữ hướng) + hình học box. Action ~ Gaussian Δ(cx,cy,logw,logh).
Reward = ΔIoU. REINFORCE. Train với box nhiễu, eval trên box detector. Chia theo video."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336; FS=21   # feature map 21x21
T=int(sys.argv[1]) if len(sys.argv)>1 else 5
STEP=float(sys.argv[2]) if len(sys.argv)>2 else 0.1
EPOCHS=int(sys.argv[3]) if len(sys.argv)>3 else 40
LR=float(sys.argv[4]) if len(sys.argv)>4 else 3e-4

f=h5py.File('data/rl_detect2.h5','r'); keys=list(f.keys())
vids=sorted(set(f[k].attrs['vid'] for k in keys))
rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
nte=max(4,int(0.25*len(order))); te=set(order[:nte])
def load(split):
    ks=[k for k in keys if (f[k].attrs['vid'] in te)==(split=='te')]
    fm=np.stack([f[k]['fm'][...] for k in ks]); gt=np.stack([f[k]['gt'][...] for k in ks]); ib=np.stack([f[k]['init'][...] for k in ks])
    return torch.tensor(fm),torch.tensor(gt),torch.tensor(ib)
FMtr,GTtr,IBtr=load('tr'); FMte,GTte,IBte=load('te')
print('train %d | test %d (%d video) | T=%d STEP=%.2f'%(len(FMtr),len(FMte),len(te),T,STEP))

def iou(a,b):
    x0=torch.max(a[:,0],b[:,0]); y0=torch.max(a[:,1],b[:,1]); x1=torch.min(a[:,2],b[:,2]); y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0)
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)
def roifeat(fm,box):
    N=box.shape[0]; idx=torch.arange(N,device=box.device).float().view(-1,1)
    rois=torch.cat([idx,box*(FS/W)],1)
    return roi_align(fm,rois,output_size=(3,3),aligned=True).reshape(N,-1)   # (N,256*9)
def applyc(box,a):   # a: (N,4) = d_cx,d_cy,d_logw,d_logh
    cx=(box[:,0]+box[:,2])/2; cy=(box[:,1]+box[:,3])/2; w=(box[:,2]-box[:,0]).clamp(min=2); h=(box[:,3]-box[:,1]).clamp(min=2)
    cx=cx+a[:,0]*STEP*w; cy=cy+a[:,1]*STEP*h; w=w*torch.exp(a[:,2]*STEP); h=h*torch.exp(a[:,3]*STEP)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)

class Pol(nn.Module):
    def __init__(s,d=256*9+4,h=256):
        super().__init__(); s.body=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        s.mu=nn.Linear(h,4); s.logstd=nn.Parameter(torch.zeros(4)-0.5)
    def forward(s,x):
        z=s.body(x); return s.mu(z), s.logstd.clamp(-2,1)
pol=Pol().to(dev); opt=torch.optim.Adam(pol.parameters(),lr=LR)

def rollout(FM,GT,box0,train=True):
    FM=FM.to(dev); GT=GT.to(dev); box=box0.to(dev).clone()
    logps=[]; rews=[]
    for t in range(T):
        st=torch.cat([roifeat(FM,box),box/W],1)
        mu,ls=pol(st); std=ls.exp()
        d=torch.distributions.Normal(mu,std)
        a=d.sample() if train else mu
        lp=d.log_prob(a).sum(1)
        old=iou(box,GT); box=applyc(box,a); new=iou(box,GT)
        logps.append(lp); rews.append(new-old)
    return box,torch.stack(logps),torch.stack(rews)

def perturb(gt):
    g=gt.clone(); cx=(g[:,0]+g[:,2])/2; cy=(g[:,1]+g[:,3])/2; w=(g[:,2]-g[:,0]); h=(g[:,3]-g[:,1])
    cx=cx+(torch.rand_like(cx)-0.5)*0.6*w; cy=cy+(torch.rand_like(cy)-0.5)*0.6*h
    w=w*(0.6+torch.rand_like(w)*0.9); h=h*(0.6+torch.rand_like(h)*0.9)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)

def evaluate(sub=False):
    pol.eval()
    with torch.no_grad():
        bf,_,_=rollout(FMte,GTte,IBte,train=False); gi=GTte.to(dev)
        i0=iou(IBte.to(dev),gi); i1=iou(bf,gi)
    if sub:
        bad=i0<0.5   # box detector KEM
        return (i0[bad].mean().item(),i1[bad].mean().item(),int(bad.sum()),
                i0.mean().item(),i1.mean().item(),(i0>=0.5).float().mean().item(),(i1>=0.5).float().mean().item())
    return i0.mean().item(),i1.mean().item(),(i0>=0.5).float().mean().item(),(i1>=0.5).float().mean().item()
def eval_recover():
    pol.eval(); torch.manual_seed(1)
    with torch.no_grad():
        gi=GTte.to(dev); st=perturb(gi); bf,_,_=rollout(FMte,GTte,st.cpu(),train=False)
        return iou(st,gi).mean().item(),iou(bf,gi).mean().item()

best=(0,0)
for ep in range(EPOCHS):
    pol.train(); perm=torch.randperm(len(FMtr))
    for i in range(0,len(FMtr),96):
        j=perm[i:i+96]
        # TRỘN: 1/2 bắt đầu từ box detector (tốt -> học để yên), 1/2 từ box nhiễu (học dịch)
        pb=perturb(GTtr[j].to(dev)).cpu(); ib=IBtr[j].clone()
        mask=(torch.rand(len(j))<0.5).view(-1,1)
        start=torch.where(mask,ib,pb)
        _,lp,rw=rollout(FMtr[j],GTtr[j],start,train=True)
        ret=rw.sum(0); base=ret.mean().detach()
        loss=-(lp.sum(0)*(ret-base)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ii,ri,di,dr=evaluate()
    if ri>best[0]: best=(ri,dr)
    if (ep+1)%5==0 or ep==0: print('ep %2d | IoU %.3f->%.3f | DR %.1f%%->%.1f%%'%(ep+1,ii,ri,di*100,dr*100),flush=True)
pi,po=eval_recover()
bi,bo,nbad,ai,ao,d0,d1=evaluate(sub=True)
print('\n=== KET QUA RL v2 (lesion-aware + continuous) ===')
print('[toan bo]        IoU %.3f -> %.3f | DR %.1f%% -> %.1f%%'%(ai,ao,d0*100,d1*100))
print('[box detector KEM (IoU<0.5), %d frame] IoU %.3f -> %.3f  (%+.3f = %+.0f%%)'%(nbad,bi,bo,bo-bi,(bo-bi)/max(bi,1e-6)*100))
print('[box NHIEU]      %.3f -> %.3f  (%+.3f)'%(pi,po,po-pi))
