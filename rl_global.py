"""HƯỚNG 1 — RL GLOBAL-SEARCH: agent xuất phát từ box LỚN (phủ cả ảnh), zoom/pan dần tìm tổn thương.
Có ngữ cảnh toàn cục -> cứu được cả box detector lệch xa (mà refine cục bộ chịu thua).
Đặc trưng lesion-aware (FPN detector v3). Đo: standalone, oracle max(detector,global),
và riêng trên SUBSET detector thất bại (IoU<0.5)."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336; FS=21
T=int(sys.argv[1]) if len(sys.argv)>1 else 15
STEP=float(sys.argv[2]) if len(sys.argv)>2 else 0.12
EPOCHS=int(sys.argv[3]) if len(sys.argv)>3 else 60
LR=float(sys.argv[4]) if len(sys.argv)>4 else 3e-4
f=h5py.File('data/rl_detect2.h5','r'); keys=list(f.keys())
vids=sorted(set(f[k].attrs['vid'] for k in keys)); rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
te=set(order[:max(4,int(0.25*len(order)))])
def load(sp):
    ks=[k for k in keys if (f[k].attrs['vid'] in te)==(sp=='te')]
    return (torch.tensor(np.stack([f[k]['fm'][...] for k in ks])),torch.tensor(np.stack([f[k]['gt'][...] for k in ks])),
            torch.tensor(np.stack([f[k]['init'][...] for k in ks])))
FMtr,GTtr,IBtr=load('tr'); FMte,GTte,IBte=load('te')
print('train %d | test %d | GLOBAL-SEARCH T=%d STEP=%.2f'%(len(FMtr),len(FMte),T,STEP))
def iou(a,b):
    x0=torch.max(a[:,0],b[:,0]);y0=torch.max(a[:,1],b[:,1]);x1=torch.min(a[:,2],b[:,2]);y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0)
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)
def roifeat(fm,box):
    N=box.shape[0];idx=torch.arange(N,device=box.device).float().view(-1,1)
    return roi_align(fm,torch.cat([idx,box*(FS/W)],1),output_size=(3,3),aligned=True).reshape(N,-1)
def applyc(box,a):
    cx=(box[:,0]+box[:,2])/2;cy=(box[:,1]+box[:,3])/2;w=(box[:,2]-box[:,0]).clamp(min=2);h=(box[:,3]-box[:,1]).clamp(min=2)
    cx=cx+a[:,0]*STEP*w;cy=cy+a[:,1]*STEP*h;w=w*torch.exp(a[:,2]*STEP);h=h*torch.exp(a[:,3]*STEP)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)
def bigbox(n):
    return torch.tensor([[0.08*W,0.08*W,0.92*W,0.92*W]]*n,dtype=torch.float32)
class Pol(nn.Module):
    def __init__(s,d=256*9+4,h=256):
        super().__init__();s.body=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        s.mu=nn.Linear(h,4);s.logstd=nn.Parameter(torch.zeros(4)-0.5)
    def forward(s,x): z=s.body(x); return s.mu(z),s.logstd.clamp(-2,1)
pol=Pol().to(dev); opt=torch.optim.Adam(pol.parameters(),lr=LR)
def rollout(FM,GT,box0,train=True):
    FM=FM.to(dev);GT=GT.to(dev);box=box0.to(dev).clone();logps=[];rews=[]
    for t in range(T):
        st=torch.cat([roifeat(FM,box),box/W],1); mu,ls=pol(st);std=ls.exp()
        d=torch.distributions.Normal(mu,std); a=d.sample() if train else mu
        old=iou(box,GT);box=applyc(box,a);new=iou(box,GT)
        logps.append(d.log_prob(a).sum(1));rews.append(new-old)
    return box,torch.stack(logps),torch.stack(rews)
def evaluate():
    pol.eval()
    with torch.no_grad():
        gb,_,_=rollout(FMte,GTte,bigbox(len(FMte)),train=False); gi=GTte.to(dev)
        i_det=iou(IBte.to(dev),gi); i_glob=iou(gb,gi); i_orac=torch.max(i_det,i_glob)
    return i_det,i_glob,i_orac
best=0
for ep in range(EPOCHS):
    pol.train();perm=torch.randperm(len(FMtr))
    for i in range(0,len(FMtr),96):
        j=perm[i:i+96]; start=bigbox(len(j))
        _,lp,rw=rollout(FMtr[j],GTtr[j],start,train=True)
        ret=rw.sum(0);base=ret.mean().detach();loss=-(lp.sum(0)*(ret-base)).mean()
        opt.zero_grad();loss.backward();opt.step()
    i_det,i_glob,i_orac=evaluate()
    if i_glob.mean().item()>best: best=i_glob.mean().item()
    if (ep+1)%5==0 or ep==0: print('ep %2d | global IoU %.3f | oracle(det,global) %.3f'%(ep+1,i_glob.mean().item(),i_orac.mean().item()),flush=True)
i_det,i_glob,i_orac=evaluate()
bad=i_det<0.5
print('\n=== HUONG 1 GLOBAL-SEARCH ===')
print('[toan bo] detector %.3f | global %.3f | oracle-max %.3f'%(i_det.mean(),i_glob.mean(),i_orac.mean()))
print('[toan bo DR@0.5] detector %.1f%% | oracle %.1f%%'%((i_det>=.5).float().mean()*100,(i_orac>=.5).float().mean()*100))
print('[SUBSET detector THAT BAI (IoU<0.5), %d frame] detector %.3f -> global %.3f (%+.3f)'%(int(bad.sum()),i_det[bad].mean(),i_glob[bad].mean(),i_glob[bad].mean()-i_det[bad].mean()))
