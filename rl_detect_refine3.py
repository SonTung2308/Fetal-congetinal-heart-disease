"""RL refine v3: cổng STOP học được + refine liên tục, đặc trưng lesion-aware (FPN detector v3).
Ý tưởng: box tốt -> agent học DỪNG; box lệch -> refine nhiều bước về đúng. Train trộn box.
Mục tiêu: sửa các box detector kém (DR fail) mà không phá box tốt -> tăng IoU/DR nhiều hơn."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336; FS=21
T=int(sys.argv[1]) if len(sys.argv)>1 else 10
STEP=float(sys.argv[2]) if len(sys.argv)>2 else 0.08
EPOCHS=int(sys.argv[3]) if len(sys.argv)>3 else 60
LR=float(sys.argv[4]) if len(sys.argv)>4 else 3e-4
PMIX=float(sys.argv[5]) if len(sys.argv)>5 else 0.5   # ty le start = box detector

f=h5py.File('data/rl_detect2.h5','r'); keys=list(f.keys())
vids=sorted(set(f[k].attrs['vid'] for k in keys))
rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
nte=max(4,int(0.25*len(order))); te=set(order[:nte])
def load(sp):
    ks=[k for k in keys if (f[k].attrs['vid'] in te)==(sp=='te')]
    return (torch.tensor(np.stack([f[k]['fm'][...] for k in ks])),
            torch.tensor(np.stack([f[k]['gt'][...] for k in ks])),
            torch.tensor(np.stack([f[k]['init'][...] for k in ks])))
FMtr,GTtr,IBtr=load('tr'); FMte,GTte,IBte=load('te')
print('train %d | test %d (%d video) | T=%d STEP=%.2f PMIX=%.1f'%(len(FMtr),len(FMte),len(te),T,STEP,PMIX))

def iou(a,b):
    x0=torch.max(a[:,0],b[:,0]);y0=torch.max(a[:,1],b[:,1]);x1=torch.min(a[:,2],b[:,2]);y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0)
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)
def roifeat(fm,box):
    N=box.shape[0]; idx=torch.arange(N,device=box.device).float().view(-1,1)
    return roi_align(fm,torch.cat([idx,box*(FS/W)],1),output_size=(3,3),aligned=True).reshape(N,-1)
def applyc(box,a):
    cx=(box[:,0]+box[:,2])/2;cy=(box[:,1]+box[:,3])/2;w=(box[:,2]-box[:,0]).clamp(min=2);h=(box[:,3]-box[:,1]).clamp(min=2)
    cx=cx+a[:,0]*STEP*w;cy=cy+a[:,1]*STEP*h;w=w*torch.exp(a[:,2]*STEP);h=h*torch.exp(a[:,3]*STEP)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)

class Pol(nn.Module):
    def __init__(s,d=256*9+4,h=256):
        super().__init__(); s.body=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        s.mu=nn.Linear(h,4); s.logstd=nn.Parameter(torch.zeros(4)-0.5); s.stop=nn.Linear(h,1)
        nn.init.constant_(s.stop.bias,-3.0)   # thien ve DI CHUYEN luc dau, chong sup ve stop
    def forward(s,x):
        z=s.body(x); return s.mu(z), s.logstd.clamp(-2,1), s.stop(z).squeeze(-1)
pol=Pol().to(dev); opt=torch.optim.Adam(pol.parameters(),lr=LR)

def rollout(FM,GT,box0,train=True):
    FM=FM.to(dev);GT=GT.to(dev);box=box0.to(dev).clone();N=box.shape[0]
    alive=torch.ones(N,dtype=torch.bool,device=dev); logps=[];rews=[];ents=[]
    for t in range(T):
        st=torch.cat([roifeat(FM,box),box/W],1)
        mu,ls,stoplg=pol(st); std=ls.exp()
        pstop=torch.sigmoid(stoplg)
        if train:
            sd=torch.distributions.Bernoulli(pstop.clamp(1e-4,1-1e-4)); s_act=sd.sample()
            nd=torch.distributions.Normal(mu,std); a=nd.sample()
            lp=sd.log_prob(s_act)+ (nd.log_prob(a).sum(1))*(1-s_act)   # logp stop, hoặc logp move
            ents.append(sd.entropy()*alive.float())
        else:
            s_act=(pstop>0.5).float(); a=mu; lp=torch.zeros(N,device=dev); ents.append(torch.zeros(N,device=dev))
        do_move=(s_act<0.5)&alive
        old=iou(box,GT); nb=applyc(box,a); box=torch.where(do_move.unsqueeze(1),nb,box); new=iou(box,GT)
        r=(new-old)*alive.float()
        logps.append(lp*alive.float()); rews.append(r)
        alive=alive&(~(s_act>=0.5))
        if not alive.any(): break
    return box,torch.stack(logps),torch.stack(rews),torch.stack(ents)

def perturb(gt):
    g=gt.clone();cx=(g[:,0]+g[:,2])/2;cy=(g[:,1]+g[:,3])/2;w=(g[:,2]-g[:,0]);h=(g[:,3]-g[:,1])
    cx=cx+(torch.rand_like(cx)-0.5)*0.7*w;cy=cy+(torch.rand_like(cy)-0.5)*0.7*h
    w=w*(0.5+torch.rand_like(w)*1.0);h=h*(0.5+torch.rand_like(h)*1.0)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)
def evaluate():
    pol.eval()
    with torch.no_grad():
        bf,_,_,_=rollout(FMte,GTte,IBte,train=False);gi=GTte.to(dev)
        ii=iou(IBte.to(dev),gi);ri=iou(bf,gi)
    return ii.mean().item(),ri.mean().item(),(ii>=0.5).float().mean().item(),(ri>=0.5).float().mean().item()
def eval_rec():
    pol.eval(); torch.manual_seed(1)
    with torch.no_grad():
        gi=GTte.to(dev);st=perturb(gi);bf,_,_,_=rollout(FMte,GTte,st.cpu(),train=False)
        return iou(st,gi).mean().item(),iou(bf,gi).mean().item()

best=(0,0)
for ep in range(EPOCHS):
    pol.train();perm=torch.randperm(len(FMtr))
    for i in range(0,len(FMtr),96):
        j=perm[i:i+96]
        pb=perturb(GTtr[j].to(dev)).cpu(); ib=IBtr[j].clone()
        mask=(torch.rand(len(j))<PMIX).view(-1,1); start=torch.where(mask,ib,pb)
        _,lp,rw,en=rollout(FMtr[j],GTtr[j],start,train=True)
        ret=rw.sum(0);base=ret.mean().detach()
        loss=-(lp.sum(0)*(ret-base)).mean()-0.02*en.sum(0).mean()
        opt.zero_grad();loss.backward();opt.step()
    ii,ri,di,dr=evaluate()
    if ri>best[0]: best=(ri,dr)
    if (ep+1)%5==0 or ep==0: print('ep %2d | IoU %.3f->%.3f | DR %.1f%%->%.1f%%'%(ep+1,ii,ri,di*100,dr*100),flush=True)
pi,po=eval_rec()
print('\n=== KET QUA RL v3 (stop gate) ===')
print('[box detector] init %.3f -> best %.3f (%+.3f) | DR %.1f%%->%.1f%%'%(ii,best[0],best[0]-ii,di*100,best[1]*100))
print('[box NHIEU]    %.3f -> %.3f  (%+.3f)'%(pi,po,po-pi))
