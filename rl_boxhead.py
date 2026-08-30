"""HƯỚNG 2 — RL-train BOX HEAD bằng reward IoU trực tiếp (khác L1 của detector chuẩn).
Agent dự đoán offset box MỘT BƯỚC dứt khoát từ đặc trưng lesion-aware (FPN detector v3),
tối ưu trực tiếp IoU bằng REINFORCE. Cũng kiểm TRẦN NHÃN bằng DIoU-loss (khả vi) để biết
giới hạn thật của dữ liệu."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336; FS=21
MODE=sys.argv[1] if len(sys.argv)>1 else 'rl'   # 'rl' hoac 'diou'
STEP=float(sys.argv[2]) if len(sys.argv)>2 else 0.35
EPOCHS=int(sys.argv[3]) if len(sys.argv)>3 else 60
f=h5py.File('data/rl_detect2.h5','r'); keys=list(f.keys())
vids=sorted(set(f[k].attrs['vid'] for k in keys)); rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
te=set(order[:max(4,int(0.25*len(order)))])
def load(sp):
    ks=[k for k in keys if (f[k].attrs['vid'] in te)==(sp=='te')]
    return (torch.tensor(np.stack([f[k]['fm'][...] for k in ks])),torch.tensor(np.stack([f[k]['gt'][...] for k in ks])),
            torch.tensor(np.stack([f[k]['init'][...] for k in ks])))
FMtr,GTtr,IBtr=load('tr'); FMte,GTte,IBte=load('te')
print('train %d | test %d | MODE=%s STEP=%.2f'%(len(FMtr),len(FMte),MODE,STEP))
def iou(a,b):
    x0=torch.max(a[:,0],b[:,0]);y0=torch.max(a[:,1],b[:,1]);x1=torch.min(a[:,2],b[:,2]);y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0)
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)
def diou_loss(p,g):   # 1 - DIoU (khả vi)
    ii=iou(p,g)
    pcx=(p[:,0]+p[:,2])/2;pcy=(p[:,1]+p[:,3])/2;gcx=(g[:,0]+g[:,2])/2;gcy=(g[:,1]+g[:,3])/2
    cdist=(pcx-gcx)**2+(pcy-gcy)**2
    ex0=torch.min(p[:,0],g[:,0]);ey0=torch.min(p[:,1],g[:,1]);ex1=torch.max(p[:,2],g[:,2]);ey1=torch.max(p[:,3],g[:,3])
    diag=(ex1-ex0)**2+(ey1-ey0)**2+1e-6
    return (1-ii+cdist/diag).mean()
def roifeat(fm,box):
    N=box.shape[0];idx=torch.arange(N,device=box.device).float().view(-1,1)
    return roi_align(fm,torch.cat([idx,box*(FS/W)],1),output_size=(3,3),aligned=True).reshape(N,-1)
def applyc(box,a):
    cx=(box[:,0]+box[:,2])/2;cy=(box[:,1]+box[:,3])/2;w=(box[:,2]-box[:,0]).clamp(min=2);h=(box[:,3]-box[:,1]).clamp(min=2)
    cx=cx+a[:,0]*STEP*w;cy=cy+a[:,1]*STEP*h;w=w*torch.exp(a[:,2]*STEP);h=h*torch.exp(a[:,3]*STEP)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)
class Net(nn.Module):
    def __init__(s,d=256*9+4,h=256):
        super().__init__();s.body=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        s.mu=nn.Linear(h,4);s.logstd=nn.Parameter(torch.zeros(4)-0.5)
    def forward(s,x): z=s.body(x);return s.mu(z),s.logstd.clamp(-2,1)
net=Net().to(dev);opt=torch.optim.Adam(net.parameters(),lr=3e-4)
FMteD=FMte.to(dev);GTteD=GTte.to(dev);IBteD=IBte.to(dev)
def evaluate():
    net.eval()
    with torch.no_grad():
        mu,_=net(torch.cat([roifeat(FMteD,IBteD),IBteD/W],1)); pb=applyc(IBteD,mu)
        return iou(IBteD,GTteD),iou(pb,GTteD)
best=0
for ep in range(EPOCHS):
    net.train();perm=torch.randperm(len(FMtr))
    for i in range(0,len(FMtr),96):
        j=perm[i:i+96];FM=FMtr[j].to(dev);GT=GTtr[j].to(dev);IB=IBtr[j].to(dev)
        mu,ls=net(torch.cat([roifeat(FM,IB),IB/W],1))
        if MODE=='rl':
            std=ls.exp();d=torch.distributions.Normal(mu,std);a=d.sample()
            pb=applyc(IB,a);r=iou(pb,GT);loss=-(d.log_prob(a).sum(1)*(r-r.mean().detach())).mean()
        else:
            pb=applyc(IB,mu);loss=diou_loss(pb,GT)
        opt.zero_grad();loss.backward();opt.step()
    i0,i1=evaluate()
    if i1.mean().item()>best: best=i1.mean().item()
    if (ep+1)%5==0 or ep==0: print('ep %2d | det %.3f -> %s %.3f | DR %.1f%%->%.1f%%'%(ep+1,i0.mean(),MODE,i1.mean(),(i0>=.5).float().mean()*100,(i1>=.5).float().mean()*100),flush=True)
i0,i1=evaluate();bad=i0<0.5
print('\n=== HUONG 2 (%s) ==='%MODE)
print('[toan bo] detector %.3f -> %.3f (best %.3f, %+.3f = %+.1f%%) | DR %.1f%%->%.1f%%'%(i0.mean(),i1.mean(),best,best-i0.mean().item(),(best-i0.mean().item())/i0.mean().item()*100,(i0>=.5).float().mean()*100,(i1>=.5).float().mean()*100))
print('[SUBSET detector THAT BAI, %d frame] %.3f -> %.3f (%+.3f)'%(int(bad.sum()),i0[bad].mean(),i1[bad].mean(),i1[bad].mean()-i0[bad].mean()))
