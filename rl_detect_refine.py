"""RL tinh chỉnh bounding box v2. Bắt đầu từ box detector. 9 hành động: 8 dịch/co giãn + STOP.
STOP đóng băng box (sàn = box detector). Reward = ΔIoU. REINFORCE + entropy. Chia theo video."""
import sys, numpy as np, h5py, torch, torch.nn as nn
from torchvision.ops import roi_align
dev='cuda'; W=336
T=int(sys.argv[1]) if len(sys.argv)>1 else 8
ALPHA=float(sys.argv[2]) if len(sys.argv)>2 else 0.08
EPOCHS=int(sys.argv[3]) if len(sys.argv)>3 else 40
LR=float(sys.argv[4]) if len(sys.argv)>4 else 5e-4
ENT=0.01

f=h5py.File('data/rl_detect.h5','r'); keys=list(f.keys())
vids=sorted(set(f[k].attrs['vid'] for k in keys))
rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
nte=max(4,int(0.25*len(order))); te=set(order[:nte])
def load(split):
    ks=[k for k in keys if (f[k].attrs['vid'] in te)==(split=='te')]
    fm=np.stack([f[k]['fm'][...] for k in ks]); gt=np.stack([f[k]['gt'][...] for k in ks]); ib=np.stack([f[k]['init'][...] for k in ks])
    return torch.tensor(fm),torch.tensor(gt),torch.tensor(ib)
FMtr,GTtr,IBtr=load('tr'); FMte,GTte,IBte=load('te')
print('train %d | test %d (%d video test) | T=%d ALPHA=%.2f'%(len(FMtr),len(FMte),len(te),T,ALPHA))

def iou(a,b):
    x0=torch.max(a[:,0],b[:,0]); y0=torch.max(a[:,1],b[:,1]); x1=torch.min(a[:,2],b[:,2]); y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0)
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)

def roifeat(fm,box):  # giu bo cuc 3x3 (khong trung binh) -> agent thay HUONG lech cua ton thuong
    N=box.shape[0]; idx=torch.arange(N,device=box.device).float().view(-1,1)
    rois=torch.cat([idx,box*(7.0/W)],1)
    r=roi_align(fm,rois,output_size=(3,3),aligned=True)   # (N,512,3,3)
    # nen kenh xuong 32 de gon: dung mean theo nhom kenh
    r=r.view(N,512,9).mean(1)                             # (N,9) nang luong theo o -> huong
    r2=roi_align(fm,rois,output_size=(1,1),aligned=True).view(N,512)  # dac trung tong the
    return torch.cat([r,r2],1)                            # (N, 9+512=521)

# hành động vector hoá (N,4) delta theo w,h; action 8 = STOP
def apply(box,a):
    b=box.clone(); w=(b[:,2]-b[:,0]).clamp(min=2); h=(b[:,3]-b[:,1]).clamp(min=2)
    dx=ALPHA*w; dy=ALPHA*h
    A=a
    m=lambda k:(A==k)
    b[:,0]+= (-dx)*m(0)+dx*m(1)+(-dx/2)*m(4)+(dx/2)*m(5)
    b[:,2]+= (-dx)*m(0)+dx*m(1)+(dx/2)*m(4)+(-dx/2)*m(5)
    b[:,1]+= (-dy)*m(2)+dy*m(3)+(-dy/2)*m(6)+(dy/2)*m(7)
    b[:,3]+= (-dy)*m(2)+dy*m(3)+(dy/2)*m(6)+(-dy/2)*m(7)
    return b.clamp(0,W)

class Pol(nn.Module):
    def __init__(s,d=525,h=256,na=9):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,na))
    def forward(s,x): return s.net(x)
pol=Pol().to(dev); opt=torch.optim.Adam(pol.parameters(),lr=LR)

def rollout(FM,GT,IB,train=True):
    FM=FM.to(dev); GT=GT.to(dev); box=IB.to(dev).clone(); N=box.shape[0]
    alive=torch.ones(N,dtype=torch.bool,device=dev)
    logps=[]; rews=[]; ents=[]
    for t in range(T):
        rf=roifeat(FM,box); st=torch.cat([rf,box/W],1)
        logit=pol(st); d=torch.distributions.Categorical(logits=logit)
        a=d.sample() if train else logit.argmax(1)
        stop=(a==8)|(~alive)              # đã stop hoặc chọn stop
        old=iou(box,GT); newbox=apply(box,a)
        box=torch.where((~stop).unsqueeze(1),newbox,box)   # chỉ đổi nếu còn sống & không stop
        new=iou(box,GT)
        r=(new-old)*alive.float()          # reward 0 nếu đã chết
        logps.append(d.log_prob(a)*alive.float()); rews.append(r); ents.append(d.entropy()*alive.float())
        alive=alive&(~(a==8))
        if not alive.any(): break
    return box,torch.stack(logps),torch.stack(rews),torch.stack(ents)

def evaluate():
    pol.eval()
    with torch.no_grad():
        bf,_,_,_=rollout(FMte,GTte,IBte,train=False)
        gi=GTte.to(dev); ii=iou(IBte.to(dev),gi); ri=iou(bf,gi)
    return ii.mean().item(),ri.mean().item(),(ii>=0.5).float().mean().item(),(ri>=0.5).float().mean().item()

def eval_recover():  # cho box NHIEU -> RL co sua lai duoc khong (chung minh RL hoc dc ky nang)
    pol.eval(); torch.manual_seed(1)
    with torch.no_grad():
        gi=GTte.to(dev); start=perturb(gi)
        bf,_,_,_=rollout(FMte,GTte,start.cpu(),train=False)
        return iou(start,gi).mean().item(),iou(bf,gi).mean().item()

def perturb(gt):  # tao box bat dau nhieu tu GT -> IoU da dang, day agent hoc dich box
    g=gt.clone(); cx=(g[:,0]+g[:,2])/2; cy=(g[:,1]+g[:,3])/2; w=(g[:,2]-g[:,0]); h=(g[:,3]-g[:,1])
    cx=cx+(torch.rand_like(cx)-0.5)*0.6*w; cy=cy+(torch.rand_like(cy)-0.5)*0.6*h
    w=w*(0.6+torch.rand_like(w)*0.9); h=h*(0.6+torch.rand_like(h)*0.9)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,W)

best=(0,None)
for ep in range(EPOCHS):
    pol.train(); perm=torch.randperm(len(FMtr))
    for i in range(0,len(FMtr),128):
        j=perm[i:i+128]
        start=perturb(GTtr[j].to(dev)).cpu()          # box bat dau nhieu
        _,lp,rw,en=rollout(FMtr[j],GTtr[j],start,train=True)
        ret=rw.sum(0); base=ret.mean().detach()
        loss=-(lp.sum(0)*(ret-base)).mean()-ENT*en.sum(0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ii,ri,di,dr=evaluate()
    if ri>best[0]: best=(ri,dr)
    if (ep+1)%5==0 or ep==0:
        print('ep %2d | IoU %.3f->%.3f | DR@0.5 %.1f%%->%.1f%%'%(ep+1,ii,ri,di*100,dr*100),flush=True)
pi,po=eval_recover()
print('\n=== KET QUA ===')
print('[Tren box detector] init IoU %.3f | sau RL %.3f | best %.3f  (%+.3f)'%(ii,ri,best[0],best[0]-ii))
print('[Tren box NHIEU]    truoc %.3f -> sau RL %.3f  (RL SUA box nhieu: %+.3f)'%(pi,po,po-pi))
