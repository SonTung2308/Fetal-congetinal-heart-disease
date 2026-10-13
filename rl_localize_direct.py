"""RL localization — CƠ CHẾ KHÁC: policy xuất TRỰC TIẾP box (1 bước), KHÔNG detector, KHÔNG tìm-kiếm-tuần-tự.
CNN(ảnh 84) -> mean box; lấy mẫu Gaussian; reward = DIoU(box,GT); REINFORCE + value baseline + entropy.
= 'reward-driven regression'. Kiểm tra: thủ phạm overfit có phải do tìm-kiếm-tuần-tự? Chia test theo VIDEO. Có augment."""
import sys, numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F, random
dev='cuda'; W=84
STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 8000
AUG=int(sys.argv[2]) if len(sys.argv)>2 else 1
B=64; LR=3e-4
torch.manual_seed(0); np.random.seed(0); random.seed(0)
f=h5py.File('data/loc_raw.h5','r'); IMG=np.asarray(f['img']); GT=np.asarray(f['gt'],np.float32); VID=[v.decode() for v in f['vid']]
vids=sorted(set(VID)); rng=np.random.RandomState(0); o=list(vids); rng.shuffle(o); te=set(o[:max(3,int(0.25*len(o)))])
tr=np.array([i for i in range(len(IMG)) if VID[i] not in te]); teI=np.array([i for i in range(len(IMG)) if VID[i] in te])
print('frame train %d | test %d | video test %d | AUG %d'%(len(tr),len(teI),len(te),AUG))
def batch_imgs(ids,aug):
    ims=IMG[ids].astype(np.float32).copy(); gts=GT[ids].copy()
    if aug:
        for j in range(len(ids)):
            if random.random()<0.5: ims[j]=ims[j,:,::-1,:]; x1,y1,x2,y2=gts[j]; gts[j]=[W-x2,y1,W-x1,y2]
            if random.random()<0.5: ims[j]=ims[j,::-1,:,:]; x1,y1,x2,y2=gts[j]; gts[j]=[x1,W-y2,x2,W-y1]
        ims=np.clip(ims*random.uniform(0.8,1.2)+random.uniform(-20,20),0,255)
    x=torch.tensor(ims.transpose(0,3,1,2)/255.0,dtype=torch.float32,device=dev)
    return x,torch.tensor(gts,dtype=torch.float32,device=dev)
def iou_t(a,b):  # (N,4) xyxy
    x0=torch.max(a[:,0],b[:,0]);y0=torch.max(a[:,1],b[:,1]);x1=torch.min(a[:,2],b[:,2]);y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0)
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)
def diou_t(a,b):
    io=iou_t(a,b); ca=(a[:,:2]+a[:,2:])/2; cb=(b[:,:2]+b[:,2:])/2; rho2=((ca-cb)**2).sum(1)
    ex0=torch.min(a[:,0],b[:,0]);ey0=torch.min(a[:,1],b[:,1]);ex1=torch.max(a[:,2],b[:,2]);ey1=torch.max(a[:,3],b[:,3])
    c2=((ex1-ex0)**2+(ey1-ey0)**2).clamp(min=1e-6); return io-rho2/c2
def to_box(p):  # p:(N,4) in (0,1) [cx,cy,bw,bh] -> xyxy (pixel)
    cx=p[:,0]*W; cy=p[:,1]*W; bw=(0.05+0.95*p[:,2])*W; bh=(0.05+0.95*p[:,3])*W
    return torch.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],1).clamp(0,W)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(nn.Conv2d(3,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad(): d=s.c(torch.zeros(1,3,W,W)).shape[1]
        s.fc=nn.Sequential(nn.Linear(d,512),nn.ReLU()); s.mu=nn.Linear(512,4); s.val=nn.Linear(512,1)
        s.logstd=nn.Parameter(torch.zeros(4)-0.7)
    def forward(s,x): h=s.fc(s.c(x)); return s.mu(h), s.val(h).squeeze(1)
net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=LR)
def evaluate(ids):
    net.eval(); I=[]
    with torch.no_grad():
        for k in range(0,len(ids),256):
            x,g=batch_imgs(ids[k:k+256],0); mu,_=net(x); box=to_box(torch.sigmoid(mu)); I.append(iou_t(box,g).cpu().numpy())
    net.train(); I=np.concatenate(I); return I.mean(),(I>=.5).mean()
best=0
for it in range(1,STEPS+1):
    ids=tr[np.random.randint(0,len(tr),B)]; x,g=batch_imgs(ids,AUG)
    mu,V=net(x); std=net.logstd.exp().clamp(0.05,1.0)
    dist=torch.distributions.Normal(mu,std); z=dist.rsample() if False else dist.sample()
    box=to_box(torch.sigmoid(z)); R=diou_t(box,g)                      # reward in [-1,1]
    logp=dist.log_prob(z).sum(1); adv=(R-V).detach()
    ploss=-(adv*logp).mean(); vloss=F.mse_loss(V,R); ent=dist.entropy().sum(1).mean()
    loss=ploss+0.5*vloss-0.01*ent
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
    if it%500==0:
        mi,dr=evaluate(teI); tri,_=evaluate(tr[:800])
        if mi>best: best=mi
        print('it %5d | R %.3f | TEST IoU %.3f DR %.1f%% | TRAIN IoU %.3f | std %.2f'%(it,R.mean().item(),mi,dr*100,tri,std.mean().item()),flush=True)
mi,dr=evaluate(teI); tri,tdr=evaluate(tr)
print('\n=== RL DIRECT-BOX (không detector, không tìm-kiếm-tuần-tự) ===')
print('TEST : IoU %.3f (best %.3f) | DR@0.5 %.1f%%'%(mi,best,dr*100))
print('TRAIN: IoU %.3f | DR@0.5 %.1f%%'%(tri,tdr*100))
print('(sequential-search trước = test 0.06-0.11 ; detector 0.61/76%%)')
