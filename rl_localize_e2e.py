"""Rainbow DQN localization — END-TO-END, KHÔNG detector, KHÔNG feature pretrain.
Mắt agent = CNN học từ SỐ 0 chỉ bằng reward ΔDIoU. Quan sát = ảnh thô RGB 84x84 + kênh mask cửa sổ.
AUGMENTATION mỗi episode (lật ngang/dọc + sáng/tương phản) để nhân đa dạng, chống overfit 16 video.
Rainbow: Double+Dueling+PER+N-step+Noisy+C51. Navigation-only. Chia train/test theo VIDEO."""
import sys, math, random, numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
dev='cuda'; W=84
EPISODES=int(sys.argv[1]) if len(sys.argv)>1 else 20000
ALPHA=0.2; MAXSTEP=40; NA=8
NATOMS=51; VMIN,VMAX=-12.0,12.0; GAMMA=0.99; NSTEP=3
BATCH=32; BUF=40000; WARM=2000; TRAIN_EVERY=4; TGT_EVERY=1500; LR=6.25e-5
AUG=int(sys.argv[2]) if len(sys.argv)>2 else 1
torch.manual_seed(0); np.random.seed(0); random.seed(0)
f=h5py.File('data/loc_raw.h5','r'); IMG=np.asarray(f['img']); GT=np.asarray(f['gt'],np.float32); VID=[v.decode() for v in f['vid']]
vids=sorted(set(VID)); rng=np.random.RandomState(0); o=list(vids); rng.shuffle(o); te=set(o[:max(3,int(0.25*len(o)))])
tr_ids=[i for i in range(len(IMG)) if VID[i] not in te]; te_ids=[i for i in range(len(IMG)) if VID[i] in te]
print('frame train %d | test %d | video test %d | AUG %d'%(len(tr_ids),len(te_ids),len(te),AUG))
def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def diou(a,b):
    io=iou(a,b); cax,cay=(a[0]+a[2])/2,(a[1]+a[3])/2; cbx,cby=(b[0]+b[2])/2,(b[1]+b[3])/2
    rho2=(cax-cbx)**2+(cay-cby)**2
    ex0=min(a[0],b[0]);ey0=min(a[1],b[1]);ex1=max(a[2],b[2]);ey1=max(a[3],b[3]); c2=(ex1-ex0)**2+(ey1-ey0)**2
    return io-rho2/c2 if c2>0 else io
def obs_of(img,box):  # img uint8 (84,84,3) đã augment; box thang 84 -> (4,84,84) float
    o=np.zeros((4,W,W),np.float32); o[:3]=img.transpose(2,0,1)/255.0
    x1,y1,x2,y2=[int(round(v)) for v in box]; x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(W,y2)
    o[3,y1:y2,x1:x2]=1.0; return o
class Env:
    def reset(s,i,aug):
        img=IMG[i].copy(); gt=GT[i].tolist()
        if aug:
            if random.random()<0.5: img=img[:,::-1,:].copy(); gt=[W-gt[2],gt[1],W-gt[0],gt[3]]   # hflip
            if random.random()<0.5: img=img[::-1,:,:].copy(); gt=[gt[0],W-gt[3],gt[2],W-gt[1]]   # vflip
            c=random.uniform(0.8,1.2); b=random.uniform(-20,20)
            img=np.clip(img.astype(np.float32)*c+b,0,255).astype(np.uint8)                        # sáng/tương phản
        s.img=img; s.gt=gt; s.box=[0.0,0.0,float(W),float(W)]; s.t=0; s.pd=diou(s.box,gt); return obs_of(img,s.box)
    def step(s,a):
        b=s.box; w=b[2]-b[0]; h=b[3]-b[1]; aw=ALPHA*w; ah=ALPHA*h; s.t+=1
        if a==0: nb=[b[0]+aw,b[1],b[2]+aw,b[3]]
        elif a==1: nb=[b[0]-aw,b[1],b[2]-aw,b[3]]
        elif a==2: nb=[b[0],b[1]+ah,b[2],b[3]+ah]
        elif a==3: nb=[b[0],b[1]-ah,b[2],b[3]-ah]
        elif a==4: nb=[b[0]-aw/2,b[1]-ah/2,b[2]+aw/2,b[3]+ah/2]
        elif a==5: nb=[b[0]+aw/2,b[1]+ah/2,b[2]-aw/2,b[3]-ah/2]
        elif a==6: nb=[b[0],b[1]+ah/2,b[2],b[3]-ah/2]
        elif a==7: nb=[b[0]+aw/2,b[1],b[2]-aw/2,b[3]]
        nb=[max(0,min(W,nb[0])),max(0,min(W,nb[1])),max(0,min(W,nb[2])),max(0,min(W,nb[3]))]
        if nb[2]-nb[0]<4 or nb[3]-nb[1]<4: nb=b[:]
        nd=diou(nb,s.gt); r=1.0 if nd>s.pd+1e-6 else (-1.0 if nd<s.pd-1e-6 else -0.2)
        s.box=nb; s.pd=nd; return obs_of(s.img,nb),r,s.t>=MAXSTEP
class Noisy(nn.Module):
    def __init__(s,i,o,sig=0.5):
        super().__init__(); s.i,s.o=i,o
        s.wm=nn.Parameter(torch.empty(o,i)); s.ws=nn.Parameter(torch.empty(o,i)); s.bm=nn.Parameter(torch.empty(o)); s.bs=nn.Parameter(torch.empty(o))
        s.register_buffer('we',torch.empty(o,i)); s.register_buffer('be',torch.empty(o))
        mu=1/math.sqrt(i); s.wm.data.uniform_(-mu,mu); s.bm.data.uniform_(-mu,mu); s.ws.data.fill_(sig/math.sqrt(i)); s.bs.data.fill_(sig/math.sqrt(o)); s.reset_noise()
    def _n(s,n): x=torch.randn(n); return x.sign()*x.abs().sqrt()
    def reset_noise(s): ei=s._n(s.i); eo=s._n(s.o); s.we.copy_(eo.ger(ei)); s.be.copy_(eo)
    def forward(s,x): return F.linear(x,s.wm+s.ws*s.we,s.bm+s.bs*s.be) if s.training else F.linear(x,s.wm,s.bm)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(4,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad(): d=s.c(torch.zeros(1,4,W,W)).shape[1]
        s.fc=nn.Sequential(nn.Linear(d,512),nn.ReLU())
        s.v1=Noisy(512,256); s.v2=Noisy(256,NATOMS); s.a1=Noisy(512,256); s.a2=Noisy(256,NA*NATOMS)
    def forward(s,x):
        h=s.fc(s.c(x)); v=s.v2(F.relu(s.v1(h))).view(-1,1,NATOMS); a=s.a2(F.relu(s.a1(h))).view(-1,NA,NATOMS)
        return F.log_softmax(v+a-a.mean(1,keepdim=True),dim=2)
    def reset_noise(s):
        for m in (s.v1,s.v2,s.a1,s.a2): m.reset_noise()
net=Net().to(dev); tgt=Net().to(dev); tgt.load_state_dict(net.state_dict()); tgt.eval()
opt=torch.optim.Adam(net.parameters(),lr=LR,eps=1.5e-4)
zsup=torch.linspace(VMIN,VMAX,NATOMS,device=dev); dz=(VMAX-VMIN)/(NATOMS-1)
class PER:
    def __init__(s,cap):
        s.cap=cap; s.S=np.zeros((cap,4,W,W),np.float16); s.NS=np.zeros((cap,4,W,W),np.float16)
        s.A=np.zeros(cap,np.int64); s.R=np.zeros(cap,np.float32); s.D=np.zeros(cap,np.float32); s.P=np.zeros(cap,np.float32); s.n=0; s.i=0
    def add(s,st,a,r,ns,d):
        j=s.i; s.S[j]=st; s.NS[j]=ns; s.A[j]=a; s.R[j]=r; s.D[j]=d; s.P[j]=s.P[:s.n].max() if s.n>0 else 1.0; s.i=(j+1)%s.cap; s.n=min(s.n+1,s.cap)
    def sample(s,b,beta):
        p=s.P[:s.n]**0.5; p=p/p.sum(); idx=np.random.choice(s.n,b,p=p); w=(s.n*p[idx])**(-beta); return idx,(w/w.max()).astype(np.float32)
    def update(s,idx,pr): s.P[idx]=pr+1e-5
mem=PER(BUF)
from collections import deque
nbuf=deque()
def push_n(force=False):
    while len(nbuf)>=NSTEP or (force and nbuf):
        R=0.0
        for j in range(min(NSTEP,len(nbuf))): R+=(GAMMA**j)*nbuf[j][2]
        s0,a0=nbuf[0][0],nbuf[0][1]; last=nbuf[min(NSTEP,len(nbuf))-1]; ns,d=last[3],last[4]
        mem.add(s0,a0,R,ns,1.0 if d else 0.0); nbuf.popleft()
        if not force and len(nbuf)<NSTEP: break
def learn(beta):
    idx,w=mem.sample(BATCH,beta)
    S=torch.tensor(mem.S[idx],dtype=torch.float32,device=dev); NS=torch.tensor(mem.NS[idx],dtype=torch.float32,device=dev)
    A=torch.tensor(mem.A[idx],device=dev); R=torch.tensor(mem.R[idx],device=dev); D=torch.tensor(mem.D[idx],device=dev); Wt=torch.tensor(w,device=dev)
    with torch.no_grad():
        net.reset_noise(); a_=(net(NS).exp()*zsup).sum(2).argmax(1); tgt.reset_noise(); tp=tgt(NS).exp()[range(BATCH),a_]
        Tz=(R.unsqueeze(1)+(GAMMA**NSTEP)*zsup.unsqueeze(0)*(1-D.unsqueeze(1))).clamp(VMIN,VMAX)
        bj=((Tz-VMIN)/dz).clamp(0,NATOMS-1); l=bj.floor().long(); u=bj.ceil().long()
        l[(u>0)&(l==u)]-=1; u[(l<NATOMS-1)&(l==u)]+=1; l=l.clamp(0,NATOMS-1); u=u.clamp(0,NATOMS-1)
        m=torch.zeros(BATCH,NATOMS,device=dev); off=torch.arange(BATCH,device=dev).unsqueeze(1)*NATOMS
        m.view(-1).index_add_(0,(l+off).view(-1),(tp*(u.float()-bj)).view(-1)); m.view(-1).index_add_(0,(u+off).view(-1),(tp*(bj-l.float())).view(-1))
    net.reset_noise(); logp=net(S)[range(BATCH),A]; loss_i=-(m*logp).sum(1); loss=(Wt*loss_i).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),10.0); opt.step(); mem.update(idx,loss_i.detach().abs().cpu().numpy())
@torch.no_grad()
def evaluate(idlist,limit=400):
    net.eval(); env=Env(); ious=[];rw=[]
    ids=idlist if len(idlist)<=limit else list(np.random.RandomState(1).choice(idlist,limit,replace=False))
    for i in ids:
        st=env.reset(i,0)
        for _ in range(MAXSTEP):
            a=int((net(torch.tensor(st,device=dev).unsqueeze(0)).exp()*zsup).sum(2).argmax(1)); st,_,d=env.step(a)
            if d: break
        ious.append(iou(env.box,env.gt))
        env.reset(i,0)
        for _ in range(MAXSTEP):
            _,_,d=env.step(random.randint(0,NA-1))
            if d: break
        rw.append(iou(env.box,env.gt))
    net.train(); return np.mean(ious),(np.array(ious)>=.5).mean(),np.mean(rw)
env=Env(); step=0; ep_r=[]
for ep in range(1,EPISODES+1):
    i=random.choice(tr_ids); st=env.reset(i,AUG); nbuf.clear(); tot=0
    for _ in range(MAXSTEP):
        net.reset_noise()
        with torch.no_grad(): a=int((net(torch.tensor(st,device=dev).unsqueeze(0)).exp()*zsup).sum(2).argmax(1))
        ns,r,d=env.step(a); nbuf.append((st,a,r,ns,d)); push_n(force=d); st=ns; tot+=r; step+=1
        if mem.n>=WARM and step%TRAIN_EVERY==0:
            beta=min(1.0,0.4+0.6*step/(EPISODES*8)); learn(beta)
        if step%TGT_EVERY==0: tgt.load_state_dict(net.state_dict())
        if d: break
    ep_r.append(tot)
    if ep%500==0:
        mi,dr,rw=evaluate(te_ids); print('ep %5d | step %7d | epR %.2f | TEST IoU %.3f DR %.1f%% | randwalk %.3f'%(ep,step,np.mean(ep_r[-500:]),mi,dr*100,rw),flush=True)
mi,dr,rw=evaluate(te_ids,10**9); tri,tdr,_=evaluate(tr_ids,800)
print('\n=== RAINBOW DQN LOCALIZATION — END-TO-END (không detector, không pretrain) ===')
print('TEST : IoU %.3f | DR@0.5 %.1f%%  (random-walk %.3f)'%(mi,dr*100,rw))
print('TRAIN: IoU %.3f | DR@0.5 %.1f%%'%(tri,tdr*100))
print('(bản feature đông cứng trước = test 0.07-0.11 ; detector tham chiếu 0.61/76%%)')
torch.save(net.state_dict(),'data/rainbow_loc_e2e.pth')
