"""Rainbow DQN localization — STATE = TOÀN feature map + kênh vị trí cửa sổ (sửa nút thắt overfit).
Khác bản cũ (pool theo cửa sổ làm loãng tín hiệu): agent nhìn cả bản đồ đặc trưng 256x21x21 + 1 kênh mask
cửa sổ hiện tại + 2 kênh toạ độ; policy là CNN nhỏ. Reward = dấu ΔDIoU (khoảng cách-tới-tâm). Navigation-only.
Replay chỉ lưu (image_id, box) -> dựng state on-the-fly (feature map tĩnh dùng chung) -> nhẹ RAM.
Chia train/test theo VIDEO. Rainbow: Double+Dueling+PER+N-step+Noisy+C51."""
import sys, math, random, numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
dev='cuda'; W=336; G=21
EPISODES=int(sys.argv[1]) if len(sys.argv)>1 else 20000
ALPHA=0.2; MAXSTEP=40; NA=8
NATOMS=51; VMIN,VMAX=-12.0,12.0; GAMMA=0.99; NSTEP=3
BATCH=32; BUF=60000; WARM=2000; TRAIN_EVERY=4; TGT_EVERY=1500; LR=6.25e-5
torch.manual_seed(0); np.random.seed(0); random.seed(0)
# ---------- data ----------
f=h5py.File('data/loc_feat2.h5','r'); keys=list(f.keys())
CH=f[keys[0]]['fm'].shape[0]
FMall=np.stack([np.asarray(f[k]['fm']) for k in keys]).astype(np.float16)   # (N,256,21,21) CPU
GTall=np.stack([np.asarray(f[k]['gt']) for k in keys]).astype(np.float32)
vidall=[f[k].attrs['vid'] for k in keys]
vids=sorted(set(vidall)); rng=np.random.RandomState(0); o=list(vids); rng.shuffle(o); te=set(o[:max(3,int(0.25*len(o)))])
tr_ids=[i for i in range(len(keys)) if vidall[i] not in te]; te_ids=[i for i in range(len(keys)) if vidall[i] in te]
print('frame train %d | test %d | video test %d | fm %s'%(len(tr_ids),len(te_ids),len(te),FMall.shape[1:]))
# toạ độ tâm ô lưới + kênh toạ độ chuẩn hoá (precompute)
cxs=(torch.arange(G,dtype=torch.float32)+0.5)*W/G                 # (G,) tâm ô theo px
CX=(cxs/W).view(1,G).expand(G,G); CY=(cxs/W).view(G,1).expand(G,G)
COORD=torch.stack([CX,CY]).to(dev)                                # (2,G,G)
cxg=cxs.to(dev)
def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def diou(a,b):
    io=iou(a,b); cax,cay=(a[0]+a[2])/2,(a[1]+a[3])/2; cbx,cby=(b[0]+b[2])/2,(b[1]+b[3])/2
    rho2=(cax-cbx)**2+(cay-cby)**2
    ex0=min(a[0],b[0]);ey0=min(a[1],b[1]);ex1=max(a[2],b[2]);ey1=max(a[3],b[3]); c2=(ex1-ex0)**2+(ey1-ey0)**2
    return io-rho2/c2 if c2>0 else io
def build(ids,boxes):
    feat=torch.from_numpy(FMall[np.asarray(ids)]).to(dev).float()          # (B,256,G,G)
    b=torch.tensor(np.asarray(boxes,np.float32),device=dev)                 # (B,4)
    mx=(cxg.view(1,G)>=b[:,0:1])&(cxg.view(1,G)<=b[:,2:3])                   # (B,G) cột
    my=(cxg.view(1,G)>=b[:,1:2])&(cxg.view(1,G)<=b[:,3:4])                   # (B,G) hàng
    mask=(my.unsqueeze(2)&mx.unsqueeze(1)).float().unsqueeze(1)             # (B,1,G,G)
    coord=COORD.unsqueeze(0).expand(feat.size(0),-1,-1,-1)
    return torch.cat([feat,mask,coord],1)                                   # (B,256+3,G,G)
INC=CH+3
# ---------- env (navigation-only) ----------
class Env:
    def reset(s,i):
        s.i=i; s.gt=GTall[i].tolist(); s.box=[0.0,0.0,float(W),float(W)]; s.t=0; s.pd=diou(s.box,s.gt); return s.box
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
        if nb[2]-nb[0]<8 or nb[3]-nb[1]<8: nb=b[:]
        nd=diou(nb,s.gt); r=1.0 if nd>s.pd+1e-6 else (-1.0 if nd<s.pd-1e-6 else -0.2)
        s.box=nb; s.pd=nd; done=s.t>=MAXSTEP
        return nb,r,done
# ---------- Noisy ----------
class Noisy(nn.Module):
    def __init__(s,i,o,sig=0.5):
        super().__init__(); s.i,s.o=i,o
        s.wm=nn.Parameter(torch.empty(o,i)); s.ws=nn.Parameter(torch.empty(o,i))
        s.bm=nn.Parameter(torch.empty(o)); s.bs=nn.Parameter(torch.empty(o))
        s.register_buffer('we',torch.empty(o,i)); s.register_buffer('be',torch.empty(o))
        mu=1/math.sqrt(i); s.wm.data.uniform_(-mu,mu); s.bm.data.uniform_(-mu,mu)
        s.ws.data.fill_(sig/math.sqrt(i)); s.bs.data.fill_(sig/math.sqrt(o)); s.reset_noise()
    def _n(s,n): x=torch.randn(n); return x.sign()*x.abs().sqrt()
    def reset_noise(s):
        ei=s._n(s.i); eo=s._n(s.o); s.we.copy_(eo.ger(ei)); s.be.copy_(eo)
    def forward(s,x):
        if s.training: return F.linear(x,s.wm+s.ws*s.we,s.bm+s.bs*s.be)
        return F.linear(x,s.wm,s.bm)
# ---------- conv dueling + C51 ----------
class Net(nn.Module):
    def __init__(s,inc):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(inc,64,3,padding=1),nn.ReLU(),
                          nn.Conv2d(64,64,3,padding=1,stride=2),nn.ReLU(),
                          nn.Conv2d(64,64,3,padding=1,stride=2),nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1))
        s.v1=Noisy(64,128); s.v2=Noisy(128,NATOMS); s.a1=Noisy(64,128); s.a2=Noisy(128,NA*NATOMS)
    def forward(s,x):
        h=s.c(x).flatten(1); v=s.v2(F.relu(s.v1(h))).view(-1,1,NATOMS); a=s.a2(F.relu(s.a1(h))).view(-1,NA,NATOMS)
        q=v+a-a.mean(1,keepdim=True); return F.log_softmax(q,dim=2)
    def reset_noise(s):
        for m in (s.v1,s.v2,s.a1,s.a2): m.reset_noise()
net=Net(INC).to(dev); tgt=Net(INC).to(dev); tgt.load_state_dict(net.state_dict()); tgt.eval()
opt=torch.optim.Adam(net.parameters(),lr=LR,eps=1.5e-4)
zsup=torch.linspace(VMIN,VMAX,NATOMS,device=dev); dz=(VMAX-VMIN)/(NATOMS-1)
# ---------- PER (lưu id+box, dựng state khi học) ----------
class PER:
    def __init__(s,cap):
        s.cap=cap; s.ID=np.zeros(cap,np.int64); s.BOX=np.zeros((cap,4),np.float32); s.NBOX=np.zeros((cap,4),np.float32)
        s.A=np.zeros(cap,np.int64); s.R=np.zeros(cap,np.float32); s.D=np.zeros(cap,np.float32); s.P=np.zeros(cap,np.float32); s.n=0; s.i=0
    def add(s,i,box,a,r,nbox,d):
        j=s.i; s.ID[j]=i; s.BOX[j]=box; s.A[j]=a; s.R[j]=r; s.NBOX[j]=nbox; s.D[j]=d
        s.P[j]=s.P[:s.n].max() if s.n>0 else 1.0; s.i=(j+1)%s.cap; s.n=min(s.n+1,s.cap)
    def sample(s,b,beta):
        p=s.P[:s.n]**0.5; p=p/p.sum(); idx=np.random.choice(s.n,b,p=p); w=(s.n*p[idx])**(-beta); return idx,(w/w.max()).astype(np.float32)
    def update(s,idx,pr): s.P[idx]=pr+1e-5
mem=PER(BUF)
from collections import deque
nbuf=deque()
def push_n(force=False):
    while len(nbuf)>=NSTEP or (force and nbuf):
        R=0.0
        for j in range(min(NSTEP,len(nbuf))): R+=(GAMMA**j)*nbuf[j][3]
        i0,b0,a0=nbuf[0][0],nbuf[0][1],nbuf[0][2]; last=nbuf[min(NSTEP,len(nbuf))-1]; nb,d=last[4],last[5]
        mem.add(i0,b0,a0,R,nb,1.0 if d else 0.0); nbuf.popleft()
        if not force and len(nbuf)<NSTEP: break
def learn(beta):
    idx,w=mem.sample(BATCH,beta)
    S=build(mem.ID[idx],mem.BOX[idx]); NS=build(mem.ID[idx],mem.NBOX[idx])
    A=torch.tensor(mem.A[idx],device=dev); R=torch.tensor(mem.R[idx],device=dev); D=torch.tensor(mem.D[idx],device=dev); Wt=torch.tensor(w,device=dev)
    with torch.no_grad():
        net.reset_noise(); nq=(net(NS).exp()*zsup).sum(2); a_=nq.argmax(1)
        tgt.reset_noise(); tp=tgt(NS).exp()[range(BATCH),a_]
        Tz=(R.unsqueeze(1)+(GAMMA**NSTEP)*zsup.unsqueeze(0)*(1-D.unsqueeze(1))).clamp(VMIN,VMAX)
        bj=((Tz-VMIN)/dz).clamp(0,NATOMS-1); l=bj.floor().long(); u=bj.ceil().long()
        l[(u>0)&(l==u)]-=1; u[(l<NATOMS-1)&(l==u)]+=1; l=l.clamp(0,NATOMS-1); u=u.clamp(0,NATOMS-1)
        m=torch.zeros(BATCH,NATOMS,device=dev); off=torch.arange(BATCH,device=dev).unsqueeze(1)*NATOMS
        m.view(-1).index_add_(0,(l+off).view(-1),(tp*(u.float()-bj)).view(-1))
        m.view(-1).index_add_(0,(u+off).view(-1),(tp*(bj-l.float())).view(-1))
    net.reset_noise(); logp=net(S)[range(BATCH),A]; loss_i=-(m*logp).sum(1); loss=(Wt*loss_i).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),10.0); opt.step()
    mem.update(idx,loss_i.detach().abs().cpu().numpy())
@torch.no_grad()
def evaluate(idlist,limit=400):
    net.eval(); env=Env(); ious=[];rw=[]
    ids=idlist if len(idlist)<=limit else list(np.random.RandomState(1).choice(idlist,limit,replace=False))
    for i in ids:
        env.reset(i)
        for _ in range(MAXSTEP):
            q=(net(build([i],[env.box])).exp()*zsup).sum(2); a=int(q.argmax(1)); _,_,d=env.step(a)
            if d: break
        ious.append(iou(env.box,env.gt))
        env.reset(i)
        for _ in range(MAXSTEP):
            _,_,d=env.step(random.randint(0,NA-1))
            if d: break
        rw.append(iou(env.box,env.gt))
    net.train(); return np.mean(ious),(np.array(ious)>=.5).mean(),np.mean(rw)
# ---------- train ----------
env=Env(); step=0; ep_r=[]
for ep in range(1,EPISODES+1):
    i=random.choice(tr_ids); env.reset(i); nbuf.clear(); tot=0
    for _ in range(MAXSTEP):
        net.reset_noise()
        with torch.no_grad():
            q=(net(build([i],[env.box])).exp()*zsup).sum(2); a=int(q.argmax(1))
        b0=env.box[:]; nb,r,d=env.step(a); nbuf.append((i,b0,a,r,nb,d)); push_n(force=d); tot+=r; step+=1
        if mem.n>=WARM and step%TRAIN_EVERY==0:
            beta=min(1.0,0.4+0.6*step/(EPISODES*8)); learn(beta)
        if step%TGT_EVERY==0: tgt.load_state_dict(net.state_dict())
        if d: break
    ep_r.append(tot)
    if ep%500==0:
        mi,dr,rw=evaluate(te_ids); print('ep %5d | step %7d | epR %.2f | TEST IoU %.3f DR %.1f%% | randwalk %.3f'%(ep,step,np.mean(ep_r[-500:]),mi,dr*100,rw),flush=True)
mi,dr,rw=evaluate(te_ids,10**9); tri,tdr,_=evaluate(tr_ids,800)
print('\n=== RAINBOW DQN LOCALIZATION — MAP-OBS (toàn feature map + kênh vị trí) ===')
print('TEST : IoU %.3f | DR@0.5 %.1f%%  (random-walk %.3f)'%(mi,dr*100,rw))
print('TRAIN: IoU %.3f | DR@0.5 %.1f%%'%(tri,tdr*100))
print('(tham chiếu detector v3 = 0.610/76.4%% ; bản pool-cửa-sổ cũ = test 0.07-0.11)')
torch.save(net.state_dict(),'data/rainbow_loc_mapobs.pth')
