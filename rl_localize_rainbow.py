"""Active lesion localization bằng RAINBOW DQN (bỏ detector — reward lấy THẲNG từ ground-truth).
Agent bắt đầu từ cửa sổ = cả ảnh, thực hiện chuỗi hành động (dịch trái/phải/lên/xuống, to/nhỏ, béo/cao, TRIGGER)
để bám sát bbox tổn thương. Reward = dấu thay đổi IoU vs GT; trigger: +3 nếu IoU>=0.5, ngược lại -3.
State = roi_align(featmap ResNet18, cửa sổ) + lịch sử hành động. Chia train/test theo VIDEO.
Rainbow: Double + Dueling + Prioritized Replay + N-step + Noisy Nets + Distributional(C51)."""
import sys, math, random, numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.ops import roi_align
dev='cuda'; W=336
EPISODES=int(sys.argv[1]) if len(sys.argv)>1 else 12000
FEAT=sys.argv[2] if len(sys.argv)>2 else 'data/loc_feat.h5'
ROIG=int(sys.argv[3]) if len(sys.argv)>3 else 2
ALPHA=0.2; MAXSTEP=40; TAU=0.5; NA=8; HIST=10   # NA=8: bỏ trigger (navigation-only, tránh trigger-collapse)
NATOMS=51; VMIN,VMAX=-12.0,12.0; GAMMA=0.99; NSTEP=3
BATCH=32; BUF=60000; WARM=2000; TRAIN_EVERY=4; TGT_EVERY=1500; LR=6.25e-5
torch.manual_seed(0); np.random.seed(0); random.seed(0)
# ---------- data ----------
f=h5py.File(FEAT,'r'); keys=list(f.keys())
CH=f[keys[0]]['fm'].shape[0]; SPAT=f[keys[0]]['fm'].shape[1]; FS=SPAT/W
vids=sorted(set(f[k].attrs['vid'] for k in keys)); rng=np.random.RandomState(0); o=list(vids); rng.shuffle(o)
te=set(o[:max(3,int(0.25*len(o)))])
tr_keys=[k for k in keys if f[k].attrs['vid'] not in te]; te_keys=[k for k in keys if f[k].attrs['vid'] in te]
print('frame train %d | test %d | video test %d'%(len(tr_keys),len(te_keys),len(te)))
FM={k:torch.tensor(np.asarray(f[k]['fm']),dtype=torch.float32) for k in keys}  # cpu fp32
GT={k:np.asarray(f[k]['gt'],np.float32) for k in keys}
def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def diou(a,b):
    # Distance-IoU: IoU - rho^2(tâm)/c^2 (đường chéo enclosing) -> ĐẶC cả khi chưa giao (IoU=0)
    io=iou(a,b)
    cax,cay=(a[0]+a[2])/2,(a[1]+a[3])/2; cbx,cby=(b[0]+b[2])/2,(b[1]+b[3])/2
    rho2=(cax-cbx)**2+(cay-cby)**2
    ex0=min(a[0],b[0]);ey0=min(a[1],b[1]);ex1=max(a[2],b[2]);ey1=max(a[3],b[3])
    c2=(ex1-ex0)**2+(ey1-ey0)**2
    return io-rho2/c2 if c2>0 else io   # in [-1,1]
# ---------- environment ----------
class Env:
    def reset(s,k):
        s.k=k; s.fm=FM[k].to(dev).unsqueeze(0); s.gt=GT[k].tolist()
        s.box=[0.0,0.0,float(W),float(W)]; s.hist=[]; s.t=0; s.pi=iou(s.box,s.gt); s.pd=diou(s.box,s.gt); return s.state()
    def state(s):
        b=torch.tensor([[0.0,s.box[0]*FS,s.box[1]*FS,s.box[2]*FS,s.box[3]*FS]],device=dev)
        rf=roi_align(s.fm,b,output_size=(ROIG,ROIG),aligned=True).reshape(-1).cpu().numpy()
        h=np.zeros(NA*HIST,np.float32)
        for i,a in enumerate(s.hist[-HIST:]): h[i*NA+a]=1.0
        g=np.array([s.box[0]/W,s.box[1]/W,s.box[2]/W,s.box[3]/W],np.float32)
        return np.concatenate([rf,g,h]).astype(np.float32)
    def step(s,a):
        b=s.box; w=b[2]-b[0]; h=b[3]-b[1]; aw=ALPHA*w; ah=ALPHA*h; done=False; s.t+=1
        if a==8:  # trigger
            r=3.0 if iou(b,s.gt)>=TAU else -3.0; done=True; s.hist.append(a); return s.state(),r,done
        nb=b[:]
        if a==0: nb=[b[0]+aw,b[1],b[2]+aw,b[3]]
        elif a==1: nb=[b[0]-aw,b[1],b[2]-aw,b[3]]
        elif a==2: nb=[b[0],b[1]+ah,b[2],b[3]+ah]
        elif a==3: nb=[b[0],b[1]-ah,b[2],b[3]-ah]
        elif a==4: nb=[b[0]-aw/2,b[1]-ah/2,b[2]+aw/2,b[3]+ah/2]   # bigger
        elif a==5: nb=[b[0]+aw/2,b[1]+ah/2,b[2]-aw/2,b[3]-ah/2]   # smaller
        elif a==6: nb=[b[0],b[1]+ah/2,b[2],b[3]-ah/2]             # fatter/wider
        elif a==7: nb=[b[0]+aw/2,b[1],b[2]-aw/2,b[3]]             # taller
        nb=[max(0,min(W,nb[0])),max(0,min(W,nb[1])),max(0,min(W,nb[2])),max(0,min(W,nb[3]))]
        if nb[2]-nb[0]<8 or nb[3]-nb[1]<8: nb=b[:]                # giữ nguyên nếu suy biến
        nd=diou(nb,s.gt); r=1.0 if nd>s.pd+1e-6 else (-1.0 if nd<s.pd-1e-6 else -0.2)  # reward = dấu ΔDIoU (đặc, có khoảng cách-tâm)
        s.box=nb; s.pi=iou(nb,s.gt); s.pd=nd; s.hist.append(a)
        if s.t>=MAXSTEP: done=True
        return s.state(),r,done
SD=CH*ROIG*ROIG+4+NA*HIST; print('feat %s | CH %d | spat %d | ROIG %d | SD %d'%(FEAT,CH,SPAT,ROIG,SD))
# ---------- Noisy linear ----------
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
# ---------- Dueling + Distributional net ----------
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.body=nn.Sequential(nn.Linear(SD,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU())
        s.v1=Noisy(256,256); s.v2=Noisy(256,NATOMS)
        s.a1=Noisy(256,256); s.a2=Noisy(256,NA*NATOMS)
    def forward(s,x):
        h=s.body(x); v=s.v2(F.relu(s.v1(h))).view(-1,1,NATOMS); a=s.a2(F.relu(s.a1(h))).view(-1,NA,NATOMS)
        q=v+a-a.mean(1,keepdim=True); return F.log_softmax(q,dim=2)
    def reset_noise(s):
        for m in (s.v1,s.v2,s.a1,s.a2): m.reset_noise()
net=Net().to(dev); tgt=Net().to(dev); tgt.load_state_dict(net.state_dict()); tgt.eval()
opt=torch.optim.Adam(net.parameters(),lr=LR,eps=1.5e-4)
zsup=torch.linspace(VMIN,VMAX,NATOMS,device=dev); dz=(VMAX-VMIN)/(NATOMS-1)
# ---------- Prioritized replay ----------
class PER:
    def __init__(s,cap): s.cap=cap; s.S=np.zeros((cap,SD),np.float32); s.NS=np.zeros((cap,SD),np.float32)
    def _a(s):
        s.A=np.zeros(s.cap,np.int64); s.R=np.zeros(s.cap,np.float32); s.D=np.zeros(s.cap,np.float32)
        s.P=np.zeros(s.cap,np.float32); s.n=0; s.i=0
    def add(s,st,a,r,ns,d):
        i=s.i; s.S[i]=st; s.A[i]=a; s.R[i]=r; s.NS[i]=ns; s.D[i]=d
        s.P[i]=s.P[:s.n].max() if s.n>0 else 1.0; s.i=(i+1)%s.cap; s.n=min(s.n+1,s.cap)
    def sample(s,b,beta):
        p=s.P[:s.n]**0.5; p=p/p.sum(); idx=np.random.choice(s.n,b,p=p)
        w=(s.n*p[idx])**(-beta); w=w/w.max()
        return idx,w.astype(np.float32)
    def update(s,idx,pr): s.P[idx]=pr+1e-5
mem=PER(BUF); mem._a()
# ---------- n-step buffer ----------
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
    S=torch.tensor(mem.S[idx],device=dev); NS=torch.tensor(mem.NS[idx],device=dev)
    A=torch.tensor(mem.A[idx],device=dev); R=torch.tensor(mem.R[idx],device=dev); D=torch.tensor(mem.D[idx],device=dev)
    Wt=torch.tensor(w,device=dev)
    with torch.no_grad():
        net.reset_noise(); nlog=net(NS); nq=(nlog.exp()*zsup).sum(2); a_=nq.argmax(1)
        tgt.reset_noise(); tlog=tgt(NS); tp=tlog.exp()[range(BATCH),a_]  # (B,NATOMS)
        Tz=(R.unsqueeze(1)+(GAMMA**NSTEP)*zsup.unsqueeze(0)*(1-D.unsqueeze(1))).clamp(VMIN,VMAX)
        bj=((Tz-VMIN)/dz).clamp(0,NATOMS-1); l=bj.floor().long(); u=bj.ceil().long()
        l[(u>0)&(l==u)]-=1; u[(l<NATOMS-1)&(l==u)]+=1
        l=l.clamp(0,NATOMS-1); u=u.clamp(0,NATOMS-1)
        m=torch.zeros(BATCH,NATOMS,device=dev); off=torch.arange(BATCH,device=dev).unsqueeze(1)*NATOMS
        m.view(-1).index_add_(0,(l+off).view(-1),(tp*(u.float()-bj)).view(-1))
        m.view(-1).index_add_(0,(u+off).view(-1),(tp*(bj-l.float())).view(-1))
    net.reset_noise(); logp=net(S)[range(BATCH),A]
    loss_i=-(m*logp).sum(1); loss=(Wt*loss_i).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),10.0); opt.step()
    mem.update(idx,loss_i.detach().abs().cpu().numpy())
# ---------- eval ----------
@torch.no_grad()
def evaluate(klist,limit=400):
    net.eval(); env=Env(); ious=[];starts=[];rw=[]
    ks=klist if len(klist)<=limit else list(np.random.RandomState(1).choice(klist,limit,replace=False))
    for k in ks:
        st=env.reset(k); s0=env.pi
        for _ in range(MAXSTEP):
            q=(net(torch.tensor(st,device=dev).unsqueeze(0)).exp()*zsup).sum(2)
            a=int(q.argmax(1)); st,r,d=env.step(a)
            if d: break
        ious.append(iou(env.box,env.gt)); starts.append(s0)
        # random-walk baseline
        st=env.reset(k)
        for _ in range(MAXSTEP):
            a=random.randint(0,NA-1); st,r,d=env.step(a)
            if d: break
        rw.append(iou(env.box,env.gt))
    net.train()
    return np.mean(ious),(np.array(ious)>=.5).mean(),np.mean(starts),np.mean(rw),(np.array(rw)>=.5).mean()
# ---------- train ----------
env=Env(); step=0; ep_r=[]
for ep in range(1,EPISODES+1):
    k=random.choice(tr_keys); st=env.reset(k); nbuf.clear(); tot=0
    for _ in range(MAXSTEP):
        net.reset_noise()
        with torch.no_grad():
            q=(net(torch.tensor(st,device=dev).unsqueeze(0)).exp()*zsup).sum(2); a=int(q.argmax(1))
        ns,r,d=env.step(a); nbuf.append((st,a,r,ns,d)); push_n(force=d); st=ns; tot+=r; step+=1
        if mem.n>=WARM and step%TRAIN_EVERY==0:
            beta=min(1.0,0.4+0.6*step/(EPISODES*8)); learn(beta)
        if step%TGT_EVERY==0: tgt.load_state_dict(net.state_dict())
        if d: break
    ep_r.append(tot)
    if ep%500==0:
        mi,dr,s0,rwi,rwdr=evaluate(te_keys)
        print('ep %5d | step %7d | epR %.2f | TEST IoU %.3f DR %.1f%% | randwalk IoU %.3f DR %.1f%% | start IoU %.3f'
              %(ep,step,np.mean(ep_r[-500:]),mi,dr*100,rwi,rwdr*100,s0),flush=True)
mi,dr,s0,rwi,rwdr=evaluate(te_keys,limit=100000)
tri,tdr,_,_,_=evaluate(tr_keys,limit=800)
print('\n=== RAINBOW DQN LOCALIZATION (không detector) ===')
print('TEST : final IoU %.3f | DR@0.5 %.1f%%  (start %.3f, random-walk %.3f/%.1f%%)'%(mi,dr*100,s0,rwi,rwdr*100))
print('TRAIN: final IoU %.3f | DR@0.5 %.1f%%'%(tri,tdr*100))
print('(tham chiếu: detector Faster R-CNN v3 = IoU 0.610 / DR 76.4%%)')
torch.save(net.state_dict(),'data/rainbow_loc.pth')
