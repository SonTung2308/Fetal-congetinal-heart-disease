"""RL selector cho Kvasir: hoc chon SET OF FRAMES phu het cac DOAN benh o ngan sach thap.
Reward = segment_recall - lambda*budget. REINFORCE, policy GRU tren [score, smooth, pos].
So voi tham lam top-k% theo score. Chia theo video (k-fold)."""
import numpy as np, h5py, torch, torch.nn as nn
dev='cuda'; GAP=45; LAM=0.5; EPISODES=8; EPOCHS=40
fs=h5py.File('data/kvasir_video_score_full.h5','r'); fl=h5py.File('data/kvasir_video_fulllabels.h5','r')
vids=[v for v in fs.keys() if fl[v]['gtscore'][...].sum()>0]
def smooth(p,w): k=np.ones(w)/w; return np.convolve(p,k,mode='same')
def segs(g):
    idx=np.where(g>0)[0]
    if len(idx)==0: return []
    s=[[idx[0],idx[0]]]
    for i in idx[1:]:
        if i-s[-1][1]<=GAP: s[-1][1]=i
        else: s.append([i,i])
    return s
# input features per video (chuan hoa)
feat={}; segd={}
for v in vids:
    sc=fs[v]['score'][...]; n=len(sc); pos=np.linspace(0,1,n)
    x=np.stack([sc,smooth(sc,9),smooth(sc,31),pos],1).astype(np.float32)
    feat[v]=x; segd[v]=segs(fl[v]['gtscore'][...])
def seg_recall(v,sel):  # sel: bool mask
    S=segd[v]
    if not S: return 1.0
    hit=sum(1 for s in S if sel[s[0]:s[1]+1].any())
    return hit/len(S)
class Pol(nn.Module):
    def __init__(s,d=4,h=64):
        super().__init__(); s.rnn=nn.GRU(d,h,batch_first=True,bidirectional=True)
        s.head=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): o,_=s.rnn(x); return torch.sigmoid(s.head(o)).squeeze(-1)

rng=np.random.RandomState(0); order=np.array(vids); rng.shuffle(order)
folds=np.array_split(order,5)
rl_keepprob={}
for te in folds:
    te=set(te.tolist()); tr=[v for v in vids if v not in te]
    net=Pol().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    base={v:0.0 for v in tr}
    for ep in range(EPOCHS):
        for v in tr:
            x=torch.from_numpy(feat[v]).unsqueeze(0).to(dev)
            p=net(x).squeeze(0)                       # keep prob per frame
            m=torch.distributions.Bernoulli(p.clamp(1e-4,1-1e-4))
            loss=0
            for _ in range(EPISODES):
                a=m.sample(); sel=a.bool().cpu().numpy()
                r=seg_recall(v,sel)-LAM*a.float().mean().item()
                base[v]=0.9*base[v]+0.1*r
                logp=m.log_prob(a).mean()
                loss=loss-(logp*(r-base[v]))
            loss=loss/EPISODES
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for v in te:
        with torch.no_grad():
            rl_keepprob[v]=net(torch.from_numpy(feat[v]).unsqueeze(0).to(dev)).squeeze(0).cpu().numpy()

# so sanh: RL keepprob vs score tham lam, o cung ngan sach
print('So sanh SEGMENT-RECALL o cung ngan sach (RL vs tham lam top-k%):')
print('%8s %12s %12s'%('budget','tham lam','RL'))
allseg=[s for v in vids for s in segd[v]]
for kb in [5,10,20,30]:
    hg=0; hr=0
    for v in vids:
        n=len(feat[v]); lim=max(1,int(n*kb/100))
        sc=smooth(fs[v]['score'][...],9)
        selg=np.zeros(n,bool); selg[np.argsort(sc)[-lim:]]=True
        selr=np.zeros(n,bool); selr[np.argsort(rl_keepprob[v])[-lim:]]=True
        for s in segd[v]:
            if selg[s[0]:s[1]+1].any(): hg+=1
            if selr[s[0]:s[1]+1].any(): hr+=1
    print('%7d%% %11.0f%% %11.0f%%'%(kb,hg/len(allseg)*100,hr/len(allseg)*100))
