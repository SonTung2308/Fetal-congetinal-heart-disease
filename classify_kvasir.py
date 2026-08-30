"""Bo phan loai frame co giam sat: ton thuong vs binh thuong tren feature 512-d.
Chia tap theo VIDEO (tranh ro ri). Do precision@k va AUC tren video test.
Muc tieu: % frame chon ra la ton thuong >> baseline."""
import sys, math, numpy as np, h5py, torch, torch.nn as nn
sys.path.insert(0,'scripts')


H5='data/kvasir_capsule.h5'
dev='cuda' if torch.cuda.is_available() else 'cpu'
SEEDS=int(sys.argv[1]) if len(sys.argv)>1 else 5
EPOCHS=60

f=h5py.File(H5,'r'); ks=list(f.keys())
feats={k:f[k]['features'][...].astype(np.float32) for k in ks}
labs ={k:f[k]['gtscore'][...].astype(np.float32) for k in ks}
mixed=[k for k in ks if 0<labs[k].mean()<1]

class Clf(nn.Module):
    def __init__(s,d=512,h=256,p=0.4):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(p),
                            nn.Linear(h,h),nn.ReLU(),nn.Dropout(p),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)

def run(seed):
    rng=np.random.RandomState(seed)
    vids=[k for k in ks if labs[k].max()>0]        # video co it nhat 1 ton thuong
    rng.shuffle(vids)
    nte=max(4,int(0.3*len(vids)))
    te=set(vids[:nte]); tr=[k for k in ks if k not in te]
    Xtr=np.concatenate([feats[k] for k in tr]); ytr=np.concatenate([labs[k] for k in tr])
    # can bang lop bang trong so
    pos=ytr.sum(); w=torch.tensor([(len(ytr)-pos)/max(pos,1)],device=dev)
    Xtr_t=torch.from_numpy(Xtr).to(dev); ytr_t=torch.from_numpy(ytr).to(dev)
    mu,sg=Xtr_t.mean(0),Xtr_t.std(0)+1e-6
    Xtr_t=(Xtr_t-mu)/sg
    torch.manual_seed(seed)
    net=Clf().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    n=len(ytr_t); g=torch.Generator(); g.manual_seed(seed)
    for ep in range(EPOCHS):
        net.train(); perm=torch.randperm(n,generator=g).to(dev)
        for i in range(0,n,512):
            j=perm[i:i+512]; opt.zero_grad()
            lossf(net(Xtr_t[j]),ytr_t[j]).backward(); opt.step()
    net.eval()
    # danh gia tren video test
    aucs=[]; precs={10:[],15:[],25:[]}
    yall=[]; pall=[]
    for k in te:
        x=(torch.from_numpy(feats[k]).to(dev)-mu)/sg
        with torch.no_grad(): p=torch.sigmoid(net(x)).cpu().numpy()
        g_=labs[k]; yall.append(g_); pall.append(p)
        for kb in precs:
            lim=max(1,int(len(g_)*kb/100)); sel=np.argsort(p)[-lim:]
            precs[kb].append(g_[sel].mean())
    y=np.concatenate(yall); pp=np.concatenate(pall)
    # AUC = P(score_pos > score_neg) tinh bang rank (Mann-Whitney)
    if len(set(y.tolist()))>1:
        order=np.argsort(pp); ranks=np.empty(len(pp)); ranks[order]=np.arange(1,len(pp)+1)
        npos=int(y.sum()); nneg=len(y)-npos
        auc=(ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    else: auc=float('nan')
    return auc,{kb:np.mean(v) for kb,v in precs.items()},np.mean(y)

aucs=[];P={10:[],15:[],25:[]};base=[]
for s in range(SEEDS):
    a,pr,b=run(s); aucs.append(a); base.append(b)
    for kb in P: P[kb].append(pr[kb])
print('=== BO PHAN LOAI CO GIAM SAT (chia theo video, %d seed) ==='%SEEDS)
print('Frame-level AUC: %.3f ± %.3f'%(np.nanmean(aucs),np.nanstd(aucs)))
print('base rate ton thuong tren test: %.1f%%'%(np.mean(base)*100))
print('Precision (%% frame chon LA ton thuong):')
for kb in [10,15,25]:
    print('  @%d%%: %.1f%% ± %.1f'%(kb,np.mean(P[kb])*100,np.std(P[kb])*100))
