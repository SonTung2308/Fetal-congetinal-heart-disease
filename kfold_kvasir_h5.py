"""K-fold CV tren feature .h5 co san (frozen ImageNet). Dung cho ca kvasir_capsule.h5
(anh) lan kvasir_video.h5 (video timeline). Head phan loai nho tren 512-d."""
import sys, numpy as np, h5py, torch, torch.nn as nn
H5=sys.argv[1] if len(sys.argv)>1 else 'data/kvasir_video.h5'
KFOLD=5; dev='cuda'
f=h5py.File(H5,'r'); ks=list(f.keys())
feats={k:f[k]['features'][...].astype(np.float32) for k in ks}
labs ={k:f[k]['gtscore'][...].astype(np.float32) for k in ks}
def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    npos=int(y.sum()); nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg) if npos and nneg else float('nan')
rng=np.random.RandomState(0); order=np.array(ks); rng.shuffle(order)
folds=np.array_split(order,KFOLD)
class Clf(nn.Module):
    def __init__(s,d=512,h=256,p=0.4):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(p),
            nn.Linear(h,h),nn.ReLU(),nn.Dropout(p),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
per={}
for te in folds:
    te=set(te.tolist()); tr=[k for k in ks if k not in te]
    Xtr=np.concatenate([feats[k] for k in tr]); ytr=np.concatenate([labs[k] for k in tr])
    mu,sg=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sg
    pos=ytr.sum(); w=torch.tensor([(len(ytr)-pos)/max(pos,1)],device=dev)
    Xt=torch.from_numpy(Xtr).to(dev); yt=torch.from_numpy(ytr).to(dev)
    torch.manual_seed(0); net=Clf().to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w); n=len(yt); g=torch.Generator(); g.manual_seed(0)
    for ep in range(40):
        net.train(); perm=torch.randperm(n,generator=g).to(dev)
        for i in range(0,n,1024):
            j=perm[i:i+1024]; opt.zero_grad(); lossf(net(Xt[j]),yt[j]).backward(); opt.step()
    net.eval()
    for k in te:
        x=(torch.from_numpy(feats[k]).to(dev)-torch.from_numpy(mu).to(dev))/torch.from_numpy(sg).to(dev)
        with torch.no_grad(): per[k]=(torch.sigmoid(net(x)).cpu().numpy(),labs[k])
    print('fold xong',flush=True)
Y=np.concatenate([g for _,g in per.values()]); P=np.concatenate([p for p,_ in per.values()])
print('=== %s | %d frame, base %.2f%% ==='%(H5.split('/')[-1],len(Y),Y.mean()*100))
print('AUC toan cuc: %.3f'%auc(Y,P))
pv=[k for k in per if per[k][1].max()>0]
top1=[]; atR=[]; pk={1:[],5:[],10:[]}
for k in pv:
    p,g=per[k]; o=np.argsort(p); top1.append(g[o[-1]]); R=int(g.sum()); atR.append(g[o[-R:]].mean())
    for kb in pk:
        lim=max(1,int(len(g)*kb/100)); pk[kb].append(g[o[-lim:]].mean())
print('Top-1: %.0f%% | P@R: %.1f%% | P@1%%: %.1f%% | P@5%%: %.1f%% | P@10%%: %.1f%%'%(
    np.mean(top1)*100,np.mean(atR)*100,np.mean(pk[1])*100,np.mean(pk[5])*100,np.mean(pk[10])*100))
