"""Cai thien selector tren feature fine-tune da luu (kvasir_video_ft.h5).
So sanh: (a) phan loai tung frame (base), (b) lam muot theo thoi gian,
(c) BiGRU tren chuoi. Chi so: AUC, Top-1, va LESION-RECALL@k (bac si co chan doan duoc khong).
BiGRU train k-fold theo video (khong ro ri)."""
import sys, numpy as np, h5py, torch, torch.nn as nn
dev='cuda'; FT='data/kvasir_video_ft.h5'
f=h5py.File(FT,'r'); vids=list(f.keys())
feat={v:f[v]['features'][...].astype(np.float32) for v in vids}
scr ={v:f[v]['score'][...].astype(np.float32) for v in vids}
lab ={v:f[v]['gtscore'][...].astype(np.float32) for v in vids}
pv=[v for v in vids if lab[v].max()>0]

def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    a=int(y.sum()); b=len(y)-a
    return (r[y==1].sum()-a*(a+1)/2)/(a*b) if a and b else float('nan')

def smooth(p,w):
    if w<=1: return p
    k=np.ones(w)/w; return np.convolve(p,k,mode='same')

def report(name,score_of):
    Y=np.concatenate([lab[v] for v in vids]); P=np.concatenate([score_of(v) for v in vids])
    top1=[]; lesrec={5:[],10:[],20:[]}; frec={5:[],10:[],20:[]}
    for v in pv:
        p=score_of(v); g=lab[v]; o=np.argsort(p); npos=int(g.sum())
        top1.append(g[o[-1]])
        for kb in lesrec:
            lim=max(1,int(len(g)*kb/100)); sel=g[o[-lim:]]
            lesrec[kb].append(1.0 if sel.sum()>0 else 0.0)      # co >=1 frame benh (chan doan duoc)
            frec[kb].append(sel.sum()/npos)                      # bat duoc bao nhieu % frame benh
    print('%-20s AUC %.3f | Top-1 %.0f%% || LesRec(co benh) @5%%=%.0f%% @10%%=%.0f%% @20%%=%.0f%% || FrameRec(bat het benh) @5%%=%.0f%% @10%%=%.0f%% @20%%=%.0f%%'%(
        name,auc(Y,P),np.mean(top1)*100,
        np.mean(lesrec[5])*100,np.mean(lesrec[10])*100,np.mean(lesrec[20])*100,
        np.mean(frec[5])*100,np.mean(frec[10])*100,np.mean(frec[20])*100))

print('So video co ton thuong:',len(pv),'| tong frame:',sum(len(lab[v]) for v in vids))
report('frame classifier', lambda v: scr[v])
for w in [5,9,15,31]:
    report('smooth w=%d'%w, lambda v,w=w: smooth(scr[v],w))

# BiGRU k-fold tren feature fine-tune
class BiGRU(nn.Module):
    def __init__(s,d=512,h=128,p=0.3):
        super().__init__(); s.rnn=nn.GRU(d,h,batch_first=True,bidirectional=True)
        s.head=nn.Sequential(nn.Dropout(p),nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): o,_=s.rnn(x); return s.head(o).squeeze(-1)  # (B,T)

rng=np.random.RandomState(0); order=np.array(vids); rng.shuffle(order)
folds=np.array_split(order,5); gru_scr={}
for te in folds:
    te=set(te.tolist()); tr=[v for v in vids if v not in te]
    allf=np.concatenate([feat[v] for v in tr]); mu,sg=allf.mean(0),allf.std(0)+1e-6
    npos=sum(lab[v].sum() for v in tr); nneg=sum(len(lab[v]) for v in tr)-npos
    w=torch.tensor([nneg/max(npos,1)],device=dev)
    net=BiGRU().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    for ep in range(60):
        net.train()
        for v in tr:
            x=torch.from_numpy((feat[v]-mu)/sg).unsqueeze(0).to(dev)
            y=torch.from_numpy(lab[v]).unsqueeze(0).to(dev)
            opt.zero_grad(); lossf(net(x),y).backward(); opt.step()
    net.eval()
    for v in te:
        x=torch.from_numpy((feat[v]-mu)/sg).unsqueeze(0).to(dev)
        with torch.no_grad(): gru_scr[v]=torch.sigmoid(net(x)).squeeze(0).cpu().numpy()
report('BiGRU (chuoi)', lambda v: gru_scr[v])
# BiGRU + smooth
for w in [9,15]:
    report('BiGRU+smooth w=%d'%w, lambda v,w=w: smooth(gru_scr[v],w))
