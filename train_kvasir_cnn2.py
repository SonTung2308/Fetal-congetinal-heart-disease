"""Fine-tune ResNet18 phan loai ton thuong, CHIA THEO VIDEO, early-stop tren val,
danh gia nhieu seed. Chi so chinh: AUC, Top-1, Precision@R (tim dung frame benh)."""
import sys, csv, glob, os, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT='data/kvasir/labelled_images'; META='data/kvasir/metadata.csv'; dev='cuda'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
ARCH=sys.argv[1] if len(sys.argv)>1 else 'resnet18'
NSEED=int(sys.argv[2]) if len(sys.argv)>2 else 3
MAXEP=15

paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
rows=[r for r in csv.DictReader(open(META),delimiter=';') if r['filename'] in paths]
by_vid=defaultdict(list)
for r in rows:
    by_vid[r['video_id']].append((paths[r['filename']],1 if r['finding_class'] in PATH else 0))
pos_vids=[v for v in by_vid if any(l for _,l in by_vid[v])]

TF_TR=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_TE=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(s,it,tf): s.it=it; s.tf=tf
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        p,l=s.it[i]; return s.tf(Image.open(p).convert('RGB')),float(l)

def auc_of(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    npos=int(y.sum()); nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg) if npos and nneg else float('nan')

def scores_for(net,vids_items):
    net.eval(); out={}
    for v,it in vids_items.items():
        dl=DataLoader(DS(it,TF_TE),batch_size=256,num_workers=6)
        ps=[]
        with torch.no_grad():
            for x,_ in dl: ps.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
        out[v]=(np.concatenate(ps),np.array([l for _,l in it],dtype=float))
    return out

def run(seed):
    rng=np.random.RandomState(seed); vids=list(pos_vids); rng.shuffle(vids)
    nte=max(5,int(0.25*len(vids))); nva=max(3,int(0.15*len(vids)))
    te=set(vids[:nte]); va=set(vids[nte:nte+nva]); tr=set(vids[nte+nva:])
    tr_items=[it for v in tr for it in by_vid[v]]
    va_items={v:by_vid[v] for v in va}; te_items={v:by_vid[v] for v in te}
    pos=sum(l for _,l in tr_items); w=torch.tensor([(len(tr_items)-pos)/pos],device=dev)
    net=getattr(models,ARCH)(weights='DEFAULT'); net.fc=nn.Linear(net.fc.in_features,1); net=net.to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    dl=DataLoader(DS(tr_items,TF_TR),batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    best_auc=-1; best_state=None; bad=0
    for ep in range(MAXEP):
        net.train()
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); lossf(net(x).squeeze(1),y).backward(); opt.step()
        sc=scores_for(net,va_items)
        yv=np.concatenate([g for _,g in sc.values()]); pv=np.concatenate([p for p,_ in sc.values()])
        a=auc_of(yv,pv)
        if a>best_auc: best_auc=a; best_state={k:v.cpu().clone() for k,v in net.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=3: break
    net.load_state_dict(best_state)
    sc=scores_for(net,te_items)
    y=np.concatenate([g for _,g in sc.values()]); p=np.concatenate([pp for pp,_ in sc.values()])
    top1=[]; atR=[]; pk={5:[],10:[],15:[]}
    for pp,g in sc.values():
        o=np.argsort(pp); top1.append(g[o[-1]])
        R=int(g.sum()); atR.append(g[o[-R:]].mean() if R>0 else np.nan)
        for kb in pk:
            lim=max(1,int(len(g)*kb/100)); pk[kb].append(g[o[-lim:]].mean())
    return dict(auc=auc_of(y,p),top1=np.mean(top1),atR=np.nanmean(atR),
                base=y.mean(),**{f'p{kb}':np.mean(v) for kb,v in pk.items()})

R=[run(s) for s in range(NSEED)]
def ms(k): a=np.array([r[k] for r in R]); return a.mean(),a.std()
print(f'=== {ARCH} fine-tune, early-stop, {NSEED} seed (chia theo video) ===')
print('base rate test        : %.1f%%'%(ms("base")[0]*100))
print('Frame AUC             : %.3f ± %.3f'%ms('auc'))
print('Top-1 (frame cao nhat la benh): %.0f%% ± %.0f'%(ms('top1')[0]*100,ms('top1')[1]*100))
print('Precision@R (dung so frame benh): %.1f%% ± %.1f'%(ms('atR')[0]*100,ms('atR')[1]*100))
print('Precision@5%%  : %.1f%%'%(ms('p5')[0]*100))
print('Precision@10%% : %.1f%%'%(ms('p10')[0]*100))
print('Precision@15%% : %.1f%%'%(ms('p15')[0]*100))
