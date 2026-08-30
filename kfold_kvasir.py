"""K-fold CV theo VIDEO tren 43 video Kvasir. Moi video lan luot vao test 1 lan.
Gop du doan tren TOAN BO 47k frame -> AUC on dinh (khong nhieu nhu 1 seed test 6 video).
Chi so: AUC toan cuc, va precision/top-1 trung binh tren cac video co ton thuong."""
import sys, csv, glob, os, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT='data/kvasir/labelled_images'; META='data/kvasir/metadata.csv'; dev='cuda'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
ARCH=sys.argv[1] if len(sys.argv)>1 else 'resnet18'
KFOLD=int(sys.argv[2]) if len(sys.argv)>2 else 5
MAXEP=6   # 6 epoch la diem tot nhat da tim thay

paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
rows=[r for r in csv.DictReader(open(META),delimiter=';') if r['filename'] in paths]
by_vid=defaultdict(list)
for r in rows:
    by_vid[r['video_id']].append((paths[r['filename']],1 if r['finding_class'] in PATH else 0))
vids=sorted(by_vid); 
rng=np.random.RandomState(0); order=np.array(vids); rng.shuffle(order)
folds=np.array_split(order,KFOLD)

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

per_vid={}   # video -> (scores, labels)
for fi,te in enumerate(folds):
    te=set(te.tolist()); tr_items=[it for v in by_vid if v not in te for it in by_vid[v]]
    pos=sum(l for _,l in tr_items); w=torch.tensor([(len(tr_items)-pos)/pos],device=dev)
    net=getattr(models,ARCH)(weights='DEFAULT'); net.fc=nn.Linear(net.fc.in_features,1); net=net.to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    dl=DataLoader(DS(tr_items,TF_TR),batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    for ep in range(MAXEP):
        net.train()
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); lossf(net(x).squeeze(1),y).backward(); opt.step()
    net.eval()
    for v in te:
        dlt=DataLoader(DS(by_vid[v],TF_TE),batch_size=256,num_workers=6); ps=[]
        with torch.no_grad():
            for x,_ in dlt: ps.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
        per_vid[v]=(np.concatenate(ps),np.array([l for _,l in by_vid[v]],dtype=float))
    print(f'fold {fi+1}/{KFOLD} xong ({len(te)} video test)',flush=True)

# AUC toan cuc tren moi frame
Y=np.concatenate([g for _,g in per_vid.values()]); P=np.concatenate([p for p,_ in per_vid.values()])
print('=== K-FOLD %d, %s, gop toan bo %d frame ==='%(KFOLD,ARCH,len(Y)))
print('base rate ton thuong: %.1f%%'%(Y.mean()*100))
print('AUC toan cuc (moi frame): %.3f'%auc_of(Y,P))
# theo video co ton thuong
pv=[v for v in per_vid if per_vid[v][1].max()>0]
top1=[]; atR=[]; pk={5:[],10:[],15:[],25:[]}
for v in pv:
    p,g=per_vid[v]; o=np.argsort(p); top1.append(g[o[-1]])
    R=int(g.sum()); atR.append(g[o[-R:]].mean())
    for kb in pk:
        lim=max(1,int(len(g)*kb/100)); pk[kb].append(g[o[-lim:]].mean())
print('So video co ton thuong: %d'%len(pv))
print('Top-1 (frame cao nhat la benh): %.0f%% video'%(np.mean(top1)*100))
print('Precision@R (dung so frame benh): %.1f%%'%(np.mean(atR)*100))
for kb in [5,10,15,25]: print('Precision@%d%%: %.1f%%'%(kb,np.mean(pk[kb])*100))
