"""Fine-tune ResNet18 dau-cuoi phan loai ton thuong vs binh thuong tren anh Kvasir.
Chia theo VIDEO (tranh ro ri). Do frame-level AUC + precision@k tren video test."""
import sys, csv, glob, os, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT='data/kvasir/labelled_images'; META='data/kvasir/metadata.csv'
dev='cuda'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
EPOCHS=int(sys.argv[2]) if len(sys.argv)>2 else 6

paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
rows=[r for r in csv.DictReader(open(META),delimiter=';') if r['filename'] in paths]
by_vid=defaultdict(list)
for r in rows:
    by_vid[r['video_id']].append((paths[r['filename']], 1 if r['finding_class'] in PATH else 0))
vids=[v for v in by_vid if any(l for _,l in by_vid[v])]
rng=np.random.RandomState(SEED); rng.shuffle(vids)
nte=max(4,int(0.3*len(vids))); te=set(vids[:nte])
tr_items=[it for v in by_vid for it in by_vid[v] if v not in te]
te_by_vid={v:by_vid[v] for v in te}
print(f'train {len(tr_items)} anh | test {sum(len(x) for x in te_by_vid.values())} anh ({len(te)} video)')

TF_TR=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_TE=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class DS(Dataset):
    def __init__(s,items,tf): s.items=items; s.tf=tf
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        p,l=s.items[i]; return s.tf(Image.open(p).convert('RGB')), float(l)

net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
net.fc=nn.Linear(512,1); net=net.to(dev)
pos=sum(l for _,l in tr_items); w=torch.tensor([(len(tr_items)-pos)/pos],device=dev)
opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
lossf=nn.BCEWithLogitsLoss(pos_weight=w)
dl=DataLoader(DS(tr_items,TF_TR),batch_size=128,shuffle=True,num_workers=8,pin_memory=True)

for ep in range(EPOCHS):
    net.train(); tot=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad()
        loss=lossf(net(x).squeeze(1),y); loss.backward(); opt.step(); tot+=loss.item()
    print(f'epoch {ep+1}/{EPOCHS} loss {tot/len(dl):.4f}',flush=True)

# danh gia
net.eval(); precs={10:[],15:[],25:[]}; topk={1:[],3:[],5:[]}; atR=[]; yall=[];pall=[]; top1=[]
for v,items in te_by_vid.items():
    dlt=DataLoader(DS(items,TF_TE),batch_size=256,num_workers=8)
    ps=[]
    with torch.no_grad():
        for x,_ in dlt: ps.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
    p=np.concatenate(ps); g=np.array([l for _,l in items],dtype=float)
    yall.append(g); pall.append(p)
    top1.append(g[np.argmax(p)])
    ordr=np.argsort(p)
    for kk in topk: topk[kk].append(g[ordr[-kk:]].mean())
    R=int(g.sum()); atR.append(g[ordr[-R:]].mean() if R>0 else np.nan)
    for kb in precs:
        lim=max(1,int(len(g)*kb/100)); precs[kb].append(g[ordr[-lim:]].mean())
y=np.concatenate(yall); pp=np.concatenate(pall)
order=np.argsort(pp); ranks=np.empty(len(pp)); ranks[order]=np.arange(1,len(pp)+1)
npos=int(y.sum()); auc=(ranks[y==1].sum()-npos*(npos+1)/2)/(npos*(len(y)-npos))
print('=== FINE-TUNE ResNet18 (seed %d) ==='%SEED)
print('base rate test: %.1f%%'%(y.mean()*100))
print('Frame AUC: %.3f'%auc)
print('Top-1 (frame diem cao nhat la ton thuong): %.0f%% video'%(np.mean(top1)*100))
for kk in [1,3,5]: print('Precision@top-%d frame: %.1f%%'%(kk,np.nanmean(topk[kk])*100))
print('Precision@R (chon dung so frame benh): %.1f%%'%(np.nanmean(atR)*100))
for kb in [10,15,25]: print('Precision@%d%%: %.1f%%'%(kb,np.mean(precs[kb])*100))
