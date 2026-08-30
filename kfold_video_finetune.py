"""K-fold DUT KHOAT tren VIDEO TIMENLINE full, fine-tune ResNet18.
Moi fold: fine-tune tren anh labelled_images cua video TRAIN,
roi cham diem tren full-timeline cua video TEST (doc mp4, chi frame trong frame_idx da luu).
So sanh truc tiep voi ban anh (labelled_images)."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, cv2, h5py
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

IMGROOT='data/kvasir/labelled_images'; META='data/kvasir/metadata.csv'
VDIR='data/kvasir/labelled_videos'; VH5='data/kvasir_video.h5'; dev='cuda'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
KFOLD=5

# anh de train
paths={os.path.basename(p):p for p in glob.glob(IMGROOT+'/**/*.jpg',recursive=True)}
rows=[r for r in csv.DictReader(open(META),delimiter=';') if r['filename'] in paths]
img_by_vid=defaultdict(list)
for r in rows:
    img_by_vid[r['video_id']].append((paths[r['filename']],1 if r['finding_class'] in PATH else 0))
# frame_idx can cham cho tung video test (tu h5)
vh5=h5py.File(VH5,'r')
vids=sorted(vh5.keys())

TF_TR=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
NORM_M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
NORM_S=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
class IDS(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        p,l=s.it[i]; return TF_TR(Image.open(p).convert('RGB')),float(l)

def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    npos=int(y.sum()); nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg) if npos and nneg else float('nan')

def score_video(net,vid):
    """doc mp4, cham diem cac frame co trong frame_idx da luu"""
    want=set(int(x) for x in vh5[vid]['frame_idx'][...])
    idx_order=list(vh5[vid]['frame_idx'][...])
    vp=os.path.join(VDIR,vid+'.mp4'); cap=cv2.VideoCapture(vp)
    got={}; idx=0
    buf=[]; bidx=[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        if idx in want:
            buf.append(cv2.resize(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),(224,224))); bidx.append(idx)
            if len(buf)>=512:
                x=(torch.from_numpy(np.stack(buf)).float().permute(0,3,1,2).to(dev)/255.0-NORM_M)/NORM_S
                with torch.no_grad(): pr=torch.sigmoid(net(x).squeeze(1)).cpu().numpy()
                for j,ii in enumerate(bidx): got[ii]=pr[j]
                buf=[]; bidx=[]
        idx+=1
    cap.release()
    if buf:
        x=(torch.from_numpy(np.stack(buf)).float().permute(0,3,1,2).to(dev)/255.0-NORM_M)/NORM_S
        with torch.no_grad(): pr=torch.sigmoid(net(x).squeeze(1)).cpu().numpy()
        for j,ii in enumerate(bidx): got[ii]=pr[j]
    p=np.array([got[i] for i in idx_order]); g=vh5[vid]['gtscore'][...]
    return p,g

rng=np.random.RandomState(0); order=np.array(vids); rng.shuffle(order)
folds=np.array_split(order,KFOLD)
per={}
for fi,te in enumerate(folds):
    te=set(te.tolist()); tr_items=[it for v in img_by_vid if v not in te for it in img_by_vid[v]]
    pos=sum(l for _,l in tr_items); w=torch.tensor([(len(tr_items)-pos)/pos],device=dev)
    net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.fc=nn.Linear(512,1); net=net.to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    dl=DataLoader(IDS(tr_items),batch_size=128,shuffle=True,num_workers=8,pin_memory=True)
    for ep in range(6):
        net.train()
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); lossf(net(x).squeeze(1),y).backward(); opt.step()
    net.eval()
    for v in te: per[v]=score_video(net,v)
    print('fold %d/%d xong'%(fi+1,KFOLD),flush=True)

Y=np.concatenate([g for _,g in per.values()]); P=np.concatenate([p for p,_ in per.values()])
print('=== VIDEO full, fine-tune ResNet18, 5-fold | %d frame, base %.2f%% ==='%(len(Y),Y.mean()*100))
print('AUC toan cuc: %.3f'%auc(Y,P))
pv=[k for k in per if per[k][1].max()>0]
top1=[]; atR=[]; pk={1:[],5:[],10:[]}
for k in pv:
    p,g=per[k]; o=np.argsort(p); top1.append(g[o[-1]]); R=int(g.sum()); atR.append(g[o[-R:]].mean())
    for kb in pk:
        lim=max(1,int(len(g)*kb/100)); pk[kb].append(g[o[-lim:]].mean())
print('So video co ton thuong:',len(pv))
print('Top-1: %.0f%% | P@R: %.1f%% | P@1%%: %.1f%% | P@5%%: %.1f%% | P@10%%: %.1f%%'%(
    np.mean(top1)*100,np.mean(atR)*100,np.mean(pk[1])*100,np.mean(pk[5])*100,np.mean(pk[10])*100))
