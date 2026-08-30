"""Pha B: train ResNet18 tren labelled_images + frame LAN NHAN (curated_pos), k-fold theo video,
cham diem TOAN BO frame video test -> data/kvasir_video_score_prop.h5."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, cv2, h5py
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
IMGROOT='data/kvasir/labelled_images'; META='data/kvasir/metadata.csv'
CUR='data/kvasir_curated_narrow'; VDIR='data/kvasir/labelled_videos'
LAB='data/kvasir_video_fulllabels.h5'; OUT='data/kvasir_video_score_narrow.h5'; dev='cuda'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
KFOLD=5
paths={os.path.basename(p):p for p in glob.glob(IMGROOT+'/**/*.jpg',recursive=True)}
rows=[r for r in csv.DictReader(open(META),delimiter=';') if r['filename'] in paths]
img_by_vid=defaultdict(list)
for r in rows:
    img_by_vid[r['video_id']].append((paths[r['filename']],1 if r['finding_class'] in PATH else 0))
# frame lan nhan (positive bo sung) theo video
cur_by_vid=defaultdict(list)
for v in os.listdir(CUR) if os.path.isdir(CUR) else []:
    for fn in glob.glob(os.path.join(CUR,v,'*.jpg')):
        cur_by_vid[v].append((fn,1))
fl=h5py.File(LAB,'r'); allvids=sorted(fl.keys())
TF_TR=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); S=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
class IDS(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        p,l=s.it[i]; return TF_TR(Image.open(p).convert('RGB')),float(l)
def score_all(net,vid):
    n=len(fl[vid]['gtscore'][...]); cap=cv2.VideoCapture(os.path.join(VDIR,vid+'.mp4'))
    out=np.zeros(n,np.float32); idx=0; buf=[]; bi=[]
    def flush():
        if not buf: return
        x=(torch.from_numpy(np.stack(buf)).float().permute(0,3,1,2).to(dev)/255.0-M)/S
        with torch.no_grad(): pr=torch.sigmoid(net(x).squeeze(1)).cpu().numpy()
        for j,ii in enumerate(bi): out[ii]=pr[j]
        buf.clear(); bi.clear()
    while True:
        ok,fr=cap.read()
        if not ok: break
        buf.append(cv2.resize(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),(224,224))); bi.append(idx)
        if len(buf)>=256: flush()
        idx+=1
    cap.release(); flush(); return out
rng=np.random.RandomState(0); order=np.array(allvids); rng.shuffle(order)
folds=np.array_split(order,KFOLD); fout=h5py.File(OUT,'w')
for fi,te in enumerate(folds):
    te=set(te.tolist())
    tr=[it for v in img_by_vid if v not in te for it in img_by_vid[v]]
    tr+=[it for v in cur_by_vid if v not in te for it in cur_by_vid[v]]   # + lan nhan
    pos=sum(l for _,l in tr); w=torch.tensor([(len(tr)-pos)/pos],device=dev)
    net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.fc=nn.Linear(512,1); net=net.to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss(pos_weight=w)
    dl=DataLoader(IDS(tr),batch_size=64,shuffle=True,num_workers=6,pin_memory=True)
    for ep in range(6):
        net.train()
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); lossf(net(x).squeeze(1),y).backward(); opt.step()
    net.eval()
    for v in te:
        g=fout.create_group(v); g.create_dataset('score',data=score_all(net,v),compression='gzip')
    print('fold %d/%d xong (train %d anh, pos %d)'%(fi+1,KFOLD,len(tr),pos),flush=True)
fout.close(); print('[*] xong ->',OUT)
