"""Nhu kfold_video_finetune nhung LUU feature fine-tune 512-d cua tung frame test
vao data/kvasir_video_ft.h5 -> cho phep thu nghiem lam muot / BiGRU offline nhanh."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, cv2, h5py
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

IMGROOT='data/kvasir/labelled_images'; META='data/kvasir/metadata.csv'
VDIR='data/kvasir/labelled_videos'; VH5='data/kvasir_video.h5'
OUT='data/kvasir_video_ft.h5'; dev='cuda'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
KFOLD=5
paths={os.path.basename(p):p for p in glob.glob(IMGROOT+'/**/*.jpg',recursive=True)}
rows=[r for r in csv.DictReader(open(META),delimiter=';') if r['filename'] in paths]
img_by_vid=defaultdict(list)
for r in rows:
    img_by_vid[r['video_id']].append((paths[r['filename']],1 if r['finding_class'] in PATH else 0))
vh5=h5py.File(VH5,'r'); vids=sorted(vh5.keys())
TF_TR=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
S=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
class IDS(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        p,l=s.it[i]; return TF_TR(Image.open(p).convert('RGB')),float(l)

def extract_video(backbone,head,vid):
    want=list(vh5[vid]['frame_idx'][...]); wantset=set(int(x) for x in want)
    cap=cv2.VideoCapture(os.path.join(VDIR,vid+'.mp4')); idx=0; buf=[]; bi=[]; featd={}; scored={}
    def flush():
        if not buf: return
        x=(torch.from_numpy(np.stack(buf)).float().permute(0,3,1,2).to(dev)/255.0-M)/S
        with torch.no_grad():
            fe=backbone(x).flatten(1); sc=torch.sigmoid(head(fe).squeeze(1))
        fe=fe.cpu().numpy(); sc=sc.cpu().numpy()
        for j,ii in enumerate(bi): featd[ii]=fe[j]; scored[ii]=sc[j]
        buf.clear(); bi.clear()
    while True:
        ok,fr=cap.read()
        if not ok: break
        if idx in wantset:
            buf.append(cv2.resize(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),(224,224))); bi.append(idx)
            if len(buf)>=512: flush()
        idx+=1
    cap.release(); flush()
    feat=np.stack([featd[i] for i in want]); sc=np.array([scored[i] for i in want])
    return feat.astype(np.float32), sc.astype(np.float32)

rng=np.random.RandomState(0); order=np.array(vids); rng.shuffle(order)
folds=np.array_split(order,KFOLD)
fout=h5py.File(OUT,'w')
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
    backbone=nn.Sequential(*list(net.children())[:-1]); head=net.fc
    for v in te:
        feat,sc=extract_video(backbone,head,v)
        g=fout.create_group(v)
        g.create_dataset('features',data=feat,compression='gzip')
        g.create_dataset('score',data=sc,compression='gzip')
        g.create_dataset('gtscore',data=vh5[v]['gtscore'][...],compression='gzip')
        g.create_dataset('frame_idx',data=vh5[v]['frame_idx'][...],compression='gzip')
    print('fold %d/%d xong'%(fi+1,KFOLD),flush=True)
fout.close(); print('[*] xong ->',OUT)
