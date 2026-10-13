"""Dựng ảnh: SET of frame được chọn + box detect(đỏ) vs ground-truth(xanh), so sánh QUA CÁC VÒNG RL.
Mỗi hàng = 1 vòng RL (0,3,10,60). Mỗi cột = 1 frame trong set chọn (top-6). Nhãn: di=IoU, ✓bệnh/✗thường."""
import csv, glob, os, sys, numpy as np, torch, torch.nn as nn, h5py
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
dev='cuda'; W=336; VID=sys.argv[1] if len(sys.argv)>1 else 'bca26705313a4644'; TOPK=12; ROUNDS=[0,5,20,60]
OUT='fig_frames_%s.png'%VID[:8]
META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
# frame cua video muc tieu
items=[]
for r in csv.DictReader(open(META),delimiter=';'):
    if r['video_id']!=VID or r['filename'] not in paths: continue
    isp=r['finding_class'] in PATH; gt=None
    if isp and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')];ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: gt=b
    items.append((paths[r['filename']],int(r['frame_number']),1 if isp else 0,gt))
items.sort(key=lambda x:x[1]); print('video %s: %d frame'%(VID,len(items)))
det=fasterrcnn_resnet50_fpn_v2(weights=None)
det.roi_heads.box_predictor=FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features,2)
det.load_state_dict(torch.load('data/kvasir_detector_v3.pth',map_location='cpu')); det.eval().to(dev)
enc=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); enc.fc=nn.Identity(); enc.eval().to(dev)
M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); S=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
# chay detector + feature cho toan bo frame video
feats=[];dcs=[];dis=[];topbox=[];imgs=[]; B=32
for i in range(0,len(items),B):
    ch=items[i:i+B]; ims=[Image.open(p).convert('RGB').resize((W,W)) for p,_,_,_ in ch]
    x=torch.stack([TF.to_tensor(im) for im in ims]).to(dev)
    with torch.no_grad():
        out=det(list(x)); fe=enc((x-M)/S).cpu().numpy()
    for j,(p,fr,isp,gt) in enumerate(ch):
        sc=out[j]['scores']; conf=float(sc[0]) if len(sc)>0 else 0.0
        bx=out[j]['boxes'][0].cpu().tolist() if len(out[j]['boxes'])>0 else None
        di=iou(bx,gt) if (gt is not None and bx is not None) else 0.0
        feats.append(fe[j]);dcs.append(conf);dis.append(di);topbox.append(bx);imgs.append(ims[j])
feats=np.array(feats,np.float32);dcs=np.array(dcs,np.float32);dis=np.array(dis,np.float32)
isp=np.array([it[2] for it in items]); gts=[it[3] for it in items]
X=torch.tensor(np.concatenate([feats,dcs[:,None]],1)).to(dev)
# train critic tren TRAIN videos (loop_frames), luu checkpoint tai cac vong
f=h5py.File('data/loop_frames.h5','r'); byv={}
for k in f: byv.setdefault(f[k].attrs['vid'],[]).append(k)
vids=sorted(byv); rng=np.random.RandomState(0); o=list(vids); rng.shuffle(o); te=set(o[:max(4,int(0.25*len(o)))])
tr=[v for v in vids if v not in te]
Xtr=[];Ytr=[]
for v in tr:
    ks=byv[v]; Xtr.append(np.concatenate([np.stack([f[k]['feat'][...] for k in ks]),np.array([[float(f[k]['dc'][()])] for k in ks],np.float32)],1))
    Ytr.append(np.array([float(f[k]['di'][()]) for k in ks],np.float32))
Xtr=torch.tensor(np.concatenate(Xtr).astype(np.float32)).to(dev); Ytr=torch.tensor(np.concatenate(Ytr)).to(dev)
class Q(nn.Module):
    def __init__(s,d=513,h=256):
        super().__init__();s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
torch.manual_seed(0); q=Q().to(dev); opt=torch.optim.Adam(q.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.MSELoss()
n=len(Xtr); g=torch.Generator().manual_seed(0); picks={}
def select():
    q.eval()
    with torch.no_grad(): pr=q(X).cpu().numpy()
    return np.argsort(pr)[-TOPK:][::-1]
picks[0]=select()
for ep in range(1,max(ROUNDS)+1):
    q.train(); perm=torch.randperm(n,generator=g)
    for i in range(0,n,1024):
        j=perm[i:i+1024]; opt.zero_grad(); lossf(q(Xtr[j]),Ytr[j]).backward(); opt.step()
    if ep in ROUNDS: picks[ep]=select()
# ve luoi
def draw(idx):
    im=imgs[idx].copy().convert('RGB'); d=ImageDraw.Draw(im)
    if gts[idx] is not None: d.rectangle(gts[idx],outline=(0,220,0),width=3)          # GT xanh
    if topbox[idx] is not None: d.rectangle(topbox[idx],outline=(255,40,40),width=3)  # detect do
    tag='benh' if isp[idx]==1 else 'thuong'; col=(0,180,0) if dis[idx]>=0.5 else (220,120,0) if isp[idx]==1 else (150,150,150)
    d.rectangle([0,0,W,20],fill=(0,0,0)); d.text((3,3),'%s  IoU=%.2f'%(tag,dis[idx]),fill=(255,255,255))
    return im
TH=190; pad=4; cellw=TH+pad; rowh=TH+pad+22; cols=TOPK
canvas=Image.new('RGB',(cellw*cols+pad, rowh*len(ROUNDS)+pad),(255,255,255)); dd=ImageDraw.Draw(canvas)
for ri,rd in enumerate(ROUNDS):
    y0=ri*rowh+pad+16
    dd.text((pad,ri*rowh+pad+2),'Vong RL = %d   |   set chon: benh=%d/%d, mean IoU=%.2f'%(rd,int(isp[picks[rd]].sum()),TOPK,dis[picks[rd]].mean()),fill=(0,0,0))
    for ci,idx in enumerate(picks[rd]):
        canvas.paste(draw(idx).resize((TH,TH)),(ci*cellw+pad, y0))
canvas.save(OUT); print('[*] saved',OUT)
for rd in ROUNDS: print('vong %2d: benh %d/%d | mean IoU set %.3f'%(rd,int(isp[picks[rd]].sum()),TOPK,dis[picks[rd]].mean()))
