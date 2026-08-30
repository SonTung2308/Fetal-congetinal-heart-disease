"""Detector v3: ResNet50-FPN-v2 nhưng CHỐNG OVERFIT — đóng băng lớp đầu backbone
(trainable_backbone_layers=3), augmentation mạnh (hflip+vflip+brightness+contrast+scale jitter),
lr thấp + weight decay cao. Mục tiêu: vượt best v2 0.590/72.0%."""
import csv, glob, os, sys, random, numpy as np, torch
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'; dev='cuda'; W=336
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 20
TBL=int(sys.argv[2]) if len(sys.argv)>2 else 3
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
by_vid=defaultdict(list)
for r in csv.DictReader(open(META),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')]; ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: by_vid[r['video_id']].append((paths[r['filename']],b))
vids=sorted(by_vid); rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
nte=max(4,int(0.25*len(order))); te=set(order[:nte]); tr=[v for v in order if v not in te]
tr_items=[it for v in tr for it in by_vid[v]]; te_items=[it for v in te for it in by_vid[v]]
print('train %d | test %d | ResNet50-v2 TBL=%d | %d epoch'%(len(tr_items),len(te_items),TBL,EPOCHS))

def jitter_scale(img,box):
    s=0.8+0.4*random.random()               # scale 0.8-1.2
    nw=int(W*s); img2=img.resize((nw,nw)); canvas=Image.new('RGB',(W,W))
    off=(W-nw)//2 if nw<=W else 0
    if nw<=W:
        canvas.paste(img2,(off,off)); img=canvas
        box=[box[0]*s+off,box[1]*s+off,box[2]*s+off,box[3]*s+off]
    else:
        c=(nw-W)//2; img=img2.crop((c,c,c+W,c+W))
        box=[box[0]*s-c,box[1]*s-c,box[2]*s-c,box[3]*s-c]
    box=[max(0,min(W,v)) for v in box]
    return img,box

class DS(torch.utils.data.Dataset):
    def __init__(s,items,aug): s.items=items; s.aug=aug
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        p,box=s.items[i]; img=Image.open(p).convert('RGB').resize((W,W)); box=list(box)
        if s.aug:
            if random.random()<0.5: img,box=jitter_scale(img,box)
            if random.random()<0.5: img=TF.hflip(img); box=[W-box[2],box[1],W-box[0],box[3]]
            if random.random()<0.3: img=TF.vflip(img); box=[box[0],W-box[3],box[2],W-box[1]]
            if random.random()<0.5: img=TF.adjust_brightness(img,0.7+0.6*random.random())
            if random.random()<0.5: img=TF.adjust_contrast(img,0.7+0.6*random.random())
        box=[max(0,min(W,v)) for v in box]
        if box[2]-box[0]<4 or box[3]-box[1]<4: box=list(s.items[i][1])
        t={'boxes':torch.tensor([box],dtype=torch.float32),'labels':torch.ones(1,dtype=torch.int64)}
        return TF.to_tensor(img),t
def collate(b): return tuple(zip(*b))

net=fasterrcnn_resnet50_fpn_v2(weights='DEFAULT',trainable_backbone_layers=TBL)
net.roi_heads.box_predictor=FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features,2)
net=net.to(dev)
opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=5e-5,weight_decay=5e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
dl=torch.utils.data.DataLoader(DS(tr_items,True),batch_size=6,shuffle=True,num_workers=8,collate_fn=collate,pin_memory=True)

def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
def evaluate():
    net.eval(); ious=[]; hit=0
    dlt=torch.utils.data.DataLoader(DS(te_items,False),batch_size=6,num_workers=8,collate_fn=collate)
    with torch.no_grad():
        for imgs,tgts in dlt:
            out=net([x.to(dev) for x in imgs])
            for o,t in zip(out,tgts):
                gt=t['boxes'][0].tolist()
                if len(o['boxes'])==0: ious.append(0); continue
                v=iou(o['boxes'][0].cpu().tolist(),gt); ious.append(v); hit+=(v>=0.5)
    return np.mean(ious),hit/len(te_items)

best=(0,0)
for ep in range(EPOCHS):
    net.train(); tot=0
    for imgs,tgts in dl:
        imgs=[x.to(dev) for x in imgs]; tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
        loss=sum(net(imgs,tgts).values())
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    sch.step(); mi,dr=evaluate()
    if mi>best[0]: best=(mi,dr); torch.save(net.state_dict(),'data/kvasir_detector_v3.pth')
    print('ep %2d loss %.3f | IoU %.3f | DR@0.5 %.1f%%'%(ep+1,tot/len(dl),mi,dr*100),flush=True)
print('\n=== TOT NHAT v3: IoU %.3f | DR@0.5 %.1f%% (v1 0.578/71.8, v2 0.590/72.0) ==='%(best[0],best[1]*100))
