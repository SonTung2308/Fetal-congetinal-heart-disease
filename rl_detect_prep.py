"""Chuẩn bị cho RL tinh chỉnh box: với mỗi ảnh bệnh, lưu
  - gt_box (thật), init_box (detector dự đoán top-1)
  - feature map 512x7x7 (ResNet18-ImageNet layer4) để ROI-pool nhanh khi RL rollout
Chia theo video. Xuất data/rl_detect.h5."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, h5py
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models

META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'; dev='cuda'; W=336
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
items=[]  # (filepath, gt_box, video_id)
for r in csv.DictReader(open(META),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')]; ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: items.append((paths[r['filename']],b,r['video_id']))
print('ảnh:',len(items))

# detector
det=fasterrcnn_mobilenet_v3_large_fpn(weights=None)
det.roi_heads.box_predictor=FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features,2)
det.load_state_dict(torch.load('data/kvasir_detector.pth',map_location='cpu')); det.eval().to(dev)
# feature extractor (ResNet18 layer4 -> 512x7x7)
r18=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
fex=nn.Sequential(*list(r18.children())[:-2]).eval().to(dev)  # (B,512,7,7) cho input 224
M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); Sd=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)

fout=h5py.File('data/rl_detect.h5','w')
B=16; buf=[]; CTR=[0]
def flush(buf):
    if not buf: return
    fps=[b[0] for b in buf]
    imgs=torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((W,W))) for p in fps]).to(dev)
    with torch.no_grad():
        out=det([imgs[i] for i in range(len(imgs))])
        x=torch.nn.functional.interpolate(imgs,size=(224,224),mode='bilinear',align_corners=False)
        fm=fex((x-M)/Sd).cpu().numpy()   # (B,512,7,7)
    for i,(p,gt,vid) in enumerate(buf):
        CTR[0]+=1
        if len(out[i]['boxes'])>0: ib=out[i]['boxes'][0].cpu().numpy()
        else: ib=np.array([W*0.25,W*0.25,W*0.75,W*0.75],np.float32)
        g=fout.create_group('img%06d'%CTR[0])
        g.attrs['vid']=vid
        g.create_dataset('gt',data=np.array(gt,np.float32))
        g.create_dataset('init',data=ib.astype(np.float32))
        g.create_dataset('fm',data=fm[i].astype(np.float32),compression='gzip')
for it in items:
    buf.append(it)
    if len(buf)>=B: flush(buf); buf=[]
flush(buf)
fout.close(); print('[*] xong -> data/rl_detect.h5 |',len(items),'ảnh')
