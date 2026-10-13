"""Closed-loop RL — Bước 1: precompute per-frame.
Với MỖI frame có nhãn (bệnh + bình thường): chạy detector v3 -> det_conf, det_iou (vs GT nếu bệnh,
=0 nếu bình thường/không GT); trích đặc trưng ResNet18-ImageNet 512-d cho selector.
Lưu: video_id, frame_number, is_path, det_conf, det_iou, feat(512)."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, h5py
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'; dev='cuda'; W=336
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
items=[]  # (path, video, frame, is_path, gt or None)
for r in csv.DictReader(open(META),delimiter=';'):
    fn=r['filename']
    if fn not in paths: continue
    ispath=r['finding_class'] in PATH
    gt=None
    if ispath and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')];ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: gt=b
    items.append((paths[fn],r['video_id'],int(r['frame_number']),1 if ispath else 0,gt))
print('frame có nhãn:',len(items),'| bệnh có GT:',sum(1 for it in items if it[4]))
# detector v3
det=fasterrcnn_resnet50_fpn_v2(weights=None)
det.roi_heads.box_predictor=FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features,2)
det.load_state_dict(torch.load('data/kvasir_detector_v3.pth',map_location='cpu')); det.eval().to(dev)
# resnet18 imagenet feature
enc=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); enc.fc=nn.Identity(); enc.eval().to(dev)
M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); S=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
fout=h5py.File('data/loop_frames.h5','w'); ctr=0; B=32; buf=[]
def flush(buf):
    global ctr
    ims=[Image.open(p).convert('RGB').resize((W,W)) for p,_,_,_,_ in buf]
    x=torch.stack([TF.to_tensor(im) for im in ims]).to(dev)
    with torch.no_grad():
        out=det(list(x)); feat=enc((x-M)/S).cpu().numpy()
    for i,(p,vid,fr,isp,gt) in enumerate(buf):
        ctr+=1; sc=out[i]['scores']; conf=float(sc[0]) if len(sc)>0 else 0.0
        di=0.0
        if gt is not None and len(out[i]['boxes'])>0: di=iou(out[i]['boxes'][0].cpu().tolist(),gt)
        g=fout.create_group('f%06d'%ctr); g.attrs['vid']=vid
        g.create_dataset('meta',data=np.array([fr,isp],np.int32))
        g.create_dataset('dc',data=np.float32(conf)); g.create_dataset('di',data=np.float32(di))
        g.create_dataset('feat',data=feat[i].astype(np.float32))
for it in items:
    buf.append(it)
    if len(buf)>=B: flush(buf); buf=[]
flush(buf); fout.close(); print('[*] xong ->',ctr,'frame -> data/loop_frames.h5')
