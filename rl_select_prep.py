"""Kiểm 'đất' cho RL SELECTION: detector xuất nhiều proposals, lấy top-K box+score.
So top-1 (theo score) vs oracle max IoU over top-K. Nếu oracle >> top1 -> RL chọn box có đất.
Lưu top-K để train RL selector."""
import csv, glob, os, numpy as np, torch, h5py
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'; dev='cuda'; W=336; K=10
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
items=[]
for r in csv.DictReader(open(META),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')]; ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: items.append((paths[r['filename']],b,r['video_id']))
net=fasterrcnn_resnet50_fpn_v2(weights=None)
net.roi_heads.box_predictor=FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features,2)
net.load_state_dict(torch.load('data/kvasir_detector_v3.pth',map_location='cpu'))
net.roi_heads.score_thresh=0.01; net.roi_heads.nms_thresh=0.6; net.roi_heads.detections_per_img=K
net.eval().to(dev)
def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]);x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
fout=h5py.File('data/rl_select.h5','w'); ctr=0; B=16; buf=[]
def flush(buf):
    global ctr
    imgs=[TF.to_tensor(Image.open(p).convert('RGB').resize((W,W))).to(dev) for p,_,_ in buf]
    with torch.no_grad():
        out=net(imgs); fmap=net.backbone(torch.stack(imgs))['2'].cpu().numpy()
    for i,(p,gt,vid) in enumerate(buf):
        ctr+=1; bx=out[i]['boxes'].cpu().numpy(); sc=out[i]['scores'].cpu().numpy()
        if len(bx)==0: bx=np.array([[W*.25,W*.25,W*.75,W*.75]],np.float32); sc=np.array([0.01],np.float32)
        g=fout.create_group('img%06d'%ctr); g.attrs['vid']=vid
        g.create_dataset('gt',data=np.array(gt,np.float32)); g.create_dataset('boxes',data=bx.astype(np.float32)); g.create_dataset('scores',data=sc.astype(np.float32))
for it in items:
    buf.append(it)
    if len(buf)>=B: flush(buf); buf=[]
flush(buf); fout.close()
# oracle
f=h5py.File('data/rl_select.h5','r'); t1=[]; orac=[]; nprop=[]
for k in f:
    gt=f[k]['gt'][...]; bx=f[k]['boxes'][...]; ious=[iou(b,gt) for b in bx]
    t1.append(ious[0]); orac.append(max(ious)); nprop.append(len(bx))
t1=np.array(t1); orac=np.array(orac)
print('=== DAT cho RL SELECTION ===')
print('so proposal TB: %.1f'%np.mean(nprop))
print('top-1 (theo score): IoU %.3f | DR %.1f%%'%(t1.mean(),(t1>=.5).mean()*100))
print('ORACLE max-IoU trong top-K: IoU %.3f | DR %.1f%%'%(orac.mean(),(orac>=.5).mean()*100))
print('=> tiem nang RL chon: +%.3f IoU (+%.1f%%), DR +%.1f diem'%(orac.mean()-t1.mean(),(orac.mean()-t1.mean())/t1.mean()*100,(orac>=.5).mean()*100-(t1>=.5).mean()*100))
