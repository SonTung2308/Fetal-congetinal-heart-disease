"""Demo detector v3 (ResNet50-FPN-v2, best IoU 0.610): vẽ box thật (xanh) vs dự đoán (đỏ)."""
import csv, glob, os, numpy as np, torch, cv2
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'; dev='cuda'; W=336; OUT='demo_detect_v3'
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
by_vid=defaultdict(list)
for r in csv.DictReader(open(META),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')]; ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        box=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if box[2]-box[0]>=4 and box[3]-box[1]>=4: by_vid[r['video_id']].append((paths[r['filename']],box,r['finding_class']))
net=fasterrcnn_resnet50_fpn_v2(weights=None)
net.roi_heads.box_predictor=FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features,2)
net.load_state_dict(torch.load('data/kvasir_detector_v3.pth',map_location='cpu')); net.eval().to(dev)
os.makedirs(OUT,exist_ok=True)
vids=sorted(by_vid,key=lambda v:-len(by_vid[v]))[:5]
def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
for v in vids:
    items=by_vid[v][::max(1,len(by_vid[v])//6)][:6]; vd=os.path.join(OUT,v); os.makedirs(vd,exist_ok=True)
    for i,(p,gt,cls) in enumerate(items):
        img=Image.open(p).convert('RGB').resize((W,W)); x=TF.to_tensor(img).unsqueeze(0).to(dev)
        with torch.no_grad(): o=net(x)[0]
        im=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        cv2.rectangle(im,(gt[0],gt[1]),(gt[2],gt[3]),(0,200,0),2)
        tag='no-detect'
        if len(o['boxes'])>0:
            pb=[int(c) for c in o['boxes'][0].cpu().tolist()]; sc=float(o['scores'][0])
            cv2.rectangle(im,(pb[0],pb[1]),(pb[2],pb[3]),(0,0,255),2); tag='IoU=%.2f sc=%.2f'%(iou(pb,gt),sc)
        cv2.putText(im,'%s | %s'%(cls,tag),(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)
        cv2.imwrite(os.path.join(vd,'%02d.jpg'%i),im)
    print('%s: %d ảnh'%(v,len(items)))
print('[*] xong -> %s/'%OUT)
