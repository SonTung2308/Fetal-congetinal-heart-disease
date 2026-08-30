"""Prep v2: dùng đặc trưng FPN của DETECTOR v3 (đã học tổn thương) thay ImageNet đóng băng.
Lưu: init box (detector v3), gt box, feature map FPN level '2' (256 x ~21 x 21) để ROI-pool.
Đây là tín hiệu lesion-aware cho RL agent."""
import csv, glob, os, numpy as np, torch, h5py
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

META='data/kvasir/metadata.csv'; ROOT='data/kvasir/labelled_images'; dev='cuda'; W=336
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob(ROOT+'/**/*.jpg',recursive=True)}
items=[]
for r in csv.DictReader(open(META),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in ('x1','x2','x3','x4')]; ys=[int(r[k]) for k in ('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: items.append((paths[r['filename']],b,r['video_id']))
print('ảnh:',len(items))
net=fasterrcnn_resnet50_fpn_v2(weights=None)
net.roi_heads.box_predictor=FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features,2)
net.load_state_dict(torch.load('data/kvasir_detector_v3.pth',map_location='cpu')); net.eval().to(dev)
LEVEL='2'   # FPN level, stride 16 -> ~21x21 cho input 336
fout=h5py.File('data/rl_detect2.h5','w'); B=12; buf=[]; ctr=0
def flush(buf):
    global ctr
    if not buf: return
    imgs=[TF.to_tensor(Image.open(p).convert('RGB').resize((W,W))).to(dev) for p,_,_ in buf]
    with torch.no_grad():
        out=net(imgs)                                    # box detector
        feats=net.backbone(torch.stack(imgs))            # OrderedDict FPN
        fm=feats[LEVEL].cpu().numpy()                    # (B,256,h,w)
    for i,(p,gt,vid) in enumerate(buf):
        ctr+=1
        ib=out[i]['boxes'][0].cpu().numpy() if len(out[i]['boxes'])>0 else np.array([W*.25,W*.25,W*.75,W*.75],np.float32)
        g=fout.create_group('img%06d'%ctr); g.attrs['vid']=vid
        g.create_dataset('gt',data=np.array(gt,np.float32))
        g.create_dataset('init',data=ib.astype(np.float32))
        g.create_dataset('fm',data=fm[i].astype(np.float32),compression='gzip')
for it in items:
    buf.append(it)
    if len(buf)>=B: flush(buf); buf=[]
flush(buf);
hw=fout['img000001']['fm'].shape
fout.close(); print('[*] xong -> data/rl_detect2.h5 |',len(items),'ảnh | fm',hw)
