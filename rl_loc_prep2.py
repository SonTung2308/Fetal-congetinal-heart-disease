"""Rainbow localization — precompute feature-map LESION-AWARE từ backbone detector v3 (fine-tune),
thay cho ResNet18-ImageNet. Vẫn KHÔNG dùng box của detector — chỉ mượn feature map làm 'mắt' cho agent.
FPN level '0' của fasterrcnn_resnet50_fpn_v2: 256 kênh, ~84x84 cho input 336. Lưu fm(256,H,W), gt, vid."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, h5py
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
dev='cuda'; W=336
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob('data/kvasir/labelled_images/**/*.jpg',recursive=True)}
items=[]
for r in csv.DictReader(open('data/kvasir/metadata.csv'),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in('x1','x2','x3','x4')];ys=[int(r[k]) for k in('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(W,max(xs)),min(W,max(ys))]
        if b[2]-b[0]>=4 and b[3]-b[1]>=4: items.append((paths[r['filename']],b,r['video_id']))
print('frame benh co GT:',len(items))
det=fasterrcnn_resnet50_fpn_v2(weights=None)
det.roi_heads.box_predictor=FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features,2)
det.load_state_dict(torch.load('data/kvasir_detector_v3.pth',map_location='cpu')); det.eval().to(dev)
bb=det.backbone  # ResNet50-FPN fine-tune
# downsample FPN '0' (stride4 -> 84x84) xuong 21x21 cho gon
pool=nn.AdaptiveAvgPool2d((21,21))
fout=h5py.File('data/loc_feat2.h5','w'); B=16; ctr=0
for i in range(0,len(items),B):
    ch=items[i:i+B]; ims=[TF.to_tensor(Image.open(p).convert('RGB').resize((W,W))) for p,_,_ in ch]
    x=torch.stack(ims).to(dev)
    with torch.no_grad():
        fpn=bb(x)['0']; fm=pool(fpn).cpu().numpy().astype(np.float16)  # (B,256,21,21)
    for j,(p,gt,vid) in enumerate(ch):
        ctr+=1; g=fout.create_group('i%05d'%ctr); g.attrs['vid']=vid
        g.create_dataset('fm',data=fm[j]); g.create_dataset('gt',data=np.array(gt,np.float32))
fout.close(); print('[*] xong',ctr,'-> data/loc_feat2.h5 | fm',fm.shape[1:])
