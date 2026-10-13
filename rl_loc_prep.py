"""Rainbow localization — Bước 1: precompute conv feature-map (ResNet18 layer4, 512x11x11) cho mỗi
frame bệnh có GT box. State của agent = roi_align(featmap, cửa sổ hiện tại) → khỏi chạy CNN mỗi bước.
Lưu: fm(512,11,11), gt(4 = xyxy trong toạ độ W=336), vid."""
import csv, glob, os, numpy as np, torch, torch.nn as nn, h5py
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import models
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
enc=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
back=nn.Sequential(enc.conv1,enc.bn1,enc.relu,enc.maxpool,enc.layer1,enc.layer2,enc.layer3,enc.layer4).eval().to(dev)
M=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); S=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
fout=h5py.File('data/loc_feat.h5','w'); B=32; ctr=0
for i in range(0,len(items),B):
    ch=items[i:i+B]; ims=[Image.open(p).convert('RGB').resize((W,W)) for p,_,_ in ch]
    x=torch.stack([TF.to_tensor(im) for im in ims]).to(dev)
    with torch.no_grad(): fm=back((x-M)/S).cpu().numpy().astype(np.float16)
    for j,(p,gt,vid) in enumerate(ch):
        ctr+=1; g=fout.create_group('i%05d'%ctr); g.attrs['vid']=vid
        g.create_dataset('fm',data=fm[j]); g.create_dataset('gt',data=np.array(gt,np.float32))
fout.close(); print('[*] xong',ctr,'-> data/loc_feat.h5 | fm shape',fm.shape[1:])
