"""Trich dac trung tu VIDEO FULL Kvasir-Capsule (labelled_videos).
Khac ban labelled_images: doc MOI frame trong .mp4, ton thuong THUA that (<1%).
Nhan tung frame theo frame_number trong metadata; frame khong co trong metadata = binh thuong(0).

Chi lay tap con frame de nhe: giu het frame co nhan (positive + negative da label)
+ lay mau 1/STRIDE frame con lai lam nen. Tuy chon --all de lay het."""
import argparse, csv, glob, os
from collections import defaultdict
import cv2, h5py, numpy as np, torch, torch.nn as nn
from torchvision import models
from tqdm import tqdm

P = argparse.ArgumentParser()
P.add_argument('--vdir', default='data/kvasir/labelled_videos')
P.add_argument('--meta', default='data/kvasir/metadata.csv')
P.add_argument('--out', default='data/kvasir_video.h5')
P.add_argument('--stride', type=int, default=10, help='lay 1/stride frame nen (khong nhan)')
P.add_argument('--weights', default='', help='.pth ResNet18 fine-tune; trong = ImageNet')
A = P.parse_args()

PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema',
      'Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
dev='cuda'
net=models.resnet18(weights=None if A.weights else models.ResNet18_Weights.DEFAULT)
if A.weights:
    net.fc=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,1),nn.Sigmoid())
    sd=torch.load(A.weights,map_location='cpu')
    net.load_state_dict({k.replace('base_model.',''):v for k,v in sd.items()})
net.fc=nn.Identity(); net.eval().to(dev)
mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)

# nhan tung frame: (video_id, frame_number) -> 1/0
rows=list(csv.DictReader(open(A.meta),delimiter=';'))
lab=defaultdict(dict)   # video_id -> {frame_number: 0/1}
for r in rows:
    lab[r['video_id']][int(r['frame_number'])]=1 if r['finding_class'] in PATH else 0

vids=sorted(glob.glob(A.vdir+'/*.mp4'))
print('[*]',len(vids),'video')

def feats(frames):
    x=torch.from_numpy(np.stack(frames)).float().permute(0,3,1,2).to(dev)/255.0
    x=(x-mean)/std
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),256): out.append(net(x[i:i+256]).cpu().numpy())
    return np.concatenate(out,0)

with h5py.File(A.out,'w') as f5:
    for vp in tqdm(vids,desc='video'):
        vid=os.path.splitext(os.path.basename(vp))[0]
        vlab=lab.get(vid,{})
        cap=cv2.VideoCapture(vp)
        idx=0; keep_frames=[]; keep_idx=[]; keep_lab=[]
        while True:
            ok,fr=cap.read()
            if not ok: break
            has=idx in vlab
            if has or (idx % A.stride==0):
                keep_frames.append(cv2.resize(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),(224,224)))
                keep_idx.append(idx)
                keep_lab.append(vlab.get(idx,0))
            idx+=1
        cap.release()
        if len(keep_frames)<10: continue
        # trich theo lo de khong tran RAM
        F=[]
        for i in range(0,len(keep_frames),512):
            F.append(feats(keep_frames[i:i+512]))
        feat=np.concatenate(F,0)
        g=f5.create_group(vid)
        g.create_dataset('features',data=feat,compression='gzip')
        g.create_dataset('gtscore',data=np.array(keep_lab,dtype=np.float32),compression='gzip')
        g.create_dataset('frame_idx',data=np.array(keep_idx,dtype=np.int32))
        g.create_dataset('n_frames',data=feat.shape[0])
        g.create_dataset('picks',data=np.arange(feat.shape[0],dtype=np.int32))
print('[*] xong ->',A.out)
