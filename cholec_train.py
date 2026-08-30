"""Nhan dien 7 giai doan phau thuat Cholec80. Fine-tune ResNet18, chia theo VIDEO.
Chi so: accuracy, macro-F1, per-phase recall."""
import sys, glob, os, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
ROOT='data/cholec_frames'; dev='cuda'
PHASES=['Preparation','CalotTriangleDissection','ClippingCutting','GallbladderDissection',
        'GallbladderPackaging','CleaningCoagulation','GallbladderRetraction']
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 5
imgs_by_vid=defaultdict(list)
for v in sorted(os.listdir(ROOT)):
    for p in glob.glob(os.path.join(ROOT,v,'*.jpg')):
        lab=int(os.path.basename(p).split('_p')[1].split('.')[0])
        imgs_by_vid[v].append((p,lab))
vids=sorted(imgs_by_vid)
print('video:',len(vids),'| frame:',sum(len(imgs_by_vid[v]) for v in vids))
rng=np.random.RandomState(0); order=list(vids); rng.shuffle(order)
nte=max(4,int(0.3*len(order))); te=set(order[:nte]); tr=[v for v in order if v not in te]
tr_items=[it for v in tr for it in imgs_by_vid[v]]
te_by_vid={v:imgs_by_vid[v] for v in te}
print('train',len(tr_items),'frame |',len(te),'video test')
TF_TR=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_TE=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __init__(s,it,tf): s.it=it; s.tf=tf
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        p,l=s.it[i]; return s.tf(Image.open(p).convert('RGB')),l
# can bang lop
cnt=np.bincount([l for _,l in tr_items],minlength=7)
w=torch.tensor((cnt.sum()/(7*np.maximum(cnt,1))),dtype=torch.float32,device=dev)
net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.fc=nn.Linear(512,7); net=net.to(dev)
opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
lossf=nn.CrossEntropyLoss(weight=w)
dl=DataLoader(DS(tr_items,TF_TR),batch_size=96,shuffle=True,num_workers=8,pin_memory=True)
for ep in range(EPOCHS):
    net.train(); tot=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(); l=lossf(net(x),y); l.backward(); opt.step(); tot+=l.item()
    print('epoch %d/%d loss %.3f'%(ep+1,EPOCHS,tot/len(dl)),flush=True)
net.eval(); yall=[]; pall=[]
for v,it in te_by_vid.items():
    dlt=DataLoader(DS(it,TF_TE),batch_size=256,num_workers=6)
    with torch.no_grad():
        for x,y in dlt:
            pr=net(x.to(dev)).argmax(1).cpu().numpy(); yall+=list(y.numpy()); pall+=list(pr)
y=np.array(yall); p=np.array(pall)
acc=(y==p).mean()
print('\n=== KET QUA (test %d video, %d frame) ==='%(len(te),len(y)))
print('Accuracy: %.1f%%'%(acc*100))
f1s=[]
print('%-26s %8s %8s %8s'%('phase','recall','prec','F1'))
for i,ph in enumerate(PHASES):
    tp=((y==i)&(p==i)).sum(); fn=((y==i)&(p!=i)).sum(); fp=((y!=i)&(p==i)).sum()
    rec=tp/max(tp+fn,1); pre=tp/max(tp+fp,1); f1=2*rec*pre/max(rec+pre,1e-9); f1s.append(f1)
    print('%-26s %7.0f%% %7.0f%% %7.0f%%'%(ph,rec*100,pre*100,f1*100))
print('Macro-F1: %.1f%%'%(np.mean(f1s)*100))
