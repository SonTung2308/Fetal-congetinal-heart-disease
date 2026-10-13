"""RL localization detector-free CHẠY ĐƯỢC: policy xuất TRỰC TIẾP box trên backbone ResNet18-ImageNet (160px).
Reward = DIoU(box,GT); REINFORCE + value baseline + advantage chuẩn hoá + entropy anneal (ổn định, không sập).
KHÔNG detector (không RPN/anchor/NMS). Học thuần từ reward. Chia test theo VIDEO. Có augment."""
import sys, csv, glob, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random
from PIL import Image
from torchvision import models
dev='cuda'; W=160; STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 6000; B=48
torch.manual_seed(0); np.random.seed(0); random.seed(0)
PATH={'Angiectasia','Blood - fresh','Blood - hematin','Erosion','Erythema','Foreign Body','Lymphangiectasia','Polyp','Ulcer'}
paths={os.path.basename(p):p for p in glob.glob('data/kvasir/labelled_images/**/*.jpg',recursive=True)}
IMG=[];GT=[];VID=[]
for r in csv.DictReader(open('data/kvasir/metadata.csv'),delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1','').strip():
        xs=[int(r[k]) for k in('x1','x2','x3','x4')];ys=[int(r[k]) for k in('y1','y2','y3','y4')]
        b=[max(0,min(xs)),max(0,min(ys)),min(336,max(xs)),min(336,max(ys))]
        if b[2]-b[0]<4 or b[3]-b[1]<4: continue
        IMG.append(np.asarray(Image.open(paths[r['filename']]).convert('RGB').resize((W,W)),np.uint8)); GT.append([c*W/336 for c in b]); VID.append(r['video_id'])
IMG=np.stack(IMG); GT=np.array(GT,np.float32)
vids=sorted(set(VID)); rng=np.random.RandomState(0); o=list(vids); rng.shuffle(o); te=set(o[:max(3,int(0.25*len(o)))])
tr=np.array([i for i in range(len(IMG)) if VID[i] not in te]); teI=np.array([i for i in range(len(IMG)) if VID[i] in te])
print('frame train %d | test %d | video test %d | %dpx'%(len(tr),len(teI),len(te),W))
MEAN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); STD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
def batch_imgs(ids,aug):
    ims=IMG[ids].astype(np.float32).copy(); gts=GT[ids].copy()
    if aug:
        for j in range(len(ids)):
            if random.random()<0.5: ims[j]=ims[j,:,::-1,:]; x1,y1,x2,y2=gts[j]; gts[j]=[W-x2,y1,W-x1,y2]
            if random.random()<0.5: ims[j]=ims[j,::-1,:,:]; x1,y1,x2,y2=gts[j]; gts[j]=[x1,W-y2,x2,W-y1]
        ims=np.clip(ims*random.uniform(0.8,1.2)+random.uniform(-20,20),0,255)
    x=torch.tensor(ims.transpose(0,3,1,2)/255.,dtype=torch.float32,device=dev); x=(x-MEAN)/STD
    return x,torch.tensor(gts,device=dev)
def iou_t(a,b):
    x0=torch.max(a[:,0],b[:,0]);y0=torch.max(a[:,1],b[:,1]);x1=torch.min(a[:,2],b[:,2]);y1=torch.min(a[:,3],b[:,3])
    inter=(x1-x0).clamp(min=0)*(y1-y0).clamp(min=0); ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter/ua.clamp(min=1e-6)
def diou_t(a,b):
    io=iou_t(a,b); ca=(a[:,:2]+a[:,2:])/2; cb=(b[:,:2]+b[:,2:])/2; rho2=((ca-cb)**2).sum(1)
    ex0=torch.min(a[:,0],b[:,0]);ey0=torch.min(a[:,1],b[:,1]);ex1=torch.max(a[:,2],b[:,2]);ey1=torch.max(a[:,3],b[:,3])
    c2=((ex1-ex0)**2+(ey1-ey0)**2).clamp(min=1e-6); return io-rho2/c2
def to_box(p):
    cx=p[:,0]*W;cy=p[:,1]*W;bw=(0.05+0.95*p[:,2])*W;bh=(0.05+0.95*p[:,3])*W
    return torch.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],1).clamp(0,W)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        s.bb=nn.Sequential(*list(m.children())[:-1]); s.fc=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3))
        s.mu=nn.Linear(256,4); s.val=nn.Linear(256,1); s.logstd=nn.Parameter(torch.zeros(4)-1.0)
    def forward(s,x): h=s.fc(s.bb(x).flatten(1)); return s.mu(h),s.val(h).squeeze(1)
net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
def evaluate(ids):
    net.eval(); I=[]
    with torch.no_grad():
        for k in range(0,len(ids),128):
            x,g=batch_imgs(ids[k:k+128],0); mu,_=net(x); I.append(iou_t(to_box(torch.sigmoid(mu)),g).cpu().numpy())
    net.train(); I=np.concatenate(I); return I.mean(),(I>=.5).mean()
best=0
for it in range(1,STEPS+1):
    ids=tr[np.random.randint(0,len(tr),B)]; x,g=batch_imgs(ids,1)
    mu,V=net(x); std=net.logstd.exp().clamp(0.03,0.7); dist=torch.distributions.Normal(mu,std)
    z=dist.sample(); box=to_box(torch.sigmoid(z)); R=diou_t(box,g)
    logp=dist.log_prob(z).sum(1); adv=(R-V).detach(); adv=(adv-adv.mean())/(adv.std()+1e-6)   # chuẩn hoá -> ổn định
    entc=max(0.0,0.01*(1-it/STEPS))
    loss=-(adv*logp).mean()+0.5*F.mse_loss(V,R)-entc*dist.entropy().sum(1).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
    if it%400==0:
        mi,dr=evaluate(teI); tri,tdr=evaluate(tr[:800])
        if mi>best: best=mi
        print('it %5d | R %.3f | TEST IoU %.3f DR %.1f%% | TRAIN IoU %.3f DR %.1f%% | std %.2f'%(it,R.mean().item(),mi,dr*100,tri,tdr*100,std.mean().item()),flush=True)
mi,dr=evaluate(teI); tri,tdr=evaluate(tr)
print('\n=== RL DIRECT-BOX + backbone ImageNet (detector-free, học thuần từ reward) ===')
print('TEST : IoU %.3f (best %.3f) | DR@0.5 %.1f%%'%(mi,best,dr*100))
print('TRAIN: IoU %.3f | DR@0.5 %.1f%%'%(tri,tdr*100))
print('(RL feature-yếu trước = 0.11 ; hồi quy giám sát cùng backbone = 0.59 ; detector = 0.61/76%%)')
torch.save(net.state_dict(),'data/rl_direct2.pth')
