"""So bo tom tat RL vs chon deu vs ngau nhien: ty le frame duoc chon la TON THUONG that.
Base rate ton thuong ~9%. Neu RL >> base rate va >> uniform/random => bo tom tat co gia tri."""
import sys, math, numpy as np, h5py, torch
sys.path.insert(0,'scripts')
from models import DSN

H5='data/kvasir_capsule.h5'
MODEL=sys.argv[1] if len(sys.argv)>1 else 'output_kvasir/sup/split0/model_Rall_sup_ep49_F19.77.pth'
PROP=0.15
dev='cuda' if torch.cuda.is_available() else 'cpu'

sd=torch.load(MODEL,map_location='cpu')
net=DSN(in_dim=512,hid_dim=sd['rnn.weight_hh_l0'].shape[1],num_layers=1,cell='lstm')
net.load_state_dict(sd); net.eval().to(dev)

f=h5py.File(H5,'r'); ks=[k for k in f.keys() if 0<f[k]['gtscore'][...].mean()<1]  # video hon hop
rng=np.random.RandomState(0)
res={'rl':[], 'uniform':[], 'random':[], 'base':[]}
for k in ks:
    feat=f[k]['features'][...].astype(np.float32); g=f[k]['gtscore'][...]
    n=len(g); lim=max(1,int(n*PROP))
    x=torch.from_numpy(feat).unsqueeze(0); x=(x-x.mean())/(x.std()+1e-5)
    with torch.no_grad(): p=net(x.to(dev)).squeeze().cpu().numpy().flatten()
    rl=np.argsort(p)[-lim:]
    uni=np.linspace(0,n-1,lim).astype(int)
    ran=rng.choice(n,lim,replace=False)
    res['rl'].append(g[rl].mean())
    res['uniform'].append(g[uni].mean())
    res['random'].append(g[ran].mean())
    res['base'].append(g.mean())
print('So video hon hop:',len(ks),'| ngan sach chon:',int(PROP*100),'%')
print('Ty le frame duoc chon LA TON THUONG (cao = tot):')
for m in ['rl','uniform','random','base']:
    print('  %-8s %.1f%%'%(m,np.mean(res[m])*100))
