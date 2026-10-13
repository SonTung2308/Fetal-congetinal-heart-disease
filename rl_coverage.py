"""RL chọn frame để PHỦ HẾT các đoạn bệnh trong video (segment-coverage) — nơi RL thắng thật.
Mục tiêu tổ hợp, KHÔNG khả vi: reward = +1 khi frame chọn thuộc ĐOẠN MỚI (phụ thuộc tập đã chọn -> MDP).
Policy chấm điểm frame theo [feature, score, vị trí, khoảng cách tới frame đã chọn, %budget]; chọn tuần tự.
REINFORCE + baseline. So score-greedy / random / oracle. Chia train/test theo VIDEO."""
import sys, numpy as np, h5py, torch, torch.nn as nn, random
dev='cuda'; EP=int(sys.argv[1]) if len(sys.argv)>1 else 4000
torch.manual_seed(0); np.random.seed(0); random.seed(0)
h=h5py.File('data/kvasir_video_ft.h5','r'); ks=list(h.keys())
def segid_of(gs):
    sid=np.zeros(len(gs),int); cur=0
    for i,v in enumerate(gs):
        if v>0 and (i==0 or gs[i-1]==0): cur+=1
        sid[i]=cur if v>0 else 0
    return sid,cur
def prep(g,cap=1200):
    fi=np.asarray(h[g]['frame_idx']); o=np.argsort(fi)
    gs=np.asarray(h[g]['gtscore']).astype(int)[o]; sc=np.asarray(h[g]['score']).astype(np.float32)[o]; ft=np.asarray(h[g]['features']).astype(np.float32)[o]
    sid,nseg=segid_of(gs)
    pos=np.where(sid>0)[0]; neg=np.where(sid==0)[0]
    keepneg=np.random.RandomState(0).choice(neg,min(len(neg),max(0,cap-len(pos))),replace=False) if len(neg)>0 else np.array([],int)
    idx=np.sort(np.concatenate([pos,keepneg]))
    return ft[idx],sc[idx],sid[idx],nseg,np.arange(len(idx))/max(1,len(idx))
VID=[g for g in ks if segid_of(np.asarray(h[g]['gtscore']).astype(int))[1]>0]
rng=np.random.RandomState(1); order=list(VID); rng.shuffle(order); te=set(order[:6])
DATA={g:prep(g) for g in VID}
# chuan hoa feature theo train
Ftr=np.concatenate([DATA[g][0] for g in VID if g not in te]); mu=Ftr.mean(0); sd=Ftr.std(0)+1e-6
tr=[g for g in VID if g not in te]; teL=[g for g in VID if g in te]
print('video train %d | test %d | tong segment test %d'%(len(tr),len(teL),sum(DATA[g][3] for g in teL)))
# feature chuan hoa L2 (cho cosine) cache
FN={g:((DATA[g][0]-mu)/sd) for g in VID}
FL2={g:(FN[g]/(np.linalg.norm(FN[g],axis=1,keepdims=True)+1e-6)) for g in VID}
D=512+5
class Pol(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(D,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,x): return s.net(x).squeeze(-1)
pol=Pol().to(dev); opt=torch.optim.Adam(pol.parameters(),lr=3e-4,weight_decay=1e-4)
def statef(g,selected):
    ft,sc,sid,nseg,pos=DATA[g]; n=len(sid); fn=FN[g]; fl=FL2[g]
    if selected:
        sel=np.array(selected); md=np.min(np.abs(pos[:,None]-pos[sel][None,:]),1)
        cos=np.max(fl@fl[sel].T,1)                     # tương đồng feature max tới frame đã chọn (cao=trùng)
    else: md=np.ones(n); cos=np.zeros(n)
    frac=np.full(n,len(selected)/max(1,K_of(nseg)),np.float32)
    x=np.concatenate([fn,sc[:,None],pos[:,None],md[:,None],cos[:,None],frac[:,None]],1).astype(np.float32)
    return torch.tensor(x,device=dev)
def K_of(nseg): return max(2,2*nseg)
def run_episode(g,greedy=False):
    ft,sc,sid,nseg,pos=DATA[g]; n=len(sid); K=K_of(nseg); selected=[];covered=set();logps=[];rs=[]
    for t in range(min(K,n)):
        x=statef(g,selected); lg=pol(x)
        mask=torch.zeros(n,device=dev);
        if selected: mask[torch.tensor(selected,device=dev)]=-1e9
        lg=lg+mask
        if greedy: a=int(lg.argmax())
        else:
            d=torch.distributions.Categorical(logits=lg); ai=d.sample(); a=int(ai); logps.append(d.log_prob(ai))
        s=sid[a]; r=1.0 if (s>0 and s not in covered) else -0.05
        if s>0: covered.add(s)
        selected.append(a); rs.append(r)
    rec=len(covered)/nseg
    return rec,logps,rs
best=0; bmean={}  # baseline running-mean theo bước
for ep in range(1,EP+1):
    g=random.choice(tr); rec,logps,rs=run_episode(g,greedy=False)
    if logps:
        G=np.cumsum(rs[::-1])[::-1].astype(np.float32)          # return-to-go
        adv=[]
        for t,gt in enumerate(G):
            b=bmean.get(t,0.0); bmean[t]=0.95*b+0.05*gt; adv.append(gt-bmean[t])
        adv=torch.tensor(np.array(adv),device=dev); adv=(adv-adv.mean())/(adv.std()+1e-6)
        loss=-(torch.stack(logps)*adv).sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),5.0); opt.step()
    if ep%400==0:
        pol.eval()
        with torch.no_grad():
            rl=np.mean([run_episode(g,greedy=True)[0] for g in teL])
            sg=[];rnd=[];orac=[]
            for g in teL:
                ft,sc,sid,nseg,pos=DATA[g]; n=len(sid); K=K_of(nseg)
                sgsel=np.argsort(sc)[-K:]; sg.append(len(set(sid[sgsel][sid[sgsel]>0]))/nseg)
                rs2=np.random.RandomState(0).permutation(n)[:K]; rnd.append(len(set(sid[rs2][sid[rs2]>0]))/nseg)
                orac.append(min(1.0,K/nseg))
        pol.train()
        if rl>best: best=rl
        print('ep %5d | TEST seg-recall: RL %.3f | score-greedy %.3f | random %.3f | oracle %.3f'%(ep,rl,np.mean(sg),np.mean(rnd),np.mean(orac)),flush=True)
print('\n=== RL SEGMENT-COVERAGE (reward không khả vi = phủ đoạn mới) ===')
print('best RL seg-recall test %.3f | score-greedy %.3f | oracle %.3f'%(best,np.mean(sg),np.mean(orac)))
torch.save(pol.state_dict(),'data/rl_coverage.pth')
