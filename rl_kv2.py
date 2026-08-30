"""RL selector Kvasir (bản 2): policy Conv1d trên chuỗi ĐÃ GIẢM ĐỘ PHÂN GIẢI (stride S),
tránh lỗi cuDNN chuỗi dài. Reward = segment_recall - lambda*budget. So với tham lam top-k%."""
import numpy as np, h5py, torch, torch.nn as nn
dev = 'cuda'; GAP = 45; LAM = 0.5; EPISODES = 8; EPOCHS = 40; S = 20
fs = h5py.File('data/kvasir_video_score_full.h5', 'r')
fl = h5py.File('data/kvasir_video_fulllabels.h5', 'r')
vids = [v for v in fs.keys() if fl[v]['gtscore'][...].sum() > 0]


def smooth(p, w):
    k = np.ones(w) / w
    return np.convolve(p, k, mode='same')


def segs(g):
    idx = np.where(g > 0)[0]
    if len(idx) == 0:
        return []
    s = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - s[-1][1] <= GAP:
            s[-1][1] = i
        else:
            s.append([i, i])
    return s


featc = {}; segbin = {}; nbin = {}
for v in vids:
    sc = fs[v]['score'][...]; n = len(sc)
    nb = (n + S - 1) // S
    binmax = np.array([sc[i * S:(i + 1) * S].max() for i in range(nb)], np.float32)
    binmean = np.array([sc[i * S:(i + 1) * S].mean() for i in range(nb)], np.float32)
    pos = np.linspace(0, 1, nb).astype(np.float32)
    featc[v] = np.stack([binmax, binmean, smooth(binmax, 5).astype(np.float32), pos], 1).astype(np.float32)
    nbin[v] = nb
    segbin[v] = [[s[0] // S, s[1] // S] for s in segs(fl[v]['gtscore'][...])]


def seg_recall_bin(v, selbin):
    S_ = segbin[v]
    if not S_:
        return 1.0
    return sum(1 for a, b in S_ if selbin[a:b + 1].any()) / len(S_)


class Pol(nn.Module):
    def __init__(s, d=4, h=64):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(d, h, 15, padding=7), nn.ReLU(),
                              nn.Conv1d(h, h, 15, padding=7), nn.ReLU(), nn.Conv1d(h, 1, 1))

    def forward(s, x):
        z = x.transpose(1, 2).contiguous()
        return torch.sigmoid(s.net(z)).squeeze(1).squeeze(0)


def tin(v):
    return torch.from_numpy(featc[v]).float().unsqueeze(0).to(dev)


rng = np.random.RandomState(0); order = np.array(vids); rng.shuffle(order)
folds = np.array_split(order, 5); rl_prob = {}
for te in folds:
    te = set(te.tolist()); tr = [v for v in vids if v not in te]
    net = Pol().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    base = {v: 0.0 for v in tr}
    for ep in range(EPOCHS):
        for v in tr:
            p = net(tin(v))
            m = torch.distributions.Bernoulli(p.clamp(1e-4, 1 - 1e-4))
            loss = 0
            for _ in range(EPISODES):
                a = m.sample(); sel = a.bool().cpu().numpy()
                r = seg_recall_bin(v, sel) - LAM * a.float().mean().item()
                base[v] = 0.9 * base[v] + 0.1 * r
                loss = loss - (m.log_prob(a).mean() * (r - base[v]))
            (loss / EPISODES).backward(); opt.step(); opt.zero_grad()
    net.eval()
    for v in te:
        with torch.no_grad():
            rl_prob[v] = net(tin(v)).cpu().numpy()

print('So sanh SEGMENT-RECALL cung ngan sach (tham lam top-k%% vs RL):')
print('%8s %12s %12s' % ('budget', 'tham lam', 'RL'))
allseg = [s for v in vids for s in segbin[v]]
for kb in [5, 10, 20, 30]:
    hg = hr = 0
    for v in vids:
        nb = nbin[v]; lim = max(1, int(nb * kb / 100))
        scb = featc[v][:, 0]
        selg = np.zeros(nb, bool); selg[np.argsort(scb)[-lim:]] = True
        selr = np.zeros(nb, bool); selr[np.argsort(rl_prob[v])[-lim:]] = True
        for a, b in segbin[v]:
            if selg[a:b + 1].any(): hg += 1
            if selr[a:b + 1].any(): hr += 1
    print('%7d%% %11.0f%% %11.0f%%' % (kb, hg / len(allseg) * 100, hr / len(allseg) * 100))
