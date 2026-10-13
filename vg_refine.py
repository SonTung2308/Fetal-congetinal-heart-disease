"""Box refinement trên LƯỚI: xuất phát từ box top-1 của detector, agent dịch chuyển box theo từng Ô LƯỚI (lưới 28x28,
ô = 12 px) để tiến tới GT. 13 hành động: dời cả box trái/phải/lên/xuống, nới từng cạnh ra (4), thu từng cạnh vào (4),
STOP. Tối đa T=15 bước.
Reward mỗi bước: 10 * ΔIoU(box, GT) - 0.05 (STOP = 0) -> tổng reward = 10 * (IoU cuối - IoU đầu) - 0.05 * số bước:
tối đa tổng reward <=> tối đa IoU cuối, phạt nhẹ số bước để agent biết dừng.
State: RoIAlign lưới đặc trưng FPN trên vùng box nới 1.6x (10x10) + mặt nạ box hiện tại + toạ độ box, bước t, score,
hành động vừa làm (one-hot). Hành động đảo ngược bước trước bị cấm (chặn dao động qua lại).
Học: Double + Dueling DQN, 64 môi trường song song trên GPU. Điểm xuất phát khi train (tham số start):
  mix : 50% box detector OOF, 50% GT lệch ±30% mỗi cạnh  (bản đầu: agent học 'luôn dịch vài bước', hỏng box tốt)
  curr: 1/3 box detector OOF, 2/3 GT lệch với MỨC LỆCH NGẪU NHIÊN 0–35% (IoU xuất phát trải ~0.4–1.0, có nhiều
        box đã gần đúng -> agent phải học khi nào STOP).
So với: box detector gốc, BẮT CHƯỚC oracle (giám sát: cross-entropy tới hành động tăng IoU nhiều nhất), oracle
tham lam (dùng GT, cận trên). Đánh giá trên 1.068 frame bệnh của 5 video test (split sạch).
Trực quan hoá: ở mỗi mốc train (2.000 vòng) vẽ lại fig_refine_progress_<method>.png (box qua từng bước trên 6 frame
test cố định) và fig_refine_curve.png; cuối cùng fig_refine_steps_<method>.png (vg_refine_vis.py).
Chạy: python vg_refine.py [iterations=20000] [mix|curr] [holdout|testfeat k]  -> data/vg_refine[_curr][_hold<k>|_testfeat<k>].json
Chẩn đoán: holdout k = train trên 3 fold video, đánh giá trên fold video k; testfeat k = đặc trưng test từ backbone OOF k."""
from __future__ import annotations

import json
import sys
import time
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

import vg_precompute as vp
import vg_refine_vis as rv

DEV = 'cuda'; W = 336.0; G = 28; CELL = W / G; T = 15; NA = 13; STOP = 12; VIEW = 10; EXP = 1.6
NENV = 64; GAMMA = 0.9; STEP_COST = 0.05; R_SCALE = 10.0; NSC = 8 + 13
DELTA = torch.tensor([[-1, 0, -1, 0], [1, 0, 1, 0], [0, -1, 0, -1], [0, 1, 0, 1],      # dời cả box
                      [-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1],        # nới cạnh
                      [1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, -1],        # thu cạnh
                      [0, 0, 0, 0]], dtype=torch.float32, device=DEV) * CELL
INV = torch.tensor([1, 0, 3, 2, 8, 9, 10, 11, 4, 5, 6, 7, 12], device=DEV)   # hành động đảo ngược


def valid(last: torch.Tensor) -> torch.Tensor:
    """(B,NA): cấm hành động đảo ngược hành động vừa làm (chặn vòng lặp 2 bước, vd. nới rồi thu cùng cạnh)."""
    m = torch.ones(len(last), 13, dtype=torch.bool, device=DEV); has = (last >= 0) & (last != 12)
    m[has.nonzero().squeeze(1), INV[last[has]]] = False
    return m


def masked_argmax(q: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~valid(last), -1e9).argmax(1)


def iou_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x0 = torch.maximum(a[..., 0], b[..., 0]); y0 = torch.maximum(a[..., 1], b[..., 1])
    x1 = torch.minimum(a[..., 2], b[..., 2]); y1 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    return inter / ua.clamp_min(1e-6)


def apply(box: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Áp hành động (dịch 1 ô lưới); giữ box trong ảnh và >= 2 ô mỗi chiều, vi phạm thì giữ nguyên."""
    nb = (box + DELTA[a]).clamp(0, W)
    ok = ((nb[:, 2] - nb[:, 0]) >= 2 * CELL) & ((nb[:, 3] - nb[:, 1]) >= 2 * CELL)
    return torch.where(ok[:, None], nb, box)


class Data:
    """Frame bệnh có GT: lưới FPN 28x28 (chuẩn hoá theo train), box detector top-1, GT, split."""

    def __init__(self) -> None:
        r = h5py.File('data/vg_refine_fm.h5', 'r'); row = r['row'][...]
        h = h5py.File('data/vg_frames.h5', 'r')
        self.start = torch.from_numpy(h['boxes'][...][row, 0]).to(DEV)
        self.gt = torch.from_numpy(h['gt'][...][row]).to(DEV)
        self.score = torch.from_numpy(h['scores'][...][row, 0]).to(DEV)
        self.split = h['split'][...][row]; self.vid = h['vid'][...].astype(str)[row]; self.fold = h['fold'][...][row]
        self.row = row
        fm = torch.from_numpy(r['fm'][...]).to(DEV)                                    # fp16 (M,256,G,G)
        tr = torch.from_numpy(np.where(self.split == 0)[0]).to(DEV)
        mu = torch.stack([fm[tr[i:i + 256]].float().mean((0, 2, 3)) for i in range(0, len(tr), 256)]).mean(0)
        var = torch.stack([((fm[tr[i:i + 256]].float() - mu.view(1, -1, 1, 1)) ** 2).mean((0, 2, 3))
                           for i in range(0, len(tr), 256)]).mean(0)
        self.mu = mu.view(1, -1, 1, 1); self.sd = var.sqrt().view(1, -1, 1, 1) + 1e-5
        for i in range(0, len(fm), 256):                     # chuẩn hoá từng khối: tránh bản sao fp32 3.4 GB
            fm[i:i + 256] = ((fm[i:i + 256].float() - self.mu) / self.sd).half()
        self.fm = fm
        self.tr = np.where(self.split == 0)[0]; self.te = np.where(self.split == 1)[0]
        self.start_mode = 'mix'


def state(d: Data, idx: torch.Tensor, box: torch.Tensor, t: torch.Tensor,
          last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = len(idx); w = (box[:, 2] - box[:, 0]).clamp_min(2 * CELL); h = (box[:, 3] - box[:, 1]).clamp_min(2 * CELL)
    cx = (box[:, 0] + box[:, 2]) / 2; cy = (box[:, 1] + box[:, 3]) / 2
    ex = torch.stack([cx - EXP * w / 2, cy - EXP * h / 2, cx + EXP * w / 2, cy + EXP * h / 2], 1)
    rois = torch.cat([torch.arange(b, device=DEV, dtype=torch.float32)[:, None], ex], 1)
    feat = roi_align(d.fm[idx].float(), rois, (VIEW, VIEW), spatial_scale=G / W, sampling_ratio=2, aligned=True)
    u = (torch.arange(VIEW, device=DEV, dtype=torch.float32) + 0.5) / VIEW
    xs = ex[:, 0:1] + u * (ex[:, 2:3] - ex[:, 0:1]); ys = ex[:, 1:2] + u * (ex[:, 3:4] - ex[:, 1:2])
    mx = (xs >= box[:, 0:1]) & (xs <= box[:, 2:3]); my = (ys >= box[:, 1:2]) & (ys <= box[:, 3:4])
    mask = (my[:, :, None] & mx[:, None, :]).float().unsqueeze(1)
    oh = F.one_hot(last.clamp_min(0), NA).float() * (last >= 0).float()[:, None]
    sc = torch.cat([box / W, torch.stack([w / W, h / W, t.float() / T, d.score[idx]], 1), oh], 1)
    return torch.cat([feat, mask], 1), sc


class QNet(nn.Module):
    """Dueling Q (hoặc logit hành động cho bản bắt chước)."""

    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(257, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * 25 + NSC, 256), nn.ReLU())
        self.v = nn.Linear(256, 1); self.a = nn.Linear(256, NA)

    def forward(self, x: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([self.c(x).flatten(1), sc], 1)); a = self.a(h)
        return self.v(h) + a - a.mean(1, keepdim=True)


def oracle_action(d: Data, idx: torch.Tensor, box: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """Hành động hợp lệ làm IoU tăng nhiều nhất (dùng GT); STOP nếu không tăng quá 0.005."""
    cand = torch.stack([apply(box, torch.full((len(idx),), a, device=DEV, dtype=torch.long)) for a in range(12)], 1)
    gain = iou_t(cand, d.gt[idx][:, None]) - iou_t(box, d.gt[idx])[:, None]
    gain = gain.masked_fill(~valid(last)[:, :12], -1e9)
    best = gain.argmax(1)
    return torch.where(gain.max(1).values > 0.005, best, torch.full_like(best, STOP))


def reset(d: Data, n: int, rng: np.random.RandomState) -> tuple[torch.Tensor, torch.Tensor]:
    """n môi trường mới trên frame train; phân phối điểm xuất phát theo d.start_mode (xem docstring module)."""
    idx = torch.from_numpy(rng.choice(d.tr, n)).to(DEV); g = d.gt[idx]
    wh = torch.stack([g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]] * 2, 1)
    mag = torch.full((n, 1), 0.3, device=DEV) if d.start_mode == 'mix' else torch.rand(n, 1, device=DEV) * 0.35
    jit = (g + (torch.rand(n, 4, device=DEV) * 2 - 1) * mag * wh).clamp(0, W)
    jit = torch.stack([torch.minimum(jit[:, 0], jit[:, 2] - 2 * CELL), torch.minimum(jit[:, 1], jit[:, 3] - 2 * CELL),
                       jit[:, 2], jit[:, 3]], 1).clamp(0, W)
    use_det = torch.rand(n, device=DEV) < (0.5 if d.start_mode == 'mix' else 1 / 3)
    return idx, torch.where(use_det[:, None], d.start[idx], jit)


def frame_paths() -> list[str]:
    """Đường dẫn ảnh của từng frame refine, cùng thứ tự với vg_refine_prep.py."""
    frames = vp.all_frames(); _, fold = vp.video_split(frames)
    hasgt = h5py.File('data/vg_frames.h5', 'r')['hasgt'][...]; out: list[str] = []; i = 0
    for v in sorted(frames, key=lambda v: (fold[v], v)):
        for _, p, _, _, _ in frames[v]:
            if hasgt[i]:
                out.append(p)
            i += 1
    return out


@torch.no_grad()
def trajectory(d: Data, net: QNet | None, ix: np.ndarray, oracle: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Chạy tham lam trên frame ix từ box detector. Trả (box sau mỗi bước (T+1,B,4), số bước đã đi (B,))."""
    idx = torch.from_numpy(ix).to(DEV); box = d.start[idx].clone(); t = torch.zeros(len(ix), dtype=torch.long, device=DEV)
    alive = torch.ones(len(ix), dtype=torch.bool, device=DEV); traj = [box.clone()]; last = torch.full_like(t, -1)
    for _ in range(T):
        if oracle:
            a = oracle_action(d, idx, box, last)
        else:
            x, sc = state(d, idx, box, t, last); a = masked_argmax(net(x, sc), last)
        a = torch.where(alive, a, torch.full_like(a, STOP)); alive = alive & (a != STOP)
        box = apply(box, a); t = t + alive.long(); last = a; traj.append(box.clone())
    return torch.stack(traj), t


def run_policy(d: Data, net: QNet | None, ix: np.ndarray, oracle: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Trả (box cuối, số bước)."""
    traj, t = trajectory(d, net, ix, oracle); return traj[-1], t


def evaluate(d: Data, ix: np.ndarray, box: torch.Tensor, steps: torch.Tensor) -> dict[str, float]:
    idx = torch.from_numpy(ix).to(DEV); i0 = iou_t(d.start[idx], d.gt[idx]).cpu().numpy()
    i1 = iou_t(box, d.gt[idx]).cpu().numpy(); vids = d.vid[ix]
    return {'iou_before': float(i0.mean()), 'iou_after': float(i1.mean()), 'dr_before': float((i0 >= 0.5).mean()),
            'dr_after': float((i1 >= 0.5).mean()),
            'macro_before': float(np.mean([i0[vids == v].mean() for v in np.unique(vids)])),
            'macro_after': float(np.mean([i1[vids == v].mean() for v in np.unique(vids)])),
            'improved': float((i1 > i0 + 0.01).mean()), 'worse': float((i1 < i0 - 0.01).mean()),
            'steps': float(steps.float().mean())}


def train_dqn(d: Data, iters: int, hook: Callable[[int, QNet, str], None], seed: int = 0) -> QNet:
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = QNet().to(DEV); tgt = QNet().to(DEV); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-4); cap = 200000; n = 0; ptr = 0
    buf = {'idx': torch.zeros(cap, dtype=torch.long, device=DEV), 'box': torch.zeros(cap, 4, device=DEV),
           't': torch.zeros(cap, dtype=torch.long, device=DEV), 'a': torch.zeros(cap, dtype=torch.long, device=DEV),
           'r': torch.zeros(cap, device=DEV), 'nbox': torch.zeros(cap, 4, device=DEV),
           'nt': torch.zeros(cap, dtype=torch.long, device=DEV), 'd': torch.zeros(cap, device=DEV),
           'last': torch.zeros(cap, dtype=torch.long, device=DEV), 'nlast': torch.zeros(cap, dtype=torch.long, device=DEV)}
    idx, box = reset(d, NENV, rng); t = torch.zeros(NENV, dtype=torch.long, device=DEV); ret_hist = []
    last = torch.full_like(t, -1)
    ep_ret = torch.zeros(NENV, device=DEV); t0 = time.time(); hook(0, net.eval(), 'chưa học'); net.train()
    for it in range(1, iters + 1):
        eps = max(0.05, 1.0 - it / (0.4 * iters))
        with torch.no_grad():
            x, sc = state(d, idx, box, t, last); a = masked_argmax(net(x, sc), last)
        rnd = torch.rand(NENV, device=DEV) < eps
        a = torch.where(rnd, masked_argmax(torch.rand(NENV, NA, device=DEV), last), a)
        nbox = apply(box, a); nt = t + 1
        r = torch.where(a == STOP, torch.zeros(NENV, device=DEV),
                        R_SCALE * (iou_t(nbox, d.gt[idx]) - iou_t(box, d.gt[idx])) - STEP_COST)
        done = (a == STOP) | (nt >= T)
        j = (torch.arange(NENV, device=DEV) + ptr) % cap
        for k, val in (('idx', idx), ('box', box), ('t', t), ('a', a), ('r', r), ('nbox', nbox), ('nt', nt),
                       ('d', done.float()), ('last', last), ('nlast', a)):
            buf[k][j] = val
        ptr = (ptr + NENV) % cap; n = min(n + NENV, cap); ep_ret += r
        if done.any():
            ret_hist += ep_ret[done].tolist(); ep_ret[done] = 0
            ni, nb = reset(d, int(done.sum()), rng); idx = idx.clone(); idx[done] = ni; nbox[done] = nb; nt[done] = 0
        last = torch.where(done, torch.full_like(a, -1), a); box, t = nbox, nt
        if n >= 5000:
            for _ in range(2):
                s = torch.randint(0, n, (128,), device=DEV)
                x, sc = state(d, buf['idx'][s], buf['box'][s], buf['t'][s], buf['last'][s])
                nx, nsc = state(d, buf['idx'][s], buf['nbox'][s], buf['nt'][s], buf['nlast'][s])
                with torch.no_grad():
                    a2 = masked_argmax(net(nx, nsc), buf['nlast'][s]); qn = tgt(nx, nsc).gather(1, a2[:, None]).squeeze(1)
                    y = buf['r'][s] + GAMMA * (1 - buf['d'][s]) * qn
                q = net(x, sc).gather(1, buf['a'][s][:, None]).squeeze(1)
                loss = F.smooth_l1_loss(q, y); opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if it % 1000 == 0:
            tgt.load_state_dict(net.state_dict())
        if it % 2000 == 0:
            hook(it, net.eval(), 'eps %.2f | return TB %.3f | %.0fs' % (
                eps, np.mean(ret_hist[-2000:]) if ret_hist else 0, time.time() - t0)); net.train()
    return net.eval()


def train_imitation(d: Data, iters: int, hook: Callable[[int, QNet, str], None], seed: int = 0) -> QNet:
    """Bắt chước oracle (giám sát): rollout oracle + 30% hành động ngẫu nhiên, nhãn = hành động oracle."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = QNet().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    idx, box = reset(d, NENV, rng); t = torch.zeros(NENV, dtype=torch.long, device=DEV); last = torch.full_like(t, -1)
    hook(0, net.eval(), 'chưa học'); net.train()
    for it in range(1, iters + 1):
        lab = oracle_action(d, idx, box, last); x, sc = state(d, idx, box, t, last)
        loss = F.cross_entropy(net(x, sc), lab); opt.zero_grad(); loss.backward(); opt.step()
        a = torch.where(torch.rand(NENV, device=DEV) < 0.3, masked_argmax(torch.rand(NENV, NA, device=DEV), last), lab)
        nbox = apply(box, a); nt = t + 1; done = (a == STOP) | (nt >= T)
        if done.any():
            ni, nb = reset(d, int(done.sum()), rng); idx = idx.clone(); idx[done] = ni; nbox[done] = nb; nt[done] = 0
        last = torch.where(done, torch.full_like(a, -1), a); box, t = nbox, nt
        if it % 2000 == 0:
            hook(it, net.eval(), 'loss %.3f' % float(loss)); net.train()
    return net.eval()


@torch.no_grad()
def latency(d: Data, net: QNet, ix: np.ndarray) -> float:
    """ms / frame khi refine từng frame một (batch 1), gồm dựng state + mạng, không gồm backbone."""
    z = torch.zeros(1, dtype=torch.long, device=DEV)
    net(*state(d, torch.from_numpy(ix[:1]).to(DEV), d.start[ix[:1]], z, z - 1))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in ix:
        run_policy(d, net, np.array([i]))
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t0) / len(ix)


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    mode = sys.argv[2] if len(sys.argv) > 2 else 'mix'; tag = '' if mode == 'mix' else '_' + mode
    diag = sys.argv[3] if len(sys.argv) > 3 else ''; kd = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    d = Data(); d.start_mode = mode
    if diag == 'holdout':      # chẩn đoán: train trên 3 fold video, 'test' = fold video thứ kd (cùng chế độ OOF)
        tr_all = d.tr; d.tr = tr_all[d.fold[tr_all] != kd]; d.te = tr_all[d.fold[tr_all] == kd]; tag += '_hold%d' % kd
    elif diag == 'testfeat':   # chẩn đoán: đặc trưng test tính bằng backbone OOF fold kd (box xuất phát giữ nguyên)
        z = h5py.File('data/vg_refine_fm_test_oof%d.h5' % kd, 'r'); pos = {int(r): i for i, r in enumerate(d.row)}
        ti = torch.tensor([pos[int(r)] for r in z['row'][...]], device=DEV)
        d.fm[ti] = ((torch.from_numpy(z['fm'][...]).to(DEV).float() - d.mu) / d.sd).half(); tag += '_testfeat%d' % kd
    print('frame train %d | test %d | lưới %dx%d (ô %.0f px) | %d hành động | T=%d' % (
        len(d.tr), len(d.te), G, G, CELL, NA, T), flush=True)
    res: dict = {'grid': G, 'cell_px': CELL, 'T': T}
    for name, ix, in (('test', d.te), ('train_oof', d.tr)):
        res['oracle_' + name] = evaluate(d, ix, *run_policy(d, None, ix, oracle=True))
    print('ORACLE tham lam (dùng GT): TEST IoU %.3f -> %.3f | DR %.1f%% -> %.1f%%' % (
        res['oracle_test']['iou_before'], res['oracle_test']['iou_after'], 100 * res['oracle_test']['dr_before'],
        100 * res['oracle_test']['dr_after']), flush=True)
    # 6 frame test cố định để theo dõi: mỗi video 1 frame có IoU detector gần 0.45 nhất (box lệch vừa phải, còn chỗ
    # để sửa), thêm 1 frame IoU gần 0.3 từ video nhiều frame nhất
    i0 = iou_t(d.start, d.gt).cpu().numpy(); vis_ix: list[int] = []
    for v in np.unique(d.vid[d.te]):
        cand = d.te[d.vid[d.te] == v]; vis_ix.append(int(cand[np.argmin(np.abs(i0[cand] - 0.45))]))
    big = d.te[d.vid[d.te] == max(np.unique(d.vid[d.te]), key=lambda v: (d.vid[d.te] == v).sum())]
    big = big[~np.isin(big, vis_ix)]; vis_ix.append(int(big[np.argmin(np.abs(i0[big] - 0.3))]))
    vis = np.array(vis_ix); paths = frame_paths(); imgs = rv.load_images([paths[i] for i in vis])
    ref = {'detector': res['oracle_test']['iou_before'], 'oracle': res['oracle_test']['iou_after']}
    logs: dict[str, dict] = {}

    def make_hook(name: str) -> Callable[[int, QNet, str], None]:
        lg: dict[str, list] = {'its': [], 'iou_test': [], 'traj': [], 'steps': []}; logs[name] = lg

        def hook(it: int, net: QNet, msg: str) -> None:
            m = evaluate(d, d.te, *run_policy(d, net, d.te)); tr, st = trajectory(d, net, vis)
            lg['its'].append(it); lg['iou_test'].append(m['iou_after'])
            lg['traj'].append(tr.cpu().numpy()); lg['steps'].append(st.cpu().numpy())
            print('%-9s vòng %6d | %s | TEST IoU %.3f -> %.3f | DR %.1f%% -> %.1f%% | tốt lên %.1f%% tệ đi %.1f%% | bước %.1f'
                  % (name, it, msg, m['iou_before'], m['iou_after'], 100 * m['dr_before'], 100 * m['dr_after'],
                     100 * m['improved'], 100 * m['worse'], m['steps']), flush=True)
            snap = {'its': np.array(lg['its']), 'iou_test': np.array(lg['iou_test']), 'traj': np.stack(lg['traj']),
                    'steps': np.stack(lg['steps']), 'gt': d.gt[torch.from_numpy(vis).to(DEV)].cpu().numpy(),
                    'paths': np.array([paths[i] for i in vis]), 'ref_det': ref['detector'], 'ref_oracle': ref['oracle']}
            np.savez('data/vg_refine_vis_%s%s.npz' % (name, tag), **snap)
            rv.render_progress(snap, imgs, 'fig_refine_progress_%s%s.png' % (name, tag),
                               'Quá trình học %s — box qua từng bước trên 6 frame test' % name)
            rv.render_curve({k: {'its': x['its'], 'iou_test': x['iou_test']} for k, x in logs.items() if x['its']},
                            ref, 'fig_refine_curve%s.png' % tag)
            if it == iters:
                rv.render_steps(snap, imgs, 'fig_refine_steps_%s%s.png' % (name, tag),
                                'Từng bước dịch chuyển box — %s, model cuối' % name)
        return hook

    for name, fn in (('dqn', train_dqn), ('imitation', train_imitation)):
        net = fn(d, iters, make_hook(name))
        for split, ix in (('test', d.te), ('train_oof', d.tr)):
            res['%s_%s' % (name, split)] = evaluate(d, ix, *run_policy(d, net, ix))
        res['%s_ms_frame' % name] = latency(d, net, d.te[:300])
        torch.save(net.state_dict(), 'data/vg_refine_%s%s.pth' % (name, tag))
    print('\n=== BOX REFINEMENT TRÊN LƯỚI %dx%d — 1.068 frame bệnh, 5 video test ===' % (G, G))
    print('%-22s %-17s %-15s %-15s %8s %8s %6s %8s' % ('phương pháp', 'IoU (trước->sau)', 'DR@0.5', 'TB theo video',
                                                       'tốt lên', 'tệ đi', 'bước', 'ms/frame'))
    for name in ('dqn', 'imitation', 'oracle'):
        m = res['%s_test' % name]; ms = res.get('%s_ms_frame' % name, float('nan'))
        print('%-22s %.3f -> %.3f     %.1f%% -> %.1f%%   %.3f -> %.3f   %6.1f%% %6.1f%% %6.1f %8.2f' % (
            name, m['iou_before'], m['iou_after'], 100 * m['dr_before'], 100 * m['dr_after'], m['macro_before'],
            m['macro_after'], 100 * m['improved'], 100 * m['worse'], m['steps'], ms))
    for name in ('dqn', 'imitation', 'oracle'):
        m = res['%s_train_oof' % name]
        print('TRAIN-OOF %-12s IoU %.3f -> %.3f | DR %.1f%% -> %.1f%%' % (
            name, m['iou_before'], m['iou_after'], 100 * m['dr_before'], 100 * m['dr_after']))
    json.dump(res, open('data/vg_refine%s.json' % tag, 'w'), indent=1)


if __name__ == '__main__':
    main()
