"""Sửa box detector khoanh LỆCH / TRƯỢT: mục tiêu là biến ca khoanh sai (IoU < 0.5) thành khoanh đúng vùng
(IoU >= 0.5) mà KHÔNG phá ca đã đúng. Không cần box trùng khít GT — IoU ~0.5 là đạt.

  - Reward chính = +2 ở cuối episode nếu box cuối có IoU >= 0.5 (0 nếu không). Box đã đúng -> STOP là tối ưu.
  - Reward phụ mỗi bước = ΔGIoU - 0.02: GIoU vẫn thay đổi khi box chưa chạm GT, nên đi lại gần GT vẫn được thưởng.
  - 23 hành động: 12 dịch 1 ô lưới (dời box / nới cạnh / thu cạnh) + 10 "NHẢY sang đề xuất #k" của detector + STOP.
    Cấm dịch đảo ngược bước trước; cấm NHẢY về đề xuất đã thăm trong episode (chặn nhảy qua lại).
  - State: vùng quanh box (10x10) + góc nhìn toàn ảnh (lưới 14x14 + vị trí box) + đặc trưng từng đề xuất.
  - Chế độ 'temporal' thêm tín hiệu THỜI GIAN từ ±3 frame lân cận (tối đa 6 frame):
      * bản đồ nhiệt 28x28 = đề xuất của các frame lân cận cộng dồn (trọng số score tương đối) -> thêm 1 kênh vào
        vùng quanh box và góc nhìn toàn ảnh: agent thấy các frame bên cạnh đang khoanh ở đâu ("bám box" qua frame);
      * độ ủng hộ theo thời gian của box HIỆN TẠI (tính lại mỗi bước) và của từng đề xuất #k.
  - Chọn điểm dừng train và ngưỡng tự tin δ (chỉ hành động khi max Q(hành động) - Q(STOP) > δ) trên FOLD VIDEO
    GIỮ LẠI (fold 0 của video train), không nhìn test. Train trên 3 fold còn lại. Không checkpoint nào có lợi ròng
    trên validation -> giữ nguyên box detector.
Metric chính: DR@0.5 trước/sau, số ca trượt được sửa, số ca đúng bị phá; phụ: IoU, DR@0.3.
Chạy: python vg_fixmiss.py [iterations=20000] [base|temporal]  -> data/vg_fixmiss_<mode>.json + fig_fixmiss_*_<mode>.png"""
from __future__ import annotations

import copy
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

import vg_refine as vr
import vg_refine_vis as rv

DEV = vr.DEV; W = vr.W; G = vr.G; CELL = vr.CELL; VIEW = vr.VIEW; EXP = vr.EXP
NMOVE = 12; NJUMP = 10; STOP = NMOVE + NJUMP; NA = STOP + 1; T = 10; HIT = 0.5
R_TERM = 2.0; R_SHAPE = 1.0; STEP_COST = 0.02; GAMMA = 0.95; NENV = 64; VAL_FOLD = 0
R_NB = 3; NB = 6
DELTAS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
INV = torch.tensor([1, 0, 3, 2, 8, 9, 10, 11, 4, 5, 6, 7], device=DEV)


def giou_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x0 = torch.maximum(a[..., 0], b[..., 0]); y0 = torch.maximum(a[..., 1], b[..., 1])
    x1 = torch.minimum(a[..., 2], b[..., 2]); y1 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    union = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    c = ((torch.maximum(a[..., 2], b[..., 2]) - torch.minimum(a[..., 0], b[..., 0])) *
         (torch.maximum(a[..., 3], b[..., 3]) - torch.minimum(a[..., 1], b[..., 1]))).clamp_min(1e-6)
    return inter / union.clamp_min(1e-6) - (c - union) / c


def cover28(boxes: torch.Tensor) -> torch.Tensor:
    """(...,4) -> (...,G,G) tỉ lệ diện tích mỗi ô lưới 28x28 bị box phủ."""
    e = torch.arange(G + 1, device=boxes.device, dtype=torch.float32) * CELL
    ix = (torch.minimum(boxes[..., 2:3], e[1:]) - torch.maximum(boxes[..., 0:1], e[:-1])).clamp_min(0) / CELL
    iy = (torch.minimum(boxes[..., 3:4], e[1:]) - torch.maximum(boxes[..., 1:2], e[:-1])).clamp_min(0) / CELL
    return iy[..., :, None] * ix[..., None, :]


class FixData(vr.Data):
    """Top-10 đề xuất + score + đặc trưng ROI, lưới toàn ảnh 14x14; (temporal) frame lân cận, bản đồ nhiệt, độ ủng hộ."""

    def __init__(self, temporal: bool) -> None:
        super().__init__(); self.temporal = temporal
        h = h5py.File('data/vg_frames.h5', 'r')
        self.props = torch.from_numpy(h['boxes'][...][self.row]).to(DEV)
        self.pscore = torch.from_numpy(h['scores'][...][self.row]).to(DEV)
        m = len(self.row); fm14, pf = [], []
        for i in range(0, m, 256):
            fmb = self.fm[i:i + 256].float(); pb = self.props[i:i + 256]; b = len(pb)
            fm14.append(F.avg_pool2d(fmb, 2).half())
            rois = torch.cat([torch.arange(b, device=DEV).repeat_interleave(NJUMP)[:, None].float(), pb.reshape(-1, 4)], 1)
            pf.append(roi_align(fmb, rois, (2, 2), spatial_scale=G / W, sampling_ratio=2, aligned=True)
                      .mean((2, 3)).view(b, NJUMP, -1).half())
        self.fm14 = torch.cat(fm14); self.pfeat = torch.cat(pf)
        if temporal:
            self._temporal(h)
        tr_all = self.tr; self.va = tr_all[self.fold[tr_all] == VAL_FOLD]; self.tr = tr_all[self.fold[tr_all] != VAL_FOLD]

    def _temporal(self, h: h5py.File) -> None:
        vid = h['vid'][...].astype(str); fn = h['fn'][...]; b = h['boxes'][...]; s = h['scores'][...]
        srel = s / np.maximum(s[:, :1], 1e-6); start: dict[str, int] = {}; end: dict[str, int] = {}
        for i, v in enumerate(vid):
            start.setdefault(v, i); end[v] = i + 1
        m = len(self.row); nbp = np.zeros((m, NB, NJUMP, 4), np.float32); nbs = np.zeros((m, NB, NJUMP), np.float32)
        nbm = np.zeros((m, NB), np.float32)
        for r, i in enumerate(self.row):
            j = np.arange(max(start[vid[i]], i - 3 * R_NB), min(end[vid[i]], i + 3 * R_NB + 1))
            j = j[(np.abs(fn[j] - fn[i]) <= R_NB) & (j != i)]; j = j[np.argsort(np.abs(fn[j] - fn[i]), kind='stable')][:NB]
            nbp[r, :len(j)] = b[j]; nbs[r, :len(j)] = srel[j]; nbm[r, :len(j)] = 1
        self.nbp = torch.from_numpy(nbp).to(DEV); self.nbs = torch.from_numpy(nbs).to(DEV); self.nbm = torch.from_numpy(nbm).to(DEV)
        heat = []
        for i in range(0, m, 128):
            c = cover28(self.nbp[i:i + 128]) * (self.nbs[i:i + 128] * self.nbm[i:i + 128, :, None])[..., None, None]
            hh = c.sum((1, 2)) / self.nbm[i:i + 128].sum(1).clamp_min(1)[:, None, None]
            heat.append((hh / (hh.amax((1, 2), keepdim=True) + 1e-6)).half())
        self.heat = torch.cat(heat)[:, None]; self.heat14 = F.avg_pool2d(self.heat.float(), 2).half()
        self.tsup = torch.cat([tsupport(self, torch.arange(i, min(m, i + 512), device=DEV), self.props[i:i + 512])
                               for i in range(0, m, 512)])
        self.nbfrac = self.nbm.sum(1) / NB


def tsupport(d: FixData, idx: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """(B,K,4) -> (B,K): độ ủng hộ theo thời gian = TB trên frame lân cận của score tương đối lớn nhất trong các đề
    xuất lân cận chồng lên box (IoU > 0.3)."""
    ov = vr.iou_t(boxes[:, :, None, None, :], d.nbp[idx][:, None]) > 0.3                     # (B,K,NB,KP)
    sup = (ov * d.nbs[idx][:, None]).amax(-1)                                               # (B,K,NB)
    return (sup * d.nbm[idx][:, None]).sum(-1) / d.nbm[idx].sum(-1).clamp_min(1)[:, None]


def valid(last: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """(B,NA): cấm dịch đảo ngược bước dịch trước; cấm nhảy về đề xuất đã thăm; STOP luôn hợp lệ."""
    m = torch.ones(len(last), NA, dtype=torch.bool, device=DEV); mv = (last >= 0) & (last < NMOVE)
    m[mv.nonzero().squeeze(1), INV[last[mv]]] = False
    m[:, NMOVE:STOP] &= ~vis
    return m


def step_box(d: FixData, idx: torch.Tensor, box: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    nb = box.clone(); mv = a < NMOVE; jm = (a >= NMOVE) & (a < STOP)
    if mv.any():
        nb[mv] = vr.apply(box[mv], a[mv])
    if jm.any():
        nb[jm] = d.props[idx[jm], a[jm] - NMOVE]
    return nb


def mark(vis: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    nv = vis.clone(); jm = (a >= NMOVE) & (a < STOP)
    nv[jm.nonzero().squeeze(1), a[jm] - NMOVE] = True
    return nv


def state(d: FixData, idx: torch.Tensor, box: torch.Tensor, t: torch.Tensor,
          last: torch.Tensor) -> tuple[torch.Tensor, ...]:
    b = len(idx); w = (box[:, 2] - box[:, 0]).clamp_min(2 * CELL); h = (box[:, 3] - box[:, 1]).clamp_min(2 * CELL)
    cx = (box[:, 0] + box[:, 2]) / 2; cy = (box[:, 1] + box[:, 3]) / 2
    ex = torch.stack([cx - EXP * w / 2, cy - EXP * h / 2, cx + EXP * w / 2, cy + EXP * h / 2], 1)
    rois = torch.cat([torch.arange(b, device=DEV, dtype=torch.float32)[:, None], ex], 1)
    loc = roi_align(d.fm[idx].float(), rois, (VIEW, VIEW), spatial_scale=G / W, sampling_ratio=2, aligned=True)
    u = (torch.arange(VIEW, device=DEV, dtype=torch.float32) + 0.5) / VIEW
    xs = ex[:, 0:1] + u * (ex[:, 2:3] - ex[:, 0:1]); ys = ex[:, 1:2] + u * (ex[:, 3:4] - ex[:, 1:2])
    lm = ((ys >= box[:, 1:2]) & (ys <= box[:, 3:4]))[:, :, None] & ((xs >= box[:, 0:1]) & (xs <= box[:, 2:3]))[:, None, :]
    g = (torch.arange(14, device=DEV, dtype=torch.float32) + 0.5) * W / 14
    gm = ((g >= box[:, 1:2]) & (g <= box[:, 3:4]))[:, :, None] & ((g >= box[:, 0:1]) & (g <= box[:, 2:3]))[:, None, :]
    xl = [loc, lm.float().unsqueeze(1)]; xg = [d.fm14[idx].float(), gm.float().unsqueeze(1)]
    oh = F.one_hot(last.clamp_min(0), NA).float() * (last >= 0).float()[:, None]
    sc = [box / W, torch.stack([w / W, h / W, t.float() / T, d.pscore[idx, 0]], 1), oh]
    pb = d.props[idx]; pc = torch.stack([(pb[..., 0] + pb[..., 2]) / 2, (pb[..., 1] + pb[..., 3]) / 2], -1)
    dist = torch.hypot(pc[..., 0] - cx[:, None], pc[..., 1] - cy[:, None]) / W
    pf = [d.pscore[idx][..., None], (torch.arange(NJUMP, device=DEV).float() / NJUMP).expand(b, -1)[..., None],
          vr.iou_t(pb, box[:, None])[..., None], pb / W, dist[..., None], d.pfeat[idx].float()]
    if d.temporal:
        xl.append(roi_align(d.heat[idx].float(), rois, (VIEW, VIEW), spatial_scale=G / W, sampling_ratio=2, aligned=True))
        xg.append(d.heat14[idx].float())
        cur = tsupport(d, idx, box[:, None])[:, 0]
        sc.append(torch.stack([cur, d.nbfrac[idx]], 1))
        ts = d.tsup[idx]; pf += [ts[..., None], (ts - cur[:, None])[..., None]]
    return torch.cat(xl, 1), torch.cat(xg, 1), torch.cat(sc, 1), torch.cat(pf, -1)


class FixNet(nn.Module):
    """Dueling Q cho 23 hành động: 12 dịch + STOP từ trạng thái; 10 nhảy chấm theo đặc trưng từng đề xuất."""

    def __init__(self, temporal: bool) -> None:
        super().__init__(); ex = int(temporal)
        self.loc = nn.Sequential(nn.Conv2d(257 + ex, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        self.glob = nn.Sequential(nn.Conv2d(257 + ex, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(64 * 25 + 32 * 16 + 8 + NA + 2 * ex, 256), nn.ReLU())
        self.prop = nn.Sequential(nn.Linear(8 + 256 + 2 * ex, 64), nn.ReLU())
        self.v = nn.Linear(256, 1); self.ams = nn.Linear(256, NMOVE + 1)
        self.aj = nn.Sequential(nn.Linear(256 + 64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xl: torch.Tensor, xg: torch.Tensor, sc: torch.Tensor, pf: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([self.loc(xl).flatten(1), self.glob(xg).flatten(1), sc], 1))
        ams = self.ams(h); aj = self.aj(torch.cat([h[:, None].expand(-1, NJUMP, -1), self.prop(pf)], -1)).squeeze(-1)
        a = torch.cat([ams[:, :NMOVE], aj, ams[:, NMOVE:]], 1)
        return self.v(h) + a - a.mean(1, keepdim=True)


def oracle_action(d: FixData, idx: torch.Tensor, box: torch.Tensor, last: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Hành động HỢP LỆ làm GIoU tăng nhiều nhất (dùng GT); STOP nếu không tăng quá 0.005."""
    g0 = giou_t(box, d.gt[idx]); gains = []
    for a in range(STOP):
        gains.append(giou_t(step_box(d, idx, box, torch.full((len(idx),), a, device=DEV, dtype=torch.long)), d.gt[idx]) - g0)
    gain = torch.stack(gains, 1).masked_fill(~valid(last, vis)[:, :STOP], -1e9); best = gain.argmax(1)
    return torch.where(gain.max(1).values > 0.005, best, torch.full_like(best, STOP))


@torch.no_grad()
def rollout(d: FixData, net: FixNet | None, ix: np.ndarray, delta: float = 0.0,
            oracle: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tham lam từ box top-1 detector. δ: chỉ hành động nếu max Q(không STOP) - Q(STOP) > δ.
    Trả (box mỗi bước (T+1,B,4), số bước, có dùng nhảy hay không)."""
    idx = torch.from_numpy(ix).to(DEV); box = d.props[idx, 0].clone(); t = torch.zeros(len(ix), dtype=torch.long, device=DEV)
    last = torch.full_like(t, -1); alive = torch.ones(len(ix), dtype=torch.bool, device=DEV)
    vis = torch.zeros(len(ix), NJUMP, dtype=torch.bool, device=DEV); vis[:, 0] = True
    jumped = torch.zeros(len(ix), dtype=torch.bool, device=DEV); traj = [box.clone()]
    for _ in range(T):
        if oracle:
            a = oracle_action(d, idx, box, last, vis)
        else:
            q = net(*state(d, idx, box, t, last)).masked_fill(~valid(last, vis), -1e9)
            best = q[:, :STOP].argmax(1); margin = q[:, :STOP].max(1).values - q[:, STOP]
            a = torch.where(margin > delta, best, torch.full_like(best, STOP))
        a = torch.where(alive, a, torch.full_like(a, STOP)); alive = alive & (a != STOP)
        jumped = jumped | (alive & (a >= NMOVE)); vis = mark(vis, a)
        box = step_box(d, idx, box, a); t = t + alive.long(); last = torch.where(alive, a, last); traj.append(box.clone())
        if not alive.any():
            break
    while len(traj) < T + 1:
        traj.append(box.clone())
    return torch.stack(traj), t, jumped


def metrics(d: FixData, ix: np.ndarray, box: torch.Tensor, steps: torch.Tensor, jumped: torch.Tensor) -> dict[str, float]:
    idx = torch.from_numpy(ix).to(DEV); g = d.gt[idx]
    i0 = vr.iou_t(d.props[idx, 0], g).cpu().numpy(); i1 = vr.iou_t(box, g).cpu().numpy()
    h0 = i0 >= HIT; h1 = i1 >= HIT; miss0 = i0 == 0
    return {'n': int(len(ix)), 'dr_before': float(h0.mean()), 'dr_after': float(h1.mean()),
            'fixed': int((~h0 & h1).sum()), 'broken': int((h0 & ~h1).sum()), 'net': int((~h0 & h1).sum() - (h0 & ~h1).sum()),
            'n_miss': int((~h0).sum()), 'n_miss0': int(miss0.sum()), 'fixed_miss0': int((miss0 & h1).sum()),
            'iou_before': float(i0.mean()), 'iou_after': float(i1.mean()),
            'dr03_before': float((i0 >= 0.3).mean()), 'dr03_after': float((i1 >= 0.3).mean()),
            'steps': float(steps.float().mean()), 'jump_rate': float(jumped.float().mean())}


def reset(d: FixData, n: int, rng: np.random.RandomState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frame train; xuất phát: 50% box top-1 detector OOF, 25% GT lệch nhẹ (0–20%, phần lớn là ca đúng),
    25% một đề xuất ngẫu nhiên #2–#10 (phần lớn là ca trượt xa). Trả (frame, box, đề xuất đã thăm)."""
    idx = torch.from_numpy(rng.choice(d.tr, n)).to(DEV); g = d.gt[idx]; u = torch.rand(n, device=DEV)
    wh = torch.stack([g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]] * 2, 1)
    jit = (g + (torch.rand(n, 4, device=DEV) * 2 - 1) * torch.rand(n, 1, device=DEV) * 0.2 * wh).clamp(0, W)
    jit = torch.stack([torch.minimum(jit[:, 0], jit[:, 2] - 2 * CELL), torch.minimum(jit[:, 1], jit[:, 3] - 2 * CELL),
                       jit[:, 2], jit[:, 3]], 1).clamp(0, W)
    k = torch.randint(1, NJUMP, (n,), device=DEV); top = u < 0.5; rnd = u >= 0.75
    box = torch.where(top[:, None], d.props[idx, 0], torch.where(rnd[:, None], d.props[idx, k], jit))
    vis = torch.zeros(n, NJUMP, dtype=torch.bool, device=DEV); vis[top, 0] = True
    vis[rnd.nonzero().squeeze(1), k[rnd]] = True
    return idx, box, vis


def env_step(d: FixData, idx: torch.Tensor, box: torch.Tensor, t: torch.Tensor,
             a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nb = step_box(d, idx, box, a); g = d.gt[idx]; nt = t + 1
    r = torch.where(a == STOP, torch.zeros_like(nt, dtype=torch.float32),
                    R_SHAPE * (giou_t(nb, g) - giou_t(box, g)) - STEP_COST)
    done = (a == STOP) | (nt >= T)
    r = r + done.float() * R_TERM * (vr.iou_t(nb, g) >= HIT).float()
    return nb, r, done


def train(d: FixData, iters: int, kind: str, hook: Callable[[int, FixNet, str], None], seed: int = 0) -> None:
    """kind='dqn': Double + Dueling DQN; kind='imitation': cross-entropy tới hành động oracle."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = FixNet(d.temporal).to(DEV); tgt = copy.deepcopy(net); opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    cap = 200000; n = 0; ptr = 0
    buf = {k: torch.zeros(cap, *s, dtype=dt, device=DEV) for k, s, dt in (
        ('idx', (), torch.long), ('box', (4,), torch.float32), ('t', (), torch.long), ('last', (), torch.long),
        ('vis', (NJUMP,), torch.bool), ('a', (), torch.long), ('r', (), torch.float32), ('nbox', (4,), torch.float32),
        ('nt', (), torch.long), ('nvis', (NJUMP,), torch.bool), ('d', (), torch.float32))}
    idx, box, vis = reset(d, NENV, rng); t = torch.zeros(NENV, dtype=torch.long, device=DEV); last = torch.full_like(t, -1)
    rets: list[float] = []; ep = torch.zeros(NENV, device=DEV); t0 = time.time(); hook(0, net.eval(), 'chưa học'); net.train()
    for it in range(1, iters + 1):
        ok = valid(last, vis); ra = torch.rand(NENV, NA, device=DEV).masked_fill(~ok, -1).argmax(1)
        if kind == 'dqn':
            eps = max(0.05, 1.0 - it / (0.4 * iters))
            with torch.no_grad():
                a = net(*state(d, idx, box, t, last)).masked_fill(~ok, -1e9).argmax(1)
            a = torch.where(torch.rand(NENV, device=DEV) < eps, ra, a)
        else:
            lab = oracle_action(d, idx, box, last, vis)
            loss = F.cross_entropy(net(*state(d, idx, box, t, last)).masked_fill(~ok, -1e9), lab)
            opt.zero_grad(); loss.backward(); opt.step()
            a = torch.where(torch.rand(NENV, device=DEV) < 0.3, ra, lab)
        nb, r, done = env_step(d, idx, box, t, a); nvis = mark(vis, a)
        if kind == 'dqn':
            j = (torch.arange(NENV, device=DEV) + ptr) % cap
            for k, val in (('idx', idx), ('box', box), ('t', t), ('last', last), ('vis', vis), ('a', a), ('r', r),
                           ('nbox', nb), ('nt', t + 1), ('nvis', nvis), ('d', done.float())):
                buf[k][j] = val
            ptr = (ptr + NENV) % cap; n = min(n + NENV, cap)
        ep += r; nt = t + 1
        if done.any():
            rets += ep[done].tolist(); ep[done] = 0
            ni, nbx, nv = reset(d, int(done.sum()), rng); idx = idx.clone(); idx[done] = ni; nb[done] = nbx
            nvis[done] = nv; nt[done] = 0
        last = torch.where(done, torch.full_like(a, -1), a); box, t, vis = nb, nt, nvis
        if kind == 'dqn' and n >= 5000:
            for _ in range(2):
                s = torch.randint(0, n, (128,), device=DEV); bi = buf['idx'][s]
                x = state(d, bi, buf['box'][s], buf['t'][s], buf['last'][s])
                nx = state(d, bi, buf['nbox'][s], buf['nt'][s], buf['a'][s])
                with torch.no_grad():
                    a2 = net(*nx).masked_fill(~valid(buf['a'][s], buf['nvis'][s]), -1e9).argmax(1)
                    y = buf['r'][s] + GAMMA * (1 - buf['d'][s]) * tgt(*nx).gather(1, a2[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(net(*x).gather(1, buf['a'][s][:, None]).squeeze(1), y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if kind == 'dqn' and it % 1000 == 0:
            tgt.load_state_dict(net.state_dict())
        if it % 2000 == 0:
            msg = ('eps %.2f | return TB %.2f' % (eps, np.mean(rets[-2000:]))) if kind == 'dqn' else 'loss %.3f' % float(loss)
            hook(it, net.eval(), msg + ' | %.0fs' % (time.time() - t0)); net.train()


def fig_curve(logs: dict, ref: dict, out: str, mode: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=140); fig.patch.set_facecolor('#fcfcfb')
    for ax, key, ttl in ((axes[0], 'val_dr', 'VALIDATION (fold video giữ lại) — DR@0.5 tốt nhất theo δ'),
                         (axes[1], 'test_dr', 'TEST — DR@0.5 (δ = 0, chỉ để theo dõi)')):
        for m, c, lab in (('dqn', '#2a78d6', 'DQN (RL)'), ('imitation', '#eb6834', 'Bắt chước oracle')):
            if m in logs and logs[m]['its']:
                ax.plot(np.array(logs[m]['its']) / 1000, 100 * np.array(logs[m][key]), color=c, lw=2, marker='o', ms=4, label=lab)
        r0 = ref['val_det'] if key == 'val_dr' else ref['test_det']; ro = ref['val_orc'] if key == 'val_dr' else ref['test_orc']
        ax.axhline(100 * r0, color='#898781', lw=1.2, ls=(0, (4, 3)))
        ax.text(0, 100 * r0 + 0.4, 'detector %.1f%%' % (100 * r0), fontsize=7, color='#898781')
        ax.axhline(100 * ro, color='#0b0b0b', lw=1.2, ls=(0, (1, 2)))
        ax.text(0, 100 * ro + 0.4, 'oracle %.1f%%' % (100 * ro), fontsize=7, color='#0b0b0b')
        ax.set_title(ttl + (' [%s]' % mode), fontsize=9, loc='left'); ax.set_xlabel('vòng train (nghìn)', fontsize=8)
        ax.set_ylabel('DR@0.5 (%)', fontsize=8); ax.grid(axis='y', color='#e1e0d9'); ax.legend(frameon=False, fontsize=7, loc='lower right')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, facecolor='#fcfcfb'); plt.close(fig)


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    mode = sys.argv[2] if len(sys.argv) > 2 else 'temporal'; tag = '_' + mode
    d = FixData(mode == 'temporal'); paths = vr.frame_paths()
    print('mode %s | frame train %d | validation (fold video %d) %d | test %d | %d hành động | T=%d' % (
        mode, len(d.tr), VAL_FOLD, len(d.va), len(d.te), NA, T), flush=True)
    res: dict = {'config': {'mode': mode, 'T': T, 'NA': NA, 'R_TERM': R_TERM, 'R_SHAPE': R_SHAPE, 'STEP_COST': STEP_COST,
                            'HIT': HIT, 'R_NB': R_NB, 'NB': NB}}
    for name, ix in (('val', d.va), ('test', d.te)):
        tr, st, jp = rollout(d, None, ix, oracle=True); res['oracle_' + name] = metrics(d, ix, tr[-1], st, jp)
    o = res['oracle_test']
    print('ORACLE (GIoU, dùng GT) TEST: DR %.1f%% -> %.1f%% | sửa %d / phá %d | ca IoU=0 sửa được %d/%d' % (
        100 * o['dr_before'], 100 * o['dr_after'], o['fixed'], o['broken'], o['fixed_miss0'], o['n_miss0']), flush=True)
    ref = {'val_det': res['oracle_val']['dr_before'], 'val_orc': res['oracle_val']['dr_after'],
           'test_det': o['dr_before'], 'test_orc': o['dr_after']}
    tei = torch.from_numpy(d.te).to(DEV); i0 = vr.iou_t(d.props[tei, 0], d.gt[tei]).cpu().numpy()
    pbest = vr.iou_t(d.props[tei], d.gt[tei][:, None]).max(1).values.cpu().numpy(); rs = np.random.RandomState(3)
    vis = np.concatenate([rs.choice(d.te[(i0 == 0) & (pbest >= HIT)], 2, replace=False),
                          rs.choice(d.te[(i0 > 0.1) & (i0 < HIT)], 2, replace=False), rs.choice(d.te[i0 >= 0.7], 2, replace=False)])
    imgs = rv.load_images([paths[i] for i in vis]); logs: dict[str, dict] = {}; best: dict[str, dict] = {}

    def make_hook(name: str) -> Callable[[int, FixNet, str], None]:
        lg: dict[str, list] = {k: [] for k in ('its', 'val_dr', 'val_delta', 'test_dr', 'iou_test', 'traj', 'steps',
                                               'test_fixed', 'test_broken', 'test_fixed_miss0')}
        logs[name] = lg; best[name] = {'net': -10 ** 9}

        def hook(it: int, net: FixNet, msg: str) -> None:
            vals = {}
            for dl in DELTAS:
                tr_, st_, jp_ = rollout(d, net, d.va, dl); vals[dl] = metrics(d, d.va, tr_[-1], st_, jp_)
            dl = max(DELTAS, key=lambda k: (vals[k]['net'], -k)); v = vals[dl]
            if v['net'] > best[name]['net']:
                best[name] = {'net': v['net'], 'it': it, 'delta': dl, 'state': copy.deepcopy(net.state_dict()), 'val': v}
            tr_, st_, jp_ = rollout(d, net, d.te, 0.0); m = metrics(d, d.te, tr_[-1], st_, jp_)
            vt, vs, _ = rollout(d, net, vis, 0.0)
            for k, x in (('its', it), ('val_dr', v['dr_after']), ('val_delta', dl), ('test_dr', m['dr_after']),
                         ('iou_test', m['dr_after']), ('traj', vt.cpu().numpy()), ('steps', vs.cpu().numpy()),
                         ('test_fixed', m['fixed']), ('test_broken', m['broken']), ('test_fixed_miss0', m['fixed_miss0'])):
                lg[k].append(x)
            print('%-9s vòng %6d | %s | VAL DR %.1f%%->%.1f%% (δ=%.2f, sửa %d phá %d) | TEST(δ=0) DR %.1f%%->%.1f%% sửa %d '
                  'phá %d (IoU=0 sửa %d/%d) | nhảy %.0f%% | bước %.1f' % (
                      name, it, msg, 100 * v['dr_before'], 100 * v['dr_after'], dl, v['fixed'], v['broken'],
                      100 * m['dr_before'], 100 * m['dr_after'], m['fixed'], m['broken'], m['fixed_miss0'], m['n_miss0'],
                      100 * m['jump_rate'], m['steps']), flush=True)
            snap = {'its': np.array(lg['its']), 'iou_test': np.array(lg['iou_test']), 'traj': np.stack(lg['traj']),
                    'steps': np.stack(lg['steps']), 'gt': d.gt[torch.from_numpy(vis).to(DEV)].cpu().numpy(),
                    'paths': np.array([paths[i] for i in vis])}
            np.savez('data/vg_fixmiss_vis_%s%s.npz' % (name, tag), **snap)
            rv.render_progress(snap, imgs, 'fig_fixmiss_progress_%s%s.png' % (name, tag),
                               'Sửa box khoanh lệch — %s [%s]: 2 ca trượt hẳn · 2 ca lệch · 2 ca đã đúng' % (name, mode),
                               metric='TEST DR@0.5')
            fig_curve(logs, ref, 'fig_fixmiss_curve%s.png' % tag, mode)
            if it == iters:
                rv.render_steps(snap, imgs, 'fig_fixmiss_steps_%s%s.png' % (name, tag), 'Từng bước — %s [%s], model cuối' % (
                    name, mode), frames=(0, 1, 2, 4))
        return hook

    for kind in ('dqn', 'imitation'):
        train(d, iters, kind, make_hook(kind))
        b = best[kind]; net = FixNet(d.temporal).to(DEV); net.load_state_dict(b['state']); net.eval()
        tr_, st_, jp_ = rollout(d, net, d.te, b['delta']); res[kind] = {'chosen_it': b['it'], 'chosen_delta': b['delta'],
                                                                        'val': b['val'], 'test': metrics(d, d.te, tr_[-1], st_, jp_)}
        torch.save(b['state'], 'data/vg_fixmiss_%s%s.pth' % (kind, tag))
    print('\n=== SỬA BOX KHOANH LỆCH [%s] (mục tiêu IoU >= 0.5) — 1.068 frame bệnh, 5 video test ===' % mode)
    print('%-24s %-18s %8s %8s %6s %-17s %-15s %7s %6s' % ('phương pháp', 'DR@0.5 trước->sau', 'sửa', 'phá', 'ròng',
                                                            'IoU=0 sửa được', 'IoU TB', 'nhảy', 'bước'))
    rows = [('oracle (GIoU, dùng GT)', res['oracle_test'])] + [('%s (vòng %d, δ=%.2f)' % (k, res[k]['chosen_it'],
                                                                                         res[k]['chosen_delta']), res[k]['test'])
                                                                for k in ('dqn', 'imitation')]
    for lab, m in rows:
        print('%-24s %5.1f%% -> %5.1f%%  %8d %8d %6d %6d/%-10d %.3f->%.3f %6.0f%% %6.1f' % (
            lab, 100 * m['dr_before'], 100 * m['dr_after'], m['fixed'], m['broken'], m['net'], m['fixed_miss0'], m['n_miss0'],
            m['iou_before'], m['iou_after'], 100 * m['jump_rate'], m['steps']))
    res['logs'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('traj', 'steps')} for k, v in logs.items()}
    json.dump(res, open('data/vg_fixmiss%s.json' % tag, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
