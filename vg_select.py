"""Video-grid — TẦNG THỜI GIAN: RL chọn SET frame trong video (attention "nhảy" giữa các frame, như jump giữa ô).
Mỗi bước agent chọn 1 frame chưa chọn; box output của frame = box của grid-attention re-ranker (vg_localize.py).
Reward tuỳ biến mỗi lần chọn:  IoU(box output, GT) + LAMBDA * [phủ đoạn bệnh MỚI] - 0.1 * [frame bình thường].
State mỗi frame ứng viên — HẠNG % TRONG VIDEO (bất biến với thang score của detector OOF vs detector chính):
q̂ (IoU dự đoán, OOF), score detector, đỉnh/mean attention, và bản làm mượt ±3 frame của q̂ và score;
+ ngữ cảnh set: khoảng cách tới frame đã chọn, frame đã chọn ở gần, cosine đặc trưng tối đa tới frame đã chọn,
% ngân sách đã dùng, ngân sách. Học REINFORCE + self-critical baseline (greedy rollout).
So với: random, det-conf top-k (pipeline cũ, box top-1), q̂ top-k, q̂ + NMS thời gian, oracle. Chia theo video.
Chạy: python vg_select.py <lambda> [episodes] [rerank|top1] [K]  -> in bảng + data/vg_select_l<lambda>_<box>[_k<K>].json
(K > 0: chọn CỐ ĐỊNH K frame mỗi video; bỏ trống: ngân sách 5/10/20% số frame)
(box: 'rerank' = box của re-ranker chọn theo OOF; 'top1' = box top-1 detector cho MỌI phương pháp)"""
from __future__ import annotations

import json
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from vg_localize import box_iou, vid_pct, vid_smooth

DEV = 'cuda'; BUDGETS = (0.05, 0.10, 0.20); SEG_GAP = 10; NEAR = 5; FP_PEN = 0.1; NF = 12


def n_pick(n: int, bud: float) -> int:
    """Số frame phải chọn: bud >= 1 là số frame cố định, bud < 1 là tỉ lệ số frame của video."""
    return min(n, int(bud)) if bud >= 1 else max(1, int(n * bud))


class Video:
    """Mảng theo frame (sắp theo frame_number) của 1 video."""

    def __init__(self, h: dict, ix: np.ndarray, loc: dict, box: str = 'rerank') -> None:
        self.n = len(ix); self.fn = h['fn'][ix]; isl = h['isp'][ix] > 0; self.isl = isl.astype(np.float32)
        ch = loc['choice'][ix]; bx = h['boxes'][ix, ch]
        self.iou = np.where(h['hasgt'][ix] > 0, box_iou(bx, h['gt'][ix]), 0.0).astype(np.float32)
        self.iou_top1 = np.where(h['hasgt'][ix] > 0, box_iou(h['boxes'][ix, 0], h['gt'][ix]), 0.0).astype(np.float32)
        if box == 'top1':
            self.iou = self.iou_top1
        self.q = loc['qhat'][ix].astype(np.float32); self.ds = h['scores'][ix, 0].astype(np.float32)
        att = loc['att'][ix].astype(np.float32); self.am = att.max((1, 2)); self.amean = att.mean((1, 2))
        vid = np.zeros(self.n, np.int8)
        pq, pds, pam, pamean = (vid_pct(x, vid) for x in (self.q, self.ds, self.am, self.amean))
        self.qrank = pq
        seg = np.zeros(self.n, np.int64); cur = 0; last = -10 ** 9
        for i in range(self.n):
            if isl[i]:
                if self.fn[i] - last > SEG_GAP:
                    cur += 1
                seg[i] = cur; last = self.fn[i]
        self.seg = seg; self.nseg = cur
        g = h['gfeat'][ix].astype(np.float32); g = g - g.mean(0)
        self.gf = torch.from_numpy(g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)).to(DEV)
        self.static = torch.from_numpy(np.stack([pq, pds, pam, pamean, vid_smooth(pq, vid), vid_smooth(pds, vid)],
                                                1)).to(DEV)
        self.fnt = torch.from_numpy(self.fn.astype(np.float32)).to(DEV)

    def reward(self, i: int, covered: set, lam: float) -> float:
        new = self.seg[i] > 0 and self.seg[i] not in covered
        return float(self.iou[i]) + lam * float(new) - FP_PEN * float(self.isl[i] == 0)


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NF, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rollout(pol: Policy, v: Video, bud: float, lam: float, greedy: bool) -> tuple[list[int], float, torch.Tensor]:
    """1 episode trên video v. Trả (frame đã chọn, tổng reward, tổng log-prob)."""
    k = n_pick(v.n, bud); sel = torch.zeros(v.n, dtype=torch.bool, device=DEV)
    mind = torch.full((v.n,), float(v.n), device=DEV); maxcos = torch.full((v.n,), -1.0, device=DEV)
    near = torch.zeros(v.n, device=DEV); picked: list[int] = []; covered: set = set(); ret = 0.0
    logps = []; idx = torch.arange(v.n, device=DEV, dtype=torch.float32)
    for t in range(k):
        ctx = torch.stack([(mind / v.n).clamp(max=1.0), near, maxcos,
                           torch.full_like(mind, t / k), torch.full_like(mind, k / v.n), torch.full_like(mind, float(t > 0))], 1)
        logit = pol(torch.cat([v.static, ctx], 1)).masked_fill(sel, -1e9)
        if greedy:
            a = int(logit.argmax())
        else:
            lp = torch.log_softmax(logit, 0); a = int(torch.multinomial(lp.exp(), 1)); logps.append(lp[a])
        ret += v.reward(a, covered, lam)
        if v.seg[a] > 0:
            covered.add(int(v.seg[a]))
        picked.append(a); sel = sel.clone(); sel[a] = True
        mind = torch.minimum(mind, (idx - a).abs()); maxcos = torch.maximum(maxcos, v.gf @ v.gf[a])
        near = torch.maximum(near, ((v.fnt - v.fnt[a]).abs() <= NEAR).float())
    return picked, ret, (torch.stack(logps).sum() if logps else torch.zeros((), device=DEV))


def nms_pick(v: Video, bud: float, w: int) -> list[int]:
    """Tham lam theo q̂, bỏ frame trong ±w chỉ số quanh frame đã chọn (nếu cạn thì nới)."""
    k = n_pick(v.n, bud); order = np.argsort(-v.q); blocked = np.zeros(v.n, bool); out: list[int] = []
    for i in order:
        if len(out) == k:
            break
        if not blocked[i]:
            out.append(int(i)); blocked[max(0, i - w):i + w + 1] = True
    if len(out) < k:
        out += [int(i) for i in order if i not in set(out)][:k - len(out)]
    return out


def metrics(v: Video, picked: list[int], lam: float, use_top1: bool = False) -> dict[str, float]:
    p = np.array(picked); iou = (v.iou_top1 if use_top1 else v.iou)[p]
    cov = len({int(s) for s in v.seg[p] if s > 0}) / max(v.nseg, 1)
    obj = float(iou.sum()) + lam * cov * v.nseg - FP_PEN * float((v.isl[p] == 0).sum())
    return {'iou': float(iou.mean()), 'pct': float(v.isl[p].mean()), 'dr': float((iou >= 0.5).mean()),
            'cov': cov, 'obj': obj / len(p)}


def oracle_pick(v: Video, bud: float, lam: float) -> list[int]:
    """Tham lam theo reward thật (IoU thật + LAMBDA*đoạn mới) — cận trên thực tế."""
    k = n_pick(v.n, bud); covered: set = set(); out: list[int] = []; avail = np.ones(v.n, bool)
    for _ in range(k):
        gain = v.iou + lam * np.array([s > 0 and s not in covered for s in v.seg], np.float32) - FP_PEN * (1 - v.isl)
        gain[~avail] = -1e9; a = int(gain.argmax()); out.append(a); avail[a] = False
        if v.seg[a] > 0:
            covered.add(int(v.seg[a]))
    return out


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5; episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    box = sys.argv[3] if len(sys.argv) > 3 else 'rerank'
    kfix = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    budgets = (kfix,) if kfix else BUDGETS; tag = '_k%d' % kfix if kfix else ''
    torch.manual_seed(0); np.random.seed(0)
    f = h5py.File('data/vg_frames.h5', 'r')
    h = {k: f[k][...] for k in ('fn', 'isp', 'hasgt', 'split', 'boxes', 'scores', 'gt', 'gfeat')}
    vid = f['vid'][...].astype(str); loc = dict(np.load('data/vg_loc.npz', allow_pickle=True))
    V = {v: Video(h, np.where(vid == v)[0], loc, box) for v in sorted(set(vid))}
    split = {v: int(h['split'][vid == v][0]) for v in V}
    tr = [v for v in V if split[v] == 0 and V[v].nseg > 0]; te = [v for v in V if split[v] == 1 and V[v].nseg > 0]
    print('box %s | ' % box + 'lambda %.2f | video train (có bệnh) %d | test %d | đoạn bệnh test %d' % (
        lam, len(tr), len(te), sum(V[v].nseg for v in te)), flush=True)
    # chọn w cho NMS thời gian trên TRAIN
    best_w = max((1, 2, 3, 5, 8, 12, 20), key=lambda w: np.mean(
        [metrics(V[v], nms_pick(V[v], b, w), lam)['obj'] for v in tr for b in budgets]))
    pol = Policy().to(DEV); opt = torch.optim.Adam(pol.parameters(), lr=3e-4); rng = np.random.RandomState(0)
    t0 = time.time(); hist = []
    for ep in range(1, episodes + 1):
        v = V[tr[rng.randint(len(tr))]]; b = budgets[rng.randint(len(budgets))]
        _, r_s, lp = rollout(pol, v, b, lam, False)
        with torch.no_grad():
            _, r_g, _ = rollout(pol, v, b, lam, True)
        k = n_pick(v.n, b); loss = -((r_s - r_g) / k) * lp
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(r_s / k)
        if ep % 250 == 0:
            print('ep %5d | train reward/pick %.3f | %.0fs' % (ep, np.mean(hist[-250:]), time.time() - t0), flush=True)
    pol.eval(); res: dict = {'lambda': lam, 'nms_w': best_w, 'budgets': {}}
    methods = {
        'random': lambda v, b: list(np.random.RandomState(1).permutation(v.n)[:n_pick(v.n, b)]),
        'det-conf top-k + box top-1 (cũ)': None,
        'q̂ top-k (grid-attention)': lambda v, b: list(np.argsort(-v.q)[:n_pick(v.n, b)]),
        'q̂ + NMS thời gian (w=%d)' % best_w: lambda v, b: nms_pick(v, b, best_w),
        'RL (mới)': lambda v, b: rollout(pol, v, b, lam, True)[0],
        'oracle': lambda v, b: oracle_pick(v, b, lam),
    }
    for b in budgets:
        rows = {}
        for name, fn in methods.items():
            ms = []
            for v in te:
                if fn is None:
                    ms.append(metrics(V[v], list(np.argsort(-V[v].ds)[:n_pick(V[v].n, b)]), lam, use_top1=True))
                else:
                    with torch.no_grad():
                        ms.append(metrics(V[v], fn(V[v], b), lam))
            rows[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        res['budgets'][str(b)] = rows
        label = '%d frame/video' % b if b >= 1 else '%d%%' % (b * 100)
        print('\n--- budget %s (TB trên %d video test có bệnh) ---' % (label, len(te)))
        print('%-34s %8s %8s %8s %9s %9s' % ('method', 'IoU set', '% bệnh', 'DR set', 'phủ đoạn', 'objective'))
        for name, m in rows.items():
            print('%-34s %8.3f %7.1f%% %7.1f%% %8.1f%% %9.3f' % (name, m['iou'], 100 * m['pct'], 100 * m['dr'],
                                                                 100 * m['cov'], m['obj']))
    json.dump(res, open('data/vg_select_l%s_%s%s.json' % (lam, box, tag), 'w'), indent=1, ensure_ascii=False)
    torch.save(pol.state_dict(), 'data/vg_select_l%s_%s%s.pth' % (lam, box, tag))


if __name__ == '__main__':
    main()
