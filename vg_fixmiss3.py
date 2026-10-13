"""Sửa box detector khoanh SAI (v3): như vg_fixmiss.py (reward +2·[IoU>=0.5] + ΔGIoU − 0.02, 23 hành động dịch/nhảy/
STOP, cấm dịch đảo ngược và nhảy về đề xuất đã thăm) + các thành phần bật/tắt được để ablation:
  --temporal R : tín hiệu thời gian từ ±R frame lân cận (bản đồ nhiệt 28x28 + độ ủng hộ box hiện tại / từng đề xuất)
  --attn       : LƯỚI ATTENTION 24x24 "ý kiến thứ hai" (DINOv2, vg_attn.py) vào state: 1 kênh ở vùng quanh box và
                 góc nhìn toàn ảnh + đặc trưng attention (% khối trong box, TB trong, đỉnh trong/ngoài) của box hiện tại
                 và từng đề xuất
  --beta β     : REWARD TUỲ BIẾN THEO LƯỚI ATTENTION: thêm β·ΔΦ mỗi bước, Φ(box) = Dice(box, bản đồ attention)
                 = 2·Σ A·phủ(box) / (Σ A + Σ phủ(box)). Φ quan sát được cả lúc chạy thật (không cần GT) -> agent học bám
                 theo tín hiệu có sẵn ở test, không chỉ theo GT.
  --cv4        : chọn (vòng train, δ) bằng TỔNG lợi ròng trên 4 fold video luân phiên (mỗi fold: train 3 fold, kiểm
                 1 fold), rồi trên test dùng ENSEMBLE trung bình Q của 4 model. Hai tiêu chí chọn: ròng thường và ròng
                 HIỆU CHỈNH TỈ LỆ NỀN (tỉ lệ box sai của test ước lượng bằng EM trên ảnh test KHÔNG nhãn).
Chạy: python vg_fixmiss3.py --temporal 3 --attn --beta 1 --cv4 --tag t3_attn_b1  -> data/vg_fixmiss3_<tag>.json"""
from __future__ import annotations

import argparse
import copy
import json
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

import vg_fixmiss as fx
import vg_missgate as mg
import vg_refine as vr
import vg_refine_vis as rv
from vg_localize import box_iou

DEV = fx.DEV; W = fx.W; G = fx.G; CELL = fx.CELL; VIEW = fx.VIEW; EXP = fx.EXP
NMOVE, NJUMP, STOP, NA, T, HIT = fx.NMOVE, fx.NJUMP, fx.STOP, fx.NA, fx.T, fx.HIT
P = 24; DELTAS = fx.DELTAS; NENV = fx.NENV; GAMMA = fx.GAMMA


def cover24(boxes: torch.Tensor) -> torch.Tensor:
    e = torch.arange(P + 1, device=boxes.device, dtype=torch.float32) * (W / P)
    ix = (torch.minimum(boxes[..., 2:3], e[1:]) - torch.maximum(boxes[..., 0:1], e[:-1])).clamp_min(0) / (W / P)
    iy = (torch.minimum(boxes[..., 3:4], e[1:]) - torch.maximum(boxes[..., 1:2], e[:-1])).clamp_min(0) / (W / P)
    return iy[..., :, None] * ix[..., None, :]


def attn_feats(att: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """att (B,P,P), boxes (B,K,4) -> (B,K,5): % khối trong box, TB trong, đỉnh trong, đỉnh ngoài, ngoài − trong."""
    c = cover24(boxes); a = att[:, None]; inside = (c * a).sum((-1, -2))
    mass = inside / (a.sum((-1, -2)) + 1e-6); mean_in = inside / (c.sum((-1, -2)) + 1e-6)
    peak_in = (a * (c > 0.3)).amax((-1, -2)); peak_out = (a * (c <= 0)).amax((-1, -2))
    return torch.stack([mass, mean_in, peak_in, peak_out, peak_out - peak_in], -1)


def dice_phi(att: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Φ(box) = Dice giữa box và bản đồ attention; att (B,P,P), box (B,4)."""
    c = cover24(box)
    return 2 * (c * att).sum((-1, -2)) / (att.sum((-1, -2)) + c.sum((-1, -2)) + 1e-6)


class Data3(vr.Data):
    def __init__(self, temporal: int, attn: bool, beta: float) -> None:
        super().__init__(); self.R = temporal; self.temporal = temporal > 0; self.attn = attn; self.beta = beta
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
        if self.temporal:
            fx.R_NB, fx.NB = temporal, 2 * temporal
            fx.FixData._temporal(self, h)                                   # nbp/nbs/nbm, heat, heat14, tsup, nbfrac
        if attn:
            z = np.load('data/vg_attn.npz'); assert np.array_equal(z['row'], self.row)
            self.att = torch.from_numpy(z['att'].astype(np.float32)).to(DEV)
            self.att28 = F.interpolate(self.att[:, None], size=(G, G), mode='bilinear', align_corners=False)
            self.att14 = F.interpolate(self.att[:, None], size=(14, 14), mode='bilinear', align_corners=False)
            self.pattn = torch.cat([attn_feats(self.att[i:i + 512], self.props[i:i + 512]) for i in range(0, m, 512)])
        self.tr_all = self.tr.copy()

    def set_fold(self, k: int) -> None:
        self.va = self.tr_all[self.fold[self.tr_all] == k]; self.tr = self.tr_all[self.fold[self.tr_all] != k]


def state(d: Data3, idx: torch.Tensor, box: torch.Tensor, t: torch.Tensor, last: torch.Tensor) -> tuple[torch.Tensor, ...]:
    xl, xg, sc, pf = fx.state(d, idx, box, t, last)
    if d.attn:
        b = len(idx); w = (box[:, 2] - box[:, 0]).clamp_min(2 * CELL); h = (box[:, 3] - box[:, 1]).clamp_min(2 * CELL)
        cx = (box[:, 0] + box[:, 2]) / 2; cy = (box[:, 1] + box[:, 3]) / 2
        ex = torch.stack([cx - EXP * w / 2, cy - EXP * h / 2, cx + EXP * w / 2, cy + EXP * h / 2], 1)
        rois = torch.cat([torch.arange(b, device=DEV, dtype=torch.float32)[:, None], ex], 1)
        xl = torch.cat([xl, roi_align(d.att28[idx], rois, (VIEW, VIEW), spatial_scale=G / W, sampling_ratio=2,
                                      aligned=True)], 1)
        xg = torch.cat([xg, d.att14[idx]], 1)
        cur = attn_feats(d.att[idx], box[:, None])[:, 0]; sc = torch.cat([sc, cur], 1)
        pa = d.pattn[idx]; pf = torch.cat([pf, pa, (pa[..., :1] - cur[:, None, :1])], -1)
    return xl, xg, sc, pf


class Net3(nn.Module):
    def __init__(self, d: Data3) -> None:
        super().__init__(); ec = int(d.temporal) + int(d.attn)
        nsc = 8 + NA + 2 * int(d.temporal) + 5 * int(d.attn); npf = 8 + 256 + 2 * int(d.temporal) + 6 * int(d.attn)
        self.loc = nn.Sequential(nn.Conv2d(257 + ec, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        self.glob = nn.Sequential(nn.Conv2d(257 + ec, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(64 * 25 + 32 * 16 + nsc, 256), nn.ReLU())
        self.prop = nn.Sequential(nn.Linear(npf, 64), nn.ReLU())
        self.v = nn.Linear(256, 1); self.ams = nn.Linear(256, NMOVE + 1)
        self.aj = nn.Sequential(nn.Linear(256 + 64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xl: torch.Tensor, xg: torch.Tensor, sc: torch.Tensor, pf: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([self.loc(xl).flatten(1), self.glob(xg).flatten(1), sc], 1))
        ams = self.ams(h); aj = self.aj(torch.cat([h[:, None].expand(-1, NJUMP, -1), self.prop(pf)], -1)).squeeze(-1)
        a = torch.cat([ams[:, :NMOVE], aj, ams[:, NMOVE:]], 1)
        return self.v(h) + a - a.mean(1, keepdim=True)


def env_step(d: Data3, idx: torch.Tensor, box: torch.Tensor, t: torch.Tensor,
             a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nb, r, done = fx.env_step(d, idx, box, t, a)
    if d.beta > 0:
        r = r + torch.where(a == STOP, torch.zeros_like(r),
                            d.beta * (dice_phi(d.att[idx], nb) - dice_phi(d.att[idx], box)))
    return nb, r, done


@torch.no_grad()
def rollout(d: Data3, nets: list[Net3], ix: np.ndarray, delta: float,
            log: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tham lam từ box top-1; Q = trung bình các net (ensemble); chỉ hành động nếu max Q(không STOP) − Q(STOP) > δ.
    log (tuỳ chọn): nhận hành động của từng bước (B,), STOP sau khi đã dừng."""
    idx = torch.from_numpy(ix).to(DEV); box = d.props[idx, 0].clone(); t = torch.zeros(len(ix), dtype=torch.long, device=DEV)
    last = torch.full_like(t, -1); alive = torch.ones(len(ix), dtype=torch.bool, device=DEV)
    vis = torch.zeros(len(ix), NJUMP, dtype=torch.bool, device=DEV); vis[:, 0] = True
    jumped = torch.zeros(len(ix), dtype=torch.bool, device=DEV); traj = [box.clone()]
    for _ in range(T):
        x = state(d, idx, box, t, last); q = torch.stack([n(*x) for n in nets]).mean(0).masked_fill(~fx.valid(last, vis), -1e9)
        best = q[:, :STOP].argmax(1); margin = q[:, :STOP].max(1).values - q[:, STOP]
        a = torch.where(margin > delta, best, torch.full_like(best, STOP))
        a = torch.where(alive, a, torch.full_like(a, STOP)); alive = alive & (a != STOP)
        if log is not None:
            log.append(a.clone())
        jumped = jumped | (alive & (a >= NMOVE)); vis = fx.mark(vis, a)
        box = fx.step_box(d, idx, box, a); t = t + alive.long(); last = torch.where(alive, a, last); traj.append(box.clone())
        if not alive.any():
            break
    while len(traj) < T + 1:
        traj.append(box.clone())
    return torch.stack(traj), t, jumped


def train(d: Data3, iters: int, hook, every: int = 2000, seed: int = 0) -> None:
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = Net3(d).to(DEV); tgt = copy.deepcopy(net); opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    cap = 200000; n = 0; ptr = 0
    buf = {k: torch.zeros(cap, *s, dtype=dt, device=DEV) for k, s, dt in (
        ('idx', (), torch.long), ('box', (4,), torch.float32), ('t', (), torch.long), ('last', (), torch.long),
        ('vis', (NJUMP,), torch.bool), ('a', (), torch.long), ('r', (), torch.float32), ('nbox', (4,), torch.float32),
        ('nt', (), torch.long), ('nvis', (NJUMP,), torch.bool), ('d', (), torch.float32))}
    idx, box, vis = fx.reset(d, NENV, rng); t = torch.zeros(NENV, dtype=torch.long, device=DEV); last = torch.full_like(t, -1)
    rets: list[float] = []; ep = torch.zeros(NENV, device=DEV); t0 = time.time(); hook(0, net.eval(), 'chưa học'); net.train()
    for it in range(1, iters + 1):
        ok = fx.valid(last, vis); ra = torch.rand(NENV, NA, device=DEV).masked_fill(~ok, -1).argmax(1)
        eps = max(0.05, 1.0 - it / (0.4 * iters))
        with torch.no_grad():
            a = net(*state(d, idx, box, t, last)).masked_fill(~ok, -1e9).argmax(1)
        a = torch.where(torch.rand(NENV, device=DEV) < eps, ra, a)
        nb, r, done = env_step(d, idx, box, t, a); nvis = fx.mark(vis, a)
        j = (torch.arange(NENV, device=DEV) + ptr) % cap
        for k, val in (('idx', idx), ('box', box), ('t', t), ('last', last), ('vis', vis), ('a', a), ('r', r), ('nbox', nb),
                       ('nt', t + 1), ('nvis', nvis), ('d', done.float())):
            buf[k][j] = val
        ptr = (ptr + NENV) % cap; n = min(n + NENV, cap); ep += r; nt = t + 1
        if done.any():
            rets += ep[done].tolist(); ep[done] = 0
            ni, nbx, nv = fx.reset(d, int(done.sum()), rng); idx = idx.clone(); idx[done] = ni; nb[done] = nbx
            nvis[done] = nv; nt[done] = 0
        last = torch.where(done, torch.full_like(a, -1), a); box, t, vis = nb, nt, nvis
        if n >= 5000:
            for _ in range(2):
                s = torch.randint(0, n, (128,), device=DEV); bi = buf['idx'][s]
                x = state(d, bi, buf['box'][s], buf['t'][s], buf['last'][s])
                nx = state(d, bi, buf['nbox'][s], buf['nt'][s], buf['a'][s])
                with torch.no_grad():
                    a2 = net(*nx).masked_fill(~fx.valid(buf['a'][s], buf['nvis'][s]), -1e9).argmax(1)
                    y = buf['r'][s] + GAMMA * (1 - buf['d'][s]) * tgt(*nx).gather(1, a2[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(net(*x).gather(1, buf['a'][s][:, None]).squeeze(1), y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if it % 1000 == 0:
            tgt.load_state_dict(net.state_dict())
        if it % every == 0:
            hook(it, net.eval(), 'eps %.2f | return TB %.2f | %.0fs' % (eps, np.mean(rets[-2000:]), time.time() - t0)); net.train()


def estimate_test_prior(d: Data3) -> tuple[float, float]:
    """Tỉ lệ box top-1 sai trên TEST ước lượng KHÔNG dùng nhãn test: logistic (đặc trưng detector + attention) học trên
    train, rồi hiệu chỉnh tỉ lệ nền bằng EM (Saerens và cs. 2002) trên dự đoán cho ảnh test. Trả (ước lượng, tỉ lệ train)."""
    h = h5py.File('data/vg_frames.h5', 'r'); b = h['boxes'][...][d.row]; s = h['scores'][...][d.row]
    vid = h['vid'][...].astype(str)[d.row]; gt = h['gt'][...][d.row]
    ff = mg.frame_feats(b, s, vid)
    if d.attn:
        ff = np.concatenate([ff, torch.cat([attn_feats(d.att[i:i + 512], d.props[i:i + 512, :1])[:, 0]
                                            for i in range(0, len(b), 512)]).cpu().numpy()], 1)
    ymiss = box_iou(b[:, 0], gt) < HIT; tr = d.tr_all; te = d.te
    model = mg.fit_logreg(ff[tr], ymiss[tr]); pi_tr = float(ymiss[tr].mean())
    with torch.no_grad():
        logit = model[0](torch.from_numpy((ff[te] - model[1]) / model[2])).squeeze(-1).numpy()
    pw = (1 - pi_tr) / pi_tr; p = 1 / (1 + np.exp(-(logit - np.log(pw))))           # bỏ pos_weight -> xác suất theo prior train
    pi = pi_tr
    for _ in range(100):
        q = (pi / pi_tr) * p / ((pi / pi_tr) * p + ((1 - pi) / (1 - pi_tr)) * (1 - p)); pi = float(q.mean())
    return pi, pi_tr


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--iters', type=int, default=12000); ap.add_argument('--temporal', type=int, default=3)
    ap.add_argument('--attn', action='store_true'); ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--cv4', action='store_true'); ap.add_argument('--tag', default='run')
    ap.add_argument('--every', type=int, default=1000); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    d = Data3(a.temporal, a.attn, a.beta); paths = vr.frame_paths(); folds = [0, 1, 2, 3] if a.cv4 else [0]
    pi_hat, pi_tr = estimate_test_prior(d)
    tei = torch.from_numpy(d.te).to(DEV); pi_true = float((vr.iou_t(d.props[tei, 0], d.gt[tei]) < HIT).float().mean())
    print('[%s] temporal ±%d | attn %s | β %.2f | fold %s | tỉ lệ box sai: train %.2f, test ƯỚC LƯỢNG (không nhãn) %.2f '
          '(thật %.2f, chỉ để kiểm tra)' % (a.tag, a.temporal, a.attn, a.beta, folds, pi_tr, pi_hat, pi_true), flush=True)
    i0 = vr.iou_t(d.props[tei, 0], d.gt[tei]).cpu().numpy(); rs = np.random.RandomState(3)
    pbest = vr.iou_t(d.props[tei], d.gt[tei][:, None]).max(1).values.cpu().numpy()
    visx = np.concatenate([rs.choice(d.te[(i0 == 0) & (pbest >= HIT)], 2, replace=False),
                           rs.choice(d.te[(i0 > 0.1) & (i0 < HIT)], 2, replace=False), rs.choice(d.te[i0 >= 0.7], 2, replace=False)])
    imgs = rv.load_images([paths[i] for i in visx])
    val: dict = {}; ckpt: dict = {}; lg: dict = {'its': [], 'iou_test': [], 'traj': [], 'steps': []}
    for k in folds:
        d.set_fold(k); pv = float((vr.iou_t(d.props[torch.from_numpy(d.va).to(DEV), 0],
                                          d.gt[torch.from_numpy(d.va).to(DEV)]) < HIT).float().mean())
        w_miss, w_hit = pi_hat / pv, (1 - pi_hat) / (1 - pv)

        def hook(it: int, net: Net3, msg: str, k: int = k, w_miss: float = w_miss, w_hit: float = w_hit) -> None:
            ckpt[(k, it)] = copy.deepcopy(net.state_dict())
            for dl in DELTAS:
                tr_, st_, jp_ = rollout(d, [net], d.va, dl); m = fx.metrics(d, d.va, tr_[-1], st_, jp_)
                val[(k, it, dl)] = {'net': m['net'], 'wnet': w_miss * m['fixed'] - w_hit * m['broken'], 'fixed': m['fixed'],
                                    'broken': m['broken']}
            bd = max(DELTAS, key=lambda x: val[(k, it, x)]['wnet'])
            tr_, st_, jp_ = rollout(d, [net], d.te, bd); mt = fx.metrics(d, d.te, tr_[-1], st_, jp_)
            v = val[(k, it, bd)]
            print('[%s] fold %d vòng %6d | %s | VAL δ=%.2f sửa %d phá %d (ròng %+d, hiệu chỉnh %+.1f) | TEST(cùng δ, theo '
                  'dõi) sửa %d phá %d IoU=0 %d/%d' % (a.tag, k, it, msg, bd, v['fixed'], v['broken'], v['net'], v['wnet'],
                                                     mt['fixed'], mt['broken'], mt['fixed_miss0'], mt['n_miss0']), flush=True)
            if k == folds[0]:
                vt, vs, _ = rollout(d, [net], visx, 0.0)
                for kk, xx in (('its', it), ('iou_test', mt['dr_after']), ('traj', vt.cpu().numpy()), ('steps', vs.cpu().numpy())):
                    lg[kk].append(xx)
                snap = {'its': np.array(lg['its']), 'iou_test': np.array(lg['iou_test']), 'traj': np.stack(lg['traj']),
                        'steps': np.stack(lg['steps']), 'gt': d.gt[torch.from_numpy(visx).to(DEV)].cpu().numpy(),
                        'paths': np.array([paths[i] for i in visx])}
                np.savez('data/vg_fixmiss3_vis_%s.npz' % a.tag, **snap)
                rv.render_progress(snap, imgs, 'fig_fixmiss3_progress_%s.png' % a.tag,
                                   'Sửa box khoanh sai — %s (fold %d, δ=0)' % (a.tag, k), metric='TEST DR@0.5')
        train(d, a.iters, hook, a.every, a.seed)
    its = sorted({it for (_, it) in ckpt})
    res: dict = {'args': vars(a), 'pi_train': pi_tr, 'pi_test_hat': pi_hat, 'pi_test_true': pi_true, 'select': {}}
    for crit in ('net', 'wnet'):
        score = {(it, dl): sum(val[(k, it, dl)][crit] for k in folds) for it in its for dl in DELTAS}
        (it_b, dl_b), sb = max(score.items(), key=lambda kv: (kv[1], -kv[0][0]))
        if sb <= 0:
            m = fx.metrics(d, d.te, d.props[torch.from_numpy(d.te).to(DEV), 0], torch.zeros(len(d.te)), torch.zeros(len(d.te)))
            res['select'][crit] = {'fallback': True, 'score': sb, 'test': m}
            print('[%s] chọn theo %s: không (vòng, δ) nào có tổng > 0 -> GIỮ NGUYÊN detector' % (a.tag, crit)); continue
        nets = []
        for k in folds:
            nn_ = Net3(d).to(DEV); nn_.load_state_dict(ckpt[(k, it_b)]); nets.append(nn_.eval())
        tr_, st_, jp_ = rollout(d, nets, d.te, dl_b); m = fx.metrics(d, d.te, tr_[-1], st_, jp_)
        single = []
        for n_ in nets:
            t1, s1, j1 = rollout(d, [n_], d.te, dl_b); single.append(fx.metrics(d, d.te, t1[-1], s1, j1)['net'])
        res['select'][crit] = {'fallback': False, 'it': it_b, 'delta': dl_b, 'score': sb, 'test': m, 'test_single_net': single}
        torch.save({'states': [ckpt[(k, it_b)] for k in folds], 'delta': dl_b, 'it': it_b, 'args': vars(a)},
                   'data/vg_fixmiss3_%s_%s.pth' % (a.tag, crit))
        print('[%s] chọn theo %s: vòng %d δ=%.2f (tổng val %.1f) -> TEST ensemble DR %.1f%%->%.1f%% | sửa %d phá %d ròng %+d | '
              'IoU=0 cứu %d/%d | từng model: %s' % (a.tag, crit, it_b, dl_b, sb, 100 * m['dr_before'], 100 * m['dr_after'],
                                                   m['fixed'], m['broken'], m['net'], m['fixed_miss0'], m['n_miss0'], single))
    res['val'] = {'%d_%d_%.2f' % key: v for key, v in val.items()}
    json.dump(res, open('data/vg_fixmiss3_%s.json' % a.tag, 'w'), indent=1, default=float)
    print('[%s] XONG' % a.tag, flush=True)


if __name__ == '__main__':
    main()
