"""Video-grid — TẦNG KHÔNG GIAN (v2): grid-attention re-ranker chọn lại box trong top-10 đề xuất của detector,
và head chất lượng frame q̂ (IoU dự đoán của box output) cho tầng chọn frame.

Re-ranker: điểm(đề xuất) = a * logit(score detector) + f(đặc trưng), f khởi tạo = 0 -> mặc định GIỮ thứ hạng
detector, chỉ đổi box khi có bằng chứng (bản v1 hồi quy IoU tự do làm TEST tụt 0.613 -> 0.504).
Đặc trưng f: score/hạng/hình học/đồng thuận giữa các đề xuất + thống kê attention lưới 8x8 trong/ngoài box
(+ ROI 2x2 tuỳ biến thể). Nhánh attention học thêm nhãn ô (frame bệnh và frame thường).
Biến thể: list-noroi, list-roi (listwise CE, target ∝ exp(IoU/τ)), rl-noroi (bandit REINFORCE, reward = IoU
đề xuất đã chọn). CHỌN BIẾN THỂ THEO IoU OUT-OF-FOLD trên video train (không nhìn test); nếu không biến thể nào
vượt detector top-1 OOF thì giữ detector top-1.
Head q̂: MLP trên đặc trưng của box đã chọn, học IoU(box, GT) (frame thường = 0), học chéo 4 fold.
Chạy: python vg_localize.py  -> data/vg_loc.npz (choice, qhat, att, res)"""
from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda'; W = 336; KP = 10; G = 8; CELL_THR = 0.3; TAU = 0.1
EPOCHS = 40; BATCH = 256; LR = 1e-3
VARIANTS = (('list-noroi', 'list', False), ('list-roi', 'list', True), ('rl-noroi', 'rl', False))


def roc_auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """AUC = xác suất frame dương có điểm cao hơn frame âm (Mann–Whitney, xử lý hoà bằng hạng trung bình)."""
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s[np.r_[np.diff(np.sort(s)) == 0, False] | np.r_[False, np.diff(np.sort(s)) == 0]]):
        m = s == v; ranks[m] = ranks[m].mean()
    npos = int(y.sum()); nneg = len(y) - npos
    return float((ranks[y > 0].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU theo cặp broadcast; a,b (...,4)."""
    x0 = np.maximum(a[..., 0], b[..., 0]); y0 = np.maximum(a[..., 1], b[..., 1])
    x1 = np.minimum(a[..., 2], b[..., 2]); y1 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def cell_cover(boxes: np.ndarray) -> np.ndarray:
    """(...,4) -> (...,G,G) tỉ lệ diện tích mỗi ô bị box phủ."""
    e = np.arange(G + 1) * (W / G)
    ix = np.clip(np.minimum(boxes[..., 2:3], e[1:]) - np.maximum(boxes[..., 0:1], e[:-1]), 0, None) / (W / G)
    iy = np.clip(np.minimum(boxes[..., 3:4], e[1:]) - np.maximum(boxes[..., 1:2], e[:-1]), 0, None) / (W / G)
    return iy[..., :, None] * ix[..., None, :]


def load() -> dict[str, np.ndarray]:
    h = h5py.File('data/vg_frames.h5', 'r')
    d = {k: h[k][...] for k in ('boxes', 'scores', 'gt', 'isp', 'hasgt', 'split', 'fold', 'fn', 'roi', 'grid')}
    d['vid'] = h['vid'][...].astype(str)
    normal_tr = sorted({v for v, s, f in zip(d['vid'], d['split'], d['fold']) if s == 0 and f < 0})
    nfold = {v: i % 4 for i, v in enumerate(normal_tr)}
    d['cvfold'] = np.array([f if f >= 0 else nfold.get(v, -1) for v, f in zip(d['vid'], d['fold'])], np.int32)
    d['valid'] = (d['boxes'][..., 2] - d['boxes'][..., 0]) * (d['boxes'][..., 3] - d['boxes'][..., 1]) > 0
    d['target'] = np.where(d['hasgt'][:, None] > 0, box_iou(d['boxes'], d['gt'][:, None, :]), 0.0).astype(np.float32)
    d['cell'] = ((cell_cover(d['gt']) >= CELL_THR) & (d['hasgt'][:, None, None] > 0)).astype(np.float32)
    d['cover'] = cell_cover(d['boxes']).astype(np.float32)                                   # (N,KP,G,G)
    pi = box_iou(d['boxes'][:, :, None, :], d['boxes'][:, None, :, :])                     # (N,KP,KP)
    sc = d['scores']; b = d['boxes'] / W
    w = b[..., 2] - b[..., 0]; hh = b[..., 3] - b[..., 1]
    d['support'] = ((pi > 0.5).astype(np.float32) @ sc[..., None])[..., 0]
    d['detlogit'] = np.log(np.clip(sc, 1e-4, 1 - 1e-4) / (1 - np.clip(sc, 1e-4, 1 - 1e-4))).astype(np.float32)
    d['scal'] = np.concatenate([
        (sc / np.maximum(sc[:, :1], 1e-6))[..., None],
        np.broadcast_to(np.eye(KP, dtype=np.float32), sc.shape + (KP,)),
        ((b[..., 0] + b[..., 2]) / 2)[..., None], ((b[..., 1] + b[..., 3]) / 2)[..., None], w[..., None], hh[..., None],
        np.log(np.maximum(w, 1e-3) / np.maximum(hh, 1e-3))[..., None], (w * hh)[..., None],
        pi[:, :, 0][..., None], (d['support'] / np.maximum(d['support'].max(1, keepdims=True), 1e-6))[..., None],
    ], -1).astype(np.float32)
    return d


def att_stats(cover: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """cover (B,KP,G,G), a (B,1,G,G) -> (B,KP,4): TB attention trong box, % khối attention trong box, đỉnh, ngoài box."""
    c = cover.clamp(max=1.0)
    inside = (c * a).sum((2, 3)) / c.sum((2, 3)).clamp_min(1e-6)
    mass = (c * a).sum((2, 3)) / a.sum((2, 3)).clamp_min(1e-6)
    peak = (a * (c > 0.3)).amax((2, 3))
    outside = ((1 - c) * a).sum((2, 3)) / (1 - c).sum((2, 3)).clamp_min(1e-6)
    return torch.stack([inside, mass, peak, outside], -1)


class Reranker(nn.Module):
    """điểm = a * logit(score detector) + f(đặc trưng); f khởi tạo 0."""

    def __init__(self, nscal: int, use_roi: bool) -> None:
        super().__init__(); self.use_roi = use_roi
        self.att = nn.Sequential(nn.Conv2d(256, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 1, 1))
        self.roi = nn.Sequential(nn.Linear(1024, 32), nn.ReLU(), nn.Dropout(0.5)) if use_roi else None
        last = nn.Linear(64, 1); nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.f = nn.Sequential(nn.Linear(nscal + 4 + (32 if use_roi else 0), 64), nn.ReLU(), nn.Dropout(0.2), last)
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logit = self.att(x['grid'])[:, 0]; a = torch.sigmoid(logit).unsqueeze(1)
        st = att_stats(x['cover'], a); feats = [x['scal'], st]
        if self.use_roi:
            feats.append(self.roi(x['roi']))
        s = self.a * x['detlogit'] + self.f(torch.cat(feats, -1)).squeeze(-1)
        return s.masked_fill(~x['valid'], -1e9), logit, st


class Stats:
    """Chuẩn hoá grid/roi theo tập train của mỗi lần fit."""

    def __init__(self, d: dict, ix: np.ndarray) -> None:
        g = d['grid'][ix].astype(np.float32); r = d['roi'][ix].astype(np.float32)
        self.gm = torch.from_numpy(g.mean((0, 2, 3))).to(DEV).view(1, -1, 1, 1)
        self.gs = torch.from_numpy(g.std((0, 2, 3)) + 1e-5).to(DEV).view(1, -1, 1, 1)
        self.rm = torch.from_numpy(r.mean((0, 1))).to(DEV); self.rs = torch.from_numpy(r.std((0, 1)) + 1e-5).to(DEV)


def to_dev(d: dict, ix: np.ndarray, st: Stats) -> dict[str, torch.Tensor]:
    t = {k: torch.from_numpy(np.ascontiguousarray(d[k][ix])).to(DEV)
         for k in ('scal', 'cover', 'valid', 'target', 'cell', 'detlogit', 'hasgt')}
    t['grid'] = (torch.from_numpy(d['grid'][ix]).to(DEV).float() - st.gm) / st.gs
    t['roi'] = (torch.from_numpy(d['roi'][ix]).to(DEV).float() - st.rm) / st.rs
    return t


def fit(d: dict, tr: np.ndarray, loss_kind: str, use_roi: bool, seed: int) -> tuple[Reranker, Stats]:
    """Re-rank học trên frame có GT; nhánh attention học thêm trên số frame thường bằng số frame có GT."""
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    pos = tr[d['hasgt'][tr] > 0]; neg = rng.choice(tr[d['isp'][tr] == 0], len(pos), replace=False)
    use = np.concatenate([pos, neg]); st = Stats(d, use)
    net = Reranker(d['scal'].shape[-1], use_roi).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4); base = 0.0
    for _ in range(EPOCHS):
        net.train()
        for bi in np.array_split(rng.permutation(use), max(1, len(use) // BATCH)):
            x = to_dev(d, np.sort(bi), st); s, logit, _ = net(x); m = x['hasgt'] > 0
            loss = 0.5 * F.binary_cross_entropy_with_logits(logit, x['cell'])
            if m.any():
                lp = torch.log_softmax(s[m], -1)
                if loss_kind == 'list':
                    tgt = torch.softmax((x['target'][m] / TAU).masked_fill(~x['valid'][m], -1e9), -1)
                    loss = loss - (tgt * lp).sum(-1).mean()
                else:
                    act = torch.multinomial(lp.exp(), 1).squeeze(1); r = x['target'][m][torch.arange(len(act)), act]
                    loss = loss - ((r - base) * lp[torch.arange(len(act)), act]).mean()
                    base = 0.9 * base + 0.1 * float(r.mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval(), st


@torch.no_grad()
def predict(net: Reranker, st: Stats, d: dict, ix: np.ndarray) -> dict[str, np.ndarray]:
    """ix đã sắp. Trả choice, xác suất & margin của đề xuất chọn, thống kê attention của box chọn, bản đồ attention."""
    out: dict[str, list] = {k: [] for k in ('choice', 'prob', 'margin', 'st', 'att')}
    for bi in np.array_split(ix, max(1, len(ix) // 1024)):
        x = to_dev(d, bi, st); s, logit, stt = net(x); p = torch.softmax(s, -1); top = s.topk(2, -1)
        c = top.indices[:, 0]; ar = torch.arange(len(c), device=DEV)
        out['choice'].append(c.cpu().numpy()); out['prob'].append(p[ar, c].cpu().numpy())
        out['margin'].append((top.values[:, 0] - top.values[:, 1]).clamp(-50, 50).cpu().numpy())
        out['st'].append(stt[ar, c].cpu().numpy()); out['att'].append(torch.sigmoid(logit).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def loc_metrics(d: dict, ix: np.ndarray, choice: np.ndarray) -> dict[str, float]:
    m = d['hasgt'][ix] > 0; iou = d['target'][ix[m], choice[m]]
    per_vid = [float(iou[d['vid'][ix[m]] == v].mean()) for v in np.unique(d['vid'][ix[m]])]
    return {'iou': float(iou.mean()), 'dr': float((iou >= 0.5).mean()), 'iou_macro': float(np.mean(per_vid))}


def crossfit(d: dict, tr: np.ndarray, te: np.ndarray, kind: str, roi: bool) -> dict[str, np.ndarray]:
    """Dự đoán OOF cho frame train (4 fold) + dự đoán test từ mô hình học trên toàn bộ train."""
    res: dict[str, np.ndarray] = {}
    for k in list(range(4)) + [-1]:
        fit_ix = tr[d['cvfold'][tr] != k] if k >= 0 else tr
        pred_ix = np.sort(tr[d['cvfold'][tr] == k] if k >= 0 else te)
        net, st = fit(d, fit_ix, kind, roi, 100 + k); p = predict(net, st, d, pred_ix)
        for key, val in p.items():
            if key not in res:
                res[key] = np.zeros((len(d['vid']),) + val.shape[1:], val.dtype)
            res[key][pred_ix] = val
    return res


def vid_pct(x: np.ndarray, vid: np.ndarray) -> np.ndarray:
    """Hạng phần trăm của x trong từng video (0 = thấp nhất, 1 = cao nhất). Frame đã sắp theo thời gian trong video."""
    out = np.zeros(len(x), np.float32)
    for v in np.unique(vid):
        m = np.where(vid == v)[0]; out[m] = np.argsort(np.argsort(x[m], kind='mergesort')) / max(len(m) - 1, 1)
    return out


def vid_smooth(x: np.ndarray, vid: np.ndarray, r: int = 3) -> np.ndarray:
    """Trung bình trượt ±r frame trong từng video."""
    out = np.zeros(len(x), np.float32); k = np.ones(2 * r + 1) / (2 * r + 1)
    for v in np.unique(vid):
        m = np.where(vid == v)[0]; out[m] = np.convolve(np.pad(x[m], r, mode='edge'), k, 'valid')
    return out


def quality_feats(d: dict, p: dict[str, np.ndarray]) -> np.ndarray:
    """Chỉ dùng đặc trưng BẤT BIẾN với thang score của detector: hạng % trong video, tỉ lệ trong frame, hình học.
    (Detector OOF kém tự tin hơn detector chính: score trung vị frame bệnh 0.50 ở train vs 0.96 ở test.)"""
    n = len(d['vid']); ar = np.arange(n); c = p['choice']; b = d['boxes'][ar, c] / W; att = p['att'].reshape(n, -1)
    vid = d['vid']; sc = d['scores']
    raw = [sc[:, 0], d['detlogit'][ar, c], d['support'][ar, c], p['prob'], p['margin'], att.max(1), att.mean(1),
           np.sort(att, 1)[:, -3:].mean(1), p['st'][:, 0], p['st'][:, 2]]
    pct = [vid_pct(x, vid) for x in raw]
    smooth = [vid_smooth(pct[0], vid), vid_smooth(pct[5], vid), vid_smooth(pct[0], vid, 10)]
    rel = [sc[:, 1] / np.maximum(sc[:, 0], 1e-6), c / KP, p['st'][:, 1], (b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1]),
           (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])]
    return np.stack(pct + smooth + rel, 1).astype(np.float32)


def quality_head(d: dict, tr: np.ndarray, te: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """q̂ = IoU dự đoán của box output; học chéo 4 fold, cân bằng frame bệnh/thường."""
    x = quality_feats(d, p); n = len(x); ar = np.arange(n)
    y = np.where(d['hasgt'] > 0, d['target'][ar, p['choice']], 0.0).astype(np.float32)
    keep = (d['hasgt'] > 0) | (d['isp'] == 0); q = np.zeros(n, np.float32)
    for k in list(range(4)) + [-1]:
        fi = tr[(d['cvfold'][tr] != k) & keep[tr]] if k >= 0 else tr[keep[tr]]
        pi_ = tr[d['cvfold'][tr] == k] if k >= 0 else te
        torch.manual_seed(200 + k); mu = x[fi].mean(0); sd = x[fi].std(0) + 1e-5
        net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.from_numpy((x[fi] - mu) / sd).to(DEV); yt = torch.from_numpy(y[fi]).to(DEV)
        pos = torch.from_numpy(d['isp'][fi] > 0).to(DEV)
        wt = torch.where(pos, 0.5 / pos.float().mean(), 0.5 / (1 - pos.float().mean()))
        for _ in range(300):
            for bi in torch.randperm(len(fi), device=DEV).split(2048):
                loss = (wt[bi] * (torch.sigmoid(net(xt[bi]).squeeze(-1)) - yt[bi]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            q[pi_] = torch.sigmoid(net(torch.from_numpy((x[pi_] - mu) / sd).to(DEV)).squeeze(-1)).cpu().numpy()
    return q


def main() -> None:
    d = load(); tr = np.where(d['split'] == 0)[0]; te = np.where(d['split'] == 1)[0]; n = len(d['vid'])
    print('frame train %d (GT %d) | test %d (GT %d)' % (len(tr), int(d['hasgt'][tr].sum()), len(te),
                                                      int(d['hasgt'][te].sum())), flush=True)
    zero = np.zeros(n, np.int64); res: dict = {'variants': {}}
    res['det_top1'] = {'oof': loc_metrics(d, tr, zero[tr]), 'test': loc_metrics(d, te, zero[te])}
    best = te_best = None
    orc = d['target'][te][d['hasgt'][te] > 0].max(1); res['oracle10_test'] = {'iou': float(orc.mean()),
                                                                               'dr': float((orc >= 0.5).mean())}
    print('%-22s OOF IoU %.3f DR %.1f%% | TEST IoU %.3f DR %.1f%% macro %.3f' % (
        'detector top-1', res['det_top1']['oof']['iou'], 100 * res['det_top1']['oof']['dr'],
        res['det_top1']['test']['iou'], 100 * res['det_top1']['test']['dr'], res['det_top1']['test']['iou_macro']),
        flush=True)
    preds = {}
    for name, kind, roi in VARIANTS:
        p = crossfit(d, tr, te, kind, roi); preds[name] = p
        r = {'oof': loc_metrics(d, tr, p['choice'][tr]), 'test': loc_metrics(d, te, p['choice'][te])}
        r['switch_test'] = float((p['choice'][te][d['hasgt'][te] > 0] != 0).mean()); res['variants'][name] = r
        print('%-22s OOF IoU %.3f DR %.1f%% | TEST IoU %.3f DR %.1f%% macro %.3f | đổi box %.1f%%' % (
            name, r['oof']['iou'], 100 * r['oof']['dr'], r['test']['iou'], 100 * r['test']['dr'],
            r['test']['iou_macro'], 100 * r['switch_test']), flush=True)
        if best is None or r['oof']['iou'] > res['variants'][best]['oof']['iou']:
            best = name
    use_rerank = res['variants'][best]['oof']['iou'] > res['det_top1']['oof']['iou']
    chosen = best if use_rerank else 'detector top-1'; p = dict(preds[best])
    if not use_rerank:
        p['choice'] = zero.copy()
    res['chosen'] = chosen; te_best = res['variants'][best]['test'] if use_rerank else res['det_top1']['test']
    print('=> chọn theo OOF: %s | TEST IoU %.3f DR %.1f%% | oracle best-of-10 %.3f / %.1f%%' % (
        chosen, te_best['iou'], 100 * te_best['dr'], res['oracle10_test']['iou'], 100 * res['oracle10_test']['dr']),
        flush=True)
    q = quality_head(d, tr, te, p); isl = d['isp'][te] > 0; m = d['hasgt'][te] > 0
    iou_out = d['target'][te][m, p['choice'][te][m]]
    res['auc'] = {'qhat': roc_auc_score(isl, q[te]), 'det': roc_auc_score(isl, d['scores'][te, 0])}
    res['corr_iou'] = {'qhat': float(np.corrcoef(q[te][m], iou_out)[0, 1]),
                       'det': float(np.corrcoef(d['scores'][te][m, 0], iou_out)[0, 1])}
    print('q̂ TEST: AUC bệnh/thường %.3f (det-score %.3f) | tương quan với IoU box output %.3f (det-score %.3f)' % (
        res['auc']['qhat'], res['auc']['det'], res['corr_iou']['qhat'], res['corr_iou']['det']), flush=True)
    np.savez('data/vg_loc.npz', choice=p['choice'], qhat=q, att=p['att'].astype(np.float16),
             res=np.array([res], dtype=object))


if __name__ == '__main__':
    main()
