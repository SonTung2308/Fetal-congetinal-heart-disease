"""Cổng ĐỔI ĐỀ XUẤT ít tham số cho mục tiêu "sửa box detector khoanh sai (IoU < 0.5) mà không phá ca đúng".
Hai câu hỏi, cả hai chỉ dùng đặc trưng thấp chiều, tương đối (không phụ thuộc thang score detector):
  (1) Box top-1 có đang SAI không?  -> hồi quy logistic, đo AUC (khả năng nhận ra ca sai).
  (2) Nếu đổi sang đề xuất #k (k = 2..10) thì có SỬA được không? -> hồi quy logistic trên đặc trưng từng đề xuất.
Quyết định: đổi sang đề xuất có P(sửa) cao nhất nếu P > τ, ngược lại giữ top-1. τ chọn bằng cross-validation
4 fold theo video trên dữ liệu OOF (video train), tối đa (số ca sửa − số ca phá). Không nhìn test.
Tín hiệu THỜI GIAN (riêng của video): độ ủng hộ của mỗi đề xuất = tổng score tương đối của các đề xuất ở frame lân
cận (±R frame) chồng lên nó (IoU > 0.3) — tổn thương tồn tại qua nhiều frame liên tiếp. So 3 cách: cổng logistic không
thời gian, cổng logistic + thời gian, luật không học (đổi nếu ủng hộ > ρ × ủng hộ của top-1).
Chạy: python vg_missgate.py  -> in bảng + data/vg_missgate.json"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from vg_localize import box_iou, roc_auc_score, vid_pct

HIT = 0.5; KP = 10; W = 336.0; R_NB = 3
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def temporal_support(h: h5py.File, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(len(rows),KP) độ ủng hộ theo thời gian của từng đề xuất, và số frame lân cận."""
    vid = h['vid'][...].astype(str); fn = h['fn'][...]; b = h['boxes'][...]; s = h['scores'][...]
    srel = s / np.maximum(s[:, :1], 1e-6); sup = np.zeros((len(rows), KP)); nn_ = np.zeros(len(rows))
    start = {}; end = {}
    for i, v in enumerate(vid):
        start.setdefault(v, i); end[v] = i + 1
    for r, i in enumerate(rows):
        a, e = start[vid[i]], end[vid[i]]; lo = max(a, i - 3 * R_NB); j = np.arange(lo, min(e, i + 3 * R_NB + 1))
        j = j[(np.abs(fn[j] - fn[i]) <= R_NB) & (j != i)]
        if len(j):
            ov = box_iou(b[i][:, None, None, :], b[j][None]) > 0.3                 # (KP, nj, KP)
            sup[r] = (ov * srel[j][None]).max(2).sum(1) / len(j); nn_[r] = len(j)
    return sup.astype(np.float32), nn_.astype(np.float32)


def frame_feats(b: np.ndarray, s: np.ndarray, vid: np.ndarray) -> np.ndarray:
    """Đặc trưng 'box top-1 có sai không' cho mỗi frame (N,F)."""
    p1 = b[:, 0]; pi = box_iou(b[:, :1], b)                                    # (N,KP) IoU của top-1 với mọi đề xuất
    sup = (s * (pi > 0.5)).sum(1) / np.maximum(s.sum(1), 1e-6)
    far = pi < 0.1; comp = np.where(far, s, 0).max(1); ci = np.where(far, s, -1).argmax(1)
    c1 = (p1[:, :2] + p1[:, 2:]) / 2; cc = (b[np.arange(len(b)), ci, :2] + b[np.arange(len(b)), ci, 2:]) / 2
    w = p1[:, 2] - p1[:, 0]; h = p1[:, 3] - p1[:, 1]
    return np.stack([vid_pct(s[:, 0], vid), s[:, 1] / np.maximum(s[:, 0], 1e-6), sup, comp / np.maximum(s[:, 0], 1e-6),
                     pi[:, 1], np.hypot(*(c1 - cc).T) / W * far.any(1), w * h / W ** 2, np.log(np.maximum(w, 1) / np.maximum(h, 1)),
                     np.hypot(*(c1 - W / 2).T) / W, (s > 0.5 * s[:, :1]).sum(1) / KP], 1).astype(np.float32)


def prop_feats(b: np.ndarray, s: np.ndarray, ff: np.ndarray) -> np.ndarray:
    """Đặc trưng cho đề xuất #2..#10 (N,KP-1,F): quan hệ với top-1 + đặc trưng frame."""
    n = len(b); pi = box_iou(b[:, 1:, None], b[:, None, :])                     # (N,9,KP)
    sup = (s[:, None, :] * (pi > 0.5)).sum(2) / np.maximum(s.sum(1, keepdims=True), 1e-6)
    c = (b[:, :, :2] + b[:, :, 2:]) / 2; d1 = np.hypot(*(c[:, 1:] - c[:, :1]).transpose(2, 0, 1)) / W
    w = b[:, 1:, 2] - b[:, 1:, 0]; h = b[:, 1:, 3] - b[:, 1:, 1]
    own = np.stack([s[:, 1:] / np.maximum(s[:, :1], 1e-6), np.broadcast_to(np.arange(1, KP) / KP, (n, KP - 1)),
                    box_iou(b[:, 1:], b[:, :1]), sup, d1, w * h / W ** 2], -1)
    return np.concatenate([own, np.broadcast_to(ff[:, None], (n, KP - 1, ff.shape[1]))], -1).astype(np.float32)


def fit_logreg(x: np.ndarray, y: np.ndarray, wd: float = 1e-3) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    mu = x.mean(0); sd = x.std(0) + 1e-6; xt = torch.from_numpy((x - mu) / sd); yt = torch.from_numpy(y.astype(np.float32))
    m = torch.nn.Linear(x.shape[1], 1); opt = torch.optim.LBFGS(m.parameters(), max_iter=200)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6))

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xt).squeeze(-1), yt, pos_weight=pw) + \
            wd * (m.weight ** 2).sum()
        loss.backward(); return loss
    opt.step(closure); return m, mu, sd


def predict(model: tuple, x: np.ndarray) -> np.ndarray:
    m, mu, sd = model
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy((x - mu) / sd)).squeeze(-1)).numpy()


def decide(pk: np.ndarray, tau: float) -> np.ndarray:
    """pk (N,9): P(đổi sang #k sửa được). Trả chỉ số đề xuất dùng (0 = giữ top-1)."""
    best = pk.argmax(1); return np.where(pk.max(1) > tau, best + 1, 0)


def outcome(iou: np.ndarray, choice: np.ndarray) -> dict[str, float]:
    i0 = iou[:, 0]; i1 = iou[np.arange(len(iou)), choice]; h0 = i0 >= HIT; h1 = i1 >= HIT
    return {'n': int(len(iou)), 'dr_before': float(h0.mean()), 'dr_after': float(h1.mean()), 'fixed': int((~h0 & h1).sum()),
            'broken': int((h0 & ~h1).sum()), 'net': int((~h0 & h1).sum() - (h0 & ~h1).sum()), 'switched': int((choice > 0).sum()),
            'n_miss0': int((i0 == 0).sum()), 'fixed_miss0': int(((i0 == 0) & h1).sum()),
            'iou_before': float(i0.mean()), 'iou_after': float(i1.mean())}


def main() -> None:
    h = h5py.File('data/vg_frames.h5', 'r'); m = h['hasgt'][...] > 0
    b = h['boxes'][...][m]; s = h['scores'][...][m]; gt = h['gt'][...][m]; split = h['split'][...][m]
    fold = h['fold'][...][m]; vid = h['vid'][...].astype(str)[m]
    iou = box_iou(b, gt[:, None, :]); ff0 = frame_feats(b, s, vid); pf0 = prop_feats(b, s, ff0)
    ts, nnb = temporal_support(h, np.where(m)[0])
    fft = np.concatenate([ff0, ts[:, :1], (ts[:, 1:].max(1) / np.maximum(ts[:, 0], 1e-3))[:, None].clip(0, 20),
                          (nnb / (2 * R_NB))[:, None]], 1)
    pft = np.concatenate([prop_feats(b, s, fft), ts[:, 1:, None], (ts[:, 1:] / np.maximum(ts[:, :1], 1e-3))[..., None].clip(0, 20)], -1)
    ymiss = iou[:, 0] < HIT; yfix = (iou[:, 1:] >= HIT) & ymiss[:, None]
    tr = np.where(split == 0)[0]; te = np.where(split == 1)[0]
    print('frame bệnh có GT: train (OOF) %d, tỉ lệ top-1 sai %.0f%% | test %d, tỉ lệ top-1 sai %.0f%%' % (
        len(tr), 100 * ymiss[tr].mean(), len(te), 100 * ymiss[te].mean()))
    print('trần: ca sai có đề xuất đúng trong #2..#10 — train %d/%d, test %d/%d' % (
        yfix[tr].any(1).sum(), ymiss[tr].sum(), yfix[te].any(1).sum(), ymiss[te].sum()))
    allres: dict = {}
    for name, ff, pf in (('logistic, không thời gian', ff0, pf0), ('logistic + thời gian', fft, pft)):
        print('\n--- %s ---' % name)
        oof_miss = np.zeros(len(b)); oof_pk = np.zeros((len(b), KP - 1))
        for k in range(4):
            a = tr[fold[tr] != k]; v = tr[fold[tr] == k]
            oof_miss[v] = predict(fit_logreg(ff[a], ymiss[a]), ff[v])
            mk = fit_logreg(pf[a].reshape(-1, pf.shape[-1]), yfix[a].reshape(-1))
            oof_pk[v] = predict(mk, pf[v].reshape(-1, pf.shape[-1])).reshape(len(v), KP - 1)
        res: dict = {'auc_miss_oof': roc_auc_score(ymiss[tr], oof_miss[tr])}
        cv = {float(t): outcome(iou[tr], decide(oof_pk[tr], t)) for t in TAUS}
        tau = max(cv, key=lambda t: (cv[t]['net'], t)); res['tau'] = tau; res['cv_at_tau'] = cv[tau]
        mm = fit_logreg(ff[tr], ymiss[tr]); res['auc_miss_test'] = roc_auc_score(ymiss[te], predict(mm, ff[te]))
        mk = fit_logreg(pf[tr].reshape(-1, pf.shape[-1]), yfix[tr].reshape(-1))
        pk_te = predict(mk, pf[te].reshape(-1, pf.shape[-1])).reshape(len(te), KP - 1)
        res['test'] = outcome(iou[te], decide(pk_te, tau)); res['test_by_tau'] = {float(t): outcome(iou[te], decide(pk_te, t))
                                                                                 for t in TAUS}
        res['test_oracle_switch'] = outcome(iou[te], np.where(yfix[te].any(1), yfix[te].argmax(1) + 1, 0))
        print('AUC nhận ra "top-1 đang sai": OOF %.3f | test %.3f' % (res['auc_miss_oof'], res['auc_miss_test']))
        c = res['cv_at_tau']; print('τ chọn trên OOF (CV theo video) = %.2f -> OOF: sửa %d, phá %d, ròng %+d (đổi %d frame)' % (
            tau, c['fixed'], c['broken'], c['net'], c['switched']))
        for lab, o in (('TEST, τ đã chọn', res['test']), ('TEST, đổi hoàn hảo (dùng GT)', res['test_oracle_switch'])):
            print('%-30s DR %.1f%% -> %.1f%% | sửa %d, phá %d, ròng %+d | đổi %d frame | ca IoU=0 sửa %d/%d' % (
                lab, 100 * o['dr_before'], 100 * o['dr_after'], o['fixed'], o['broken'], o['net'], o['switched'],
                o['fixed_miss0'], o['n_miss0']))
        print('TEST theo τ (chỉ để tham khảo, KHÔNG dùng để chọn):',
              ' '.join('%.2f:%+d' % (t, o['net']) for t, o in res['test_by_tau'].items()))
        allres[name] = res
    print('\n--- luật không học: đổi sang đề xuất được lân cận ủng hộ nhất nếu ủng hộ > ρ × ủng hộ của top-1 ---')
    def rule(ix: np.ndarray, rho: float) -> np.ndarray:
        k = ts[ix, 1:].argmax(1); return np.where(ts[ix, 1:].max(1) > rho * np.maximum(ts[ix, 0], 1e-3), k + 1, 0)
    rhos = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
    cvr = {r: outcome(iou[tr], rule(tr, r)) for r in rhos}; rho = max(cvr, key=lambda r: (cvr[r]['net'], r))
    o = outcome(iou[te], rule(te, rho)); allres['luật thời gian'] = {'rho': rho, 'oof': cvr[rho], 'test': o,
                                                                     'test_by_rho': {r: outcome(iou[te], rule(te, r)) for r in rhos}}
    print('ρ chọn trên OOF = %.1f -> OOF ròng %+d | TEST DR %.1f%% -> %.1f%% | sửa %d, phá %d, ròng %+d | ca IoU=0 sửa %d/%d' % (
        rho, cvr[rho]['net'], 100 * o['dr_before'], 100 * o['dr_after'], o['fixed'], o['broken'], o['net'], o['fixed_miss0'], o['n_miss0']))
    print('TEST theo ρ (tham khảo):', ' '.join('%.1f:%+d' % (r, x['net']) for r, x in allres['luật thời gian']['test_by_rho'].items()))
    res = allres
    json.dump(res, open('data/vg_missgate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
