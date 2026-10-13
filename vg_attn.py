"""Lưới attention "ý kiến thứ hai" — bước 2: học xác suất tổn thương cho từng ô của lưới 24x24 (patch DINOv2),
rồi đo xem nó có giúp NHẬN RA box top-1 của detector đang sai và CHỌN đề xuất thay thế hay không.
  - Mạng: CNN nhỏ trên lưới đặc trưng DINOv2 (PCA 256) -> bản đồ attention A (24x24) trong [0,1].
  - Nhãn mềm = tỉ lệ diện tích mỗi ô bị GT phủ. Học chéo 4 fold theo video (A trên video train là OOF),
    mô hình học trên toàn bộ train cho video test.
  - Đặc trưng attention cho 1 box: % khối attention nằm trong box, attention TB trong box, đỉnh trong box, đỉnh
    ngoài box, chênh lệch đỉnh ngoài − trong.
  - So khả năng nhận ra box sai (AUC) và cổng đổi đề xuất (sửa/phá/ròng, τ chọn bằng CV trên OOF) cho các tổ hợp:
    detector · +thời gian ±3 · +thời gian ±10 · +attention · đủ cả.
Chạy: python vg_attn.py  -> data/vg_attn.npz (att (M,24,24)) + data/vg_attn.json"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import vg_missgate as mg
from vg_localize import box_iou, roc_auc_score

DEV = 'cuda'; W = 336.0; P = 24; CELL = W / P; EPOCHS = 30; BATCH = 64


def cover(boxes: np.ndarray) -> np.ndarray:
    """(...,4) -> (...,P,P) tỉ lệ diện tích mỗi ô lưới 24x24 bị box phủ."""
    e = np.arange(P + 1) * CELL
    ix = np.clip(np.minimum(boxes[..., 2:3], e[1:]) - np.maximum(boxes[..., 0:1], e[:-1]), 0, None) / CELL
    iy = np.clip(np.minimum(boxes[..., 3:4], e[1:]) - np.maximum(boxes[..., 1:2], e[:-1]), 0, None) / CELL
    return (iy[..., :, None] * ix[..., None, :]).astype(np.float32)


class AttnNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(), nn.Dropout2d(0.2), nn.Conv2d(128, 64, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:, 0]


def fit_attn(x: np.ndarray, y: np.ndarray, tr: np.ndarray, seed: int) -> tuple[AttnNet, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    xs = torch.from_numpy(x[tr[:400]].astype(np.float32)); mu = xs.mean((0, 1, 2)).to(DEV); sd = xs.std((0, 1, 2)).to(DEV) + 1e-5
    net = AttnNet().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((1 - (y[tr] > 0.3).mean()) / max((y[tr] > 0.3).mean(), 1e-6), device=DEV).clamp(max=20)
    for _ in range(EPOCHS):
        net.train()
        for bi in np.array_split(rng.permutation(tr), max(1, len(tr) // BATCH)):
            xb = ((torch.from_numpy(x[np.sort(bi)].astype(np.float32)).to(DEV) - mu) / sd).permute(0, 3, 1, 2)
            yb = torch.from_numpy(y[np.sort(bi)]).to(DEV)
            loss = F.binary_cross_entropy_with_logits(net(xb), yb, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval(), mu, sd


@torch.no_grad()
def predict_attn(model: tuple, x: np.ndarray, ix: np.ndarray) -> np.ndarray:
    net, mu, sd = model; out = []
    for bi in np.array_split(ix, max(1, len(ix) // 256)):
        xb = ((torch.from_numpy(x[bi].astype(np.float32)).to(DEV) - mu) / sd).permute(0, 3, 1, 2)
        out.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(out)


def attn_feats(att: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """att (N,P,P), boxes (N,K,4) -> (N,K,5)."""
    c = cover(boxes); a = att[:, None]; tot = a.sum((2, 3)) + 1e-6
    inside = (c * a).sum((2, 3)); mass = inside / tot; mean_in = inside / (c.sum((2, 3)) + 1e-6)
    peak_in = np.where(c > 0.3, a, 0).max((2, 3)); peak_out = np.where(c > 0, 0, a).max((2, 3))
    return np.stack([mass, mean_in, peak_in, peak_out, peak_out - peak_in], -1).astype(np.float32)


def gate(name: str, ff: np.ndarray, pf: np.ndarray, iou: np.ndarray, fold: np.ndarray, tr: np.ndarray,
         te: np.ndarray) -> dict:
    """Cổng đổi đề xuất như vg_missgate (logistic, τ chọn bằng CV theo video trên OOF)."""
    ymiss = iou[:, 0] < mg.HIT; yfix = (iou[:, 1:] >= mg.HIT) & ymiss[:, None]
    oof_miss = np.zeros(len(iou)); oof_pk = np.zeros((len(iou), mg.KP - 1))
    for k in range(4):
        a = tr[fold[tr] != k]; v = tr[fold[tr] == k]
        oof_miss[v] = mg.predict(mg.fit_logreg(ff[a], ymiss[a]), ff[v])
        mk = mg.fit_logreg(pf[a].reshape(-1, pf.shape[-1]), yfix[a].reshape(-1))
        oof_pk[v] = mg.predict(mk, pf[v].reshape(-1, pf.shape[-1])).reshape(len(v), mg.KP - 1)
    cv = {float(t): mg.outcome(iou[tr], mg.decide(oof_pk[tr], t)) for t in mg.TAUS}
    tau = max(cv, key=lambda t: (cv[t]['net'], t))
    auc_te = roc_auc_score(ymiss[te], mg.predict(mg.fit_logreg(ff[tr], ymiss[tr]), ff[te]))
    mk = mg.fit_logreg(pf[tr].reshape(-1, pf.shape[-1]), yfix[tr].reshape(-1))
    pk = mg.predict(mk, pf[te].reshape(-1, pf.shape[-1])).reshape(len(te), mg.KP - 1)
    o = mg.outcome(iou[te], mg.decide(pk, tau))
    r = {'auc_oof': roc_auc_score(ymiss[tr], oof_miss[tr]), 'auc_test': auc_te, 'tau': tau, 'oof': cv[tau], 'test': o,
         'test_by_tau': {float(t): mg.outcome(iou[te], mg.decide(pk, t))['net'] for t in mg.TAUS}}
    print('%-32s AUC box sai OOF %.3f test %.3f | τ %.2f (OOF ròng %+d) | TEST sửa %3d phá %3d ròng %+4d | IoU=0 cứu %d/%d'
          % (name, r['auc_oof'], auc_te, tau, cv[tau]['net'], o['fixed'], o['broken'], o['net'], o['fixed_miss0'],
             o['n_miss0']), flush=True)
    return r


def main() -> None:
    z = h5py.File('data/vg_dino.h5', 'r'); x = z['feat'][...]; row = z['row'][...]
    h = h5py.File('data/vg_frames.h5', 'r'); m = np.where(h['hasgt'][...] > 0)[0]; assert np.array_equal(m, row)
    b = h['boxes'][...][row]; s = h['scores'][...][row]; gt = h['gt'][...][row]; split = h['split'][...][row]
    fold = h['fold'][...][row]; vid = h['vid'][...].astype(str)[row]
    y = np.clip(cover(gt), 0, 1); tr = np.where(split == 0)[0]; te = np.where(split == 1)[0]
    att = np.zeros((len(row), P, P), np.float32)
    for k in range(4):
        a = tr[fold[tr] != k]; v = tr[fold[tr] == k]; att[v] = predict_attn(fit_attn(x, y, a, k), x, v)
    att[te] = predict_attn(fit_attn(x, y, tr, 9), x, te)
    lab = (y > 0.3).reshape(len(y), -1)
    res: dict = {'patch_auc_oof': roc_auc_score(lab[tr].ravel(), att[tr].reshape(len(tr), -1).ravel()),
                 'patch_auc_test': roc_auc_score(lab[te].ravel(), att[te].reshape(len(te), -1).ravel())}
    af = attn_feats(att, b); iou = box_iou(b, gt[:, None, :])
    hit0 = iou[:, 0] >= 0.5
    print('Lưới attention 24x24 (DINOv2): AUC theo ô — OOF %.3f | test %.3f' % (res['patch_auc_oof'], res['patch_auc_test']))
    print('  %% khối attention trong box top-1 (test): ca đúng %.2f | ca sai %.2f | ca IoU=0 %.2f' % (
        af[te][hit0[te], 0, 0].mean(), af[te][~hit0[te], 0, 0].mean(), af[te][iou[te, 0] == 0, 0, 0].mean()))
    ff0 = mg.frame_feats(b, s, vid); ts3, n3 = mg.temporal_support(h, row)
    mg.R_NB = 10; ts10, n10 = mg.temporal_support(h, row); mg.R_NB = 3

    def tfe(ts: np.ndarray, nn_: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
        fe = np.concatenate([ts[:, :1], (ts[:, 1:].max(1) / np.maximum(ts[:, 0], 1e-3))[:, None].clip(0, 20),
                             (nn_ / (2 * r))[:, None]], 1)
        pe = np.concatenate([ts[:, 1:, None], (ts[:, 1:] / np.maximum(ts[:, :1], 1e-3))[..., None].clip(0, 20)], -1)
        return fe, pe
    f3, p3 = tfe(ts3, n3, 3); f10, p10 = tfe(ts10, n10, 10)
    fa = af[:, 0]; pa = np.concatenate([af[:, 1:], af[:, 1:] - af[:, :1]], -1)
    combos = {'detector': ([], []), '+ thời gian ±3': ([f3], [p3]), '+ thời gian ±10': ([f10], [p10]),
              '+ attention': ([fa], [pa]), '+ attention + thời gian ±10': ([fa, f10], [pa, p10]),
              '+ attention + thời gian ±3 + ±10': ([fa, f3, f10], [pa, p3, p10])}
    res['gates'] = {}
    for name, (fx, px) in combos.items():
        ff = np.concatenate([ff0] + fx, 1)
        pf = np.concatenate([mg.prop_feats(b, s, ff)] + px, -1)
        res['gates'][name] = gate(name, ff, pf, iou, fold, tr, te)
    np.savez('data/vg_attn.npz', att=att.astype(np.float16), row=row)
    json.dump(res, open('data/vg_attn.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
