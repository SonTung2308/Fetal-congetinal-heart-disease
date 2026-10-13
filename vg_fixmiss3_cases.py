"""Hình các ca TEST của ensemble đã chọn (vg_fixmiss3, tiêu chí net): mỗi hàng 1 loại ca, nền là ảnh + lưới attention
24x24 DINOv2 vẽ theo từng ô (ô đỏ = attention cao), GT xanh lá, box detector cam đứt nét, các bước trắng, box cuối xanh dương.
  hàng 1: detector trượt hẳn (IoU = 0) -> agent kéo về đúng (IoU >= 0.5)
  hàng 2: detector lệch (0 < IoU < 0.5) -> agent sửa đúng
  hàng 3: ca bị PHÁ (IoU >= 0.5 -> < 0.5), vẽ tất cả tới 6 ca
  hàng 4: ca trượt hẳn mà agent KHÔNG cứu được (để thấy giới hạn)
Chạy: python vg_fixmiss3_cases.py <tag>  -> fig_fixmiss3_cases_<tag>.png"""
from __future__ import annotations

import sys

import h5py
import matplotlib
import numpy as np
import torch

import vg_fixmiss3 as f3
import vg_fixmiss3_gridvis as gv
import vg_refine as vr
import vg_refine_vis as rv

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NCOL = 6


def pick(cand: np.ndarray, vid: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Chọn tối đa n ca, xoay vòng qua các video để đa dạng."""
    groups = [rng.permutation(cand[vid[cand] == v]).tolist() for v in np.unique(vid[cand])]
    out: list[int] = []
    while len(out) < n and any(groups):
        for g in groups:
            if g and len(out) < n:
                out.append(g.pop())
    return np.array(out, dtype=np.int64)


def main() -> None:
    tag = sys.argv[1]
    ck = torch.load('data/vg_fixmiss3_%s_net.pth' % tag, map_location=f3.DEV, weights_only=False); a = ck['args']
    d = f3.Data3(a['temporal'], a['attn'], a['beta']); nets = []
    for sd in ck['states']:
        n = f3.Net3(d).to(f3.DEV); n.load_state_dict(sd); nets.append(n.eval())
    tr, st, _ = f3.rollout(d, nets, d.te, ck['delta']); tr = tr.cpu().numpy(); st = st.cpu().numpy()
    tei = torch.from_numpy(d.te).to(f3.DEV); g = d.gt[tei].cpu().numpy()
    i0 = vr.iou_t(d.props[tei, 0], d.gt[tei]).cpu().numpy(); i1 = np.array([rv.iou(tr[-1, j], g[j]) for j in range(len(g))])
    vid = h5py.File('data/vg_frames.h5', 'r')['vid'][...].astype(str)[d.row][d.te]
    att = np.load('data/vg_attn.npz')['att'].astype(np.float32)[d.te]
    rng = np.random.RandomState(0); allj = np.arange(len(d.te))
    masks = [('Trượt hẳn (IoU=0)\n→ agent kéo về đúng', (i0 == 0) & (i1 >= f3.HIT)),
             ('Lệch (0<IoU<0.5)\n→ agent sửa đúng', (i0 > 0) & (i0 < f3.HIT) & (i1 >= f3.HIT)),
             ('Đúng → agent PHÁ', (i0 >= f3.HIT) & (i1 < f3.HIT)),
             ('Trượt hẳn, agent\nKHÔNG cứu được', (i0 == 0) & (i1 < f3.HIT))]
    rows = [('%s\n(%d ca)' % (name, int(m.sum())), pick(allj[m], vid, NCOL, rng)) for name, m in masks]
    paths = vr.frame_paths()
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(1.9 * NCOL + 1.7, 2.25 * len(rows) + 0.8), dpi=115, squeeze=False)
    fig.patch.set_facecolor(rv.SURFACE)
    for r, (name, js) in enumerate(rows):
        imgs = rv.load_images([paths[d.te[j]] for j in js])
        for c in range(NCOL):
            ax = axes[r, c]
            if c >= len(js):
                ax.axis('off'); continue
            j = js[c]; k = int(st[j])
            rv._panel(ax, imgs[c], g[j], tr[:, j], k, label=False)
            ax.imshow(gv.attn_rgba(att[j]), extent=(0, rv.W, rv.W, 0), interpolation='nearest')
            gv.grid_lines(ax, gv.PA, '#ffffff', 0.45, 0.3)
            ax.set_xlim(0, rv.W); ax.set_ylim(rv.W, 0)
            ax.set_title('%s · IoU %.2f→%.2f · %d bước' % (vid[j][:6], i0[j], i1[j], k), fontsize=6.5, color=rv.INK2, pad=2)
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=8,
                        color=rv.INK)
    rv._legend(fig, 1 - 0.25 / (2.25 * len(rows) + 0.8))
    fig.suptitle('Ensemble %s (vòng %d, δ=%.1f) trên TEST — ô đỏ: lưới attention DINOv2 24×24' % (tag, ck['it'], ck['delta']),
                 fontsize=10, color=rv.INK, y=1 - 0.05 / (2.25 * len(rows) + 0.8), x=0.02, ha='left')
    fig.subplots_adjust(left=1.7 / (1.9 * NCOL + 1.7), right=0.995, top=1 - 0.75 / (2.25 * len(rows) + 0.8), bottom=0.01,
                        wspace=0.05, hspace=0.22)
    fig.savefig('fig_fixmiss3_cases_%s.png' % tag, facecolor=rv.SURFACE); print('-> fig_fixmiss3_cases_%s.png' % tag)


if __name__ == '__main__':
    main()
