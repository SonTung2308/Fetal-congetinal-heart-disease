"""Vẽ RÕ HAI LƯỚI của agent vg_fixmiss3 trên các ca TEST, từng bước, kèm reward tách thành phần:
  - cột đầu: LƯỚI ATTENTION 24x24 (mỗi ô 14 px, DINOv2): màu đỏ đậm = xác suất tổn thương cao; Φ = Dice(box, attention)
  - các cột sau: LƯỚI DỊCH CHUYỂN 28x28 (mỗi ô 12 px): ô xanh nhạt = ô box đang phủ; mỗi bước ghi hành động (dịch 1 ô /
    nhảy sang đề xuất #k / STOP) và reward r = ΔGIoU − 0.02 + β·ΔΦ (STOP: +2 nếu IoU >= 0.5)
Phần A: δ đã chọn trên validation (model thật dùng). Phần B: cùng ensemble với δ = 0 (không lọc độ tự tin) để thấy các
bước DỊCH trên lưới mà ngưỡng δ đã loại bỏ.
Chạy: python vg_fixmiss3_gridvis.py <tag>  -> fig_fixmiss3_grid_<tag>.png"""
from __future__ import annotations

import sys

import matplotlib
import numpy as np
import torch

import vg_fixmiss as fx
import vg_fixmiss3 as f3
import vg_refine as vr
import vg_refine_vis as rv
from vg_refine_explain import NAMES

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NCOL = 6; PA = 24; CELL_A = rv.W / PA


def act_name(a: int) -> str:
    if a < f3.NMOVE:
        return NAMES[a]
    return 'nhảy #%d' % (a - f3.NMOVE + 1) if a < f3.STOP else 'STOP'


def grid_lines(ax: plt.Axes, n: int, color: str, alpha: float, lw: float = 0.35) -> None:
    for k in range(1, n):
        v = k * rv.W / n
        ax.axhline(v, color=color, alpha=alpha, lw=lw); ax.axvline(v, color=color, alpha=alpha, lw=lw)


def base(ax: plt.Axes, img: np.ndarray) -> None:
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, rv.W); ax.set_ylim(rv.W, 0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def attn_panel(ax: plt.Axes, img: np.ndarray, att: np.ndarray, gt: np.ndarray, b0: np.ndarray, phi: float) -> None:
    base(ax, img)
    ax.imshow(attn_rgba(att), extent=(0, rv.W, rv.W, 0), interpolation='nearest')
    grid_lines(ax, PA, '#ffffff', 0.6, 0.4)
    rv._box(ax, gt, rv.GT_C, 2.0); rv._box(ax, b0, rv.START_C, 1.8, '--')
    ax.set_title('lưới attention 24×24\nΦ(box detector) = %.2f' % phi, fontsize=6.5, color=rv.INK, pad=2)


def attn_rgba(att: np.ndarray) -> np.ndarray:
    """Ô attention thấp gần như trong suốt, ô cao đỏ đậm (không che ảnh ở vùng không có tổn thương)."""
    rgba = np.zeros((*att.shape, 4), np.float32); rgba[..., 0] = 0.85; rgba[..., 1] = 0.05; rgba[..., 2] = 0.05
    rgba[..., 3] = np.clip(att, 0, 1) ** 0.8 * 0.85
    return rgba


def step_panel(ax: plt.Axes, img: np.ndarray, gt: np.ndarray, prev: np.ndarray | None, box: np.ndarray, title: str,
               color: str) -> None:
    base(ax, img)
    cov = vr_cover(box) > 0.5
    ax.imshow(np.where(cov, 1.0, np.nan), cmap='Blues', alpha=0.35, vmin=0, vmax=1.4, extent=(0, rv.W, rv.W, 0),
              interpolation='nearest')
    grid_lines(ax, f3.G, '#ffffff', 0.5, 0.4)
    rv._box(ax, gt, rv.GT_C, 2.0)
    if prev is not None:
        rv._box(ax, prev, rv.STEP_C, 1.0, '--', alpha=0.9)
    rv._box(ax, box, color, 2.2, '--' if color == rv.START_C else '-')
    ax.set_title(title, fontsize=6.3, color=rv.INK2, pad=2, linespacing=1.15)


def vr_cover(box: np.ndarray) -> np.ndarray:
    return fx.cover28(torch.from_numpy(box[None].astype(np.float32)).to(f3.DEV))[0].cpu().numpy()


def row(axes: np.ndarray, img: np.ndarray, att: np.ndarray, gt: np.ndarray, traj: np.ndarray, acts: np.ndarray,
        beta: float, label: str) -> None:
    tt = torch.from_numpy(traj.astype(np.float32)).to(f3.DEV); gtt = torch.from_numpy(gt[None].astype(np.float32)).to(f3.DEV)
    at = torch.from_numpy(att[None]).to(f3.DEV).expand(len(tt), -1, -1)
    phi = f3.dice_phi(at, tt).cpu().numpy(); gi = fx.giou_t(tt, gtt.expand(len(tt), -1)).cpu().numpy()
    io = [rv.iou(b, gt) for b in traj]
    attn_panel(axes[0], img, att, gt, traj[0], phi[0])
    step_panel(axes[1], img, gt, None, traj[0], 'bước 0: box detector\nIoU %.2f · Φ %.2f\n' % (io[0], phi[0]), rv.START_C)
    k = int((acts != f3.STOP).sum()); col = 2
    for s in range(k):
        if col >= NCOL - 1 and s < k - 1:                      # quá nhiều bước: bỏ qua các bước giữa, giữ bước cuối
            continue
        r = gi[s + 1] - gi[s] - fx.STEP_COST + beta * (phi[s + 1] - phi[s])
        step_panel(axes[col], img, gt, traj[s], traj[s + 1], '%sbước %d: %s · IoU %.2f\nΔGIoU %+.2f − 0.02%s\nr = %+.2f' % (
            '… ' if s + 1 > col - 1 else '', s + 1, act_name(int(acts[s])), io[s + 1], gi[s + 1] - gi[s],
            (' + βΔΦ %+.2f' % (beta * (phi[s + 1] - phi[s]))) if beta > 0 else '', r), rv.FINAL_C)
        col += 1
    if col < NCOL:
        ok = io[k] >= f3.HIT
        step_panel(axes[col], img, gt, None, traj[k], 'STOP · IoU %.2f\nr = %s\n' % (io[k], '+2 (IoU ≥ 0.5)' if ok else '0 (IoU < 0.5)'),
                   rv.FINAL_C if k else rv.START_C)
        col += 1
    for c in range(col, NCOL):
        axes[c].axis('off')
    axes[0].text(-0.06, 0.5, label, transform=axes[0].transAxes, ha='right', va='center', fontsize=7.5, color=rv.INK)


def pick_one(mask: np.ndarray, order: np.ndarray, rng: np.random.RandomState) -> int | None:
    c = np.where(mask)[0]
    if len(c) == 0:
        return None
    return int(c[np.argsort(-order[c], kind='stable')][rng.randint(min(3, len(c)))])


def main() -> None:
    tag = sys.argv[1]
    ck = torch.load('data/vg_fixmiss3_%s_net.pth' % tag, map_location=f3.DEV, weights_only=False); a = ck['args']
    d = f3.Data3(a['temporal'], a['attn'], a['beta']); nets = []
    for sd in ck['states']:
        n = f3.Net3(d).to(f3.DEV); n.load_state_dict(sd); nets.append(n.eval())
    tei = torch.from_numpy(d.te).to(f3.DEV); gt = d.gt[tei].cpu().numpy(); att = d.att[tei].cpu().numpy()
    i0 = vr.iou_t(d.props[tei, 0], d.gt[tei]).cpu().numpy(); rng = np.random.RandomState(0); rows = []
    runs = {}
    for dl in (ck['delta'], 0.0):
        log: list[torch.Tensor] = []; tr, st, _ = f3.rollout(d, nets, d.te, dl, log)
        acts = torch.stack(log, 1).cpu().numpy(); trn = tr.cpu().numpy()
        i1 = np.array([rv.iou(trn[-1, j], gt[j]) for j in range(len(gt))]); runs[dl] = (trn, acts, i1)
        nmove = (acts < f3.NMOVE).sum(1); njump = ((acts >= f3.NMOVE) & (acts < f3.STOP)).sum(1)
        fixed = (i0 < f3.HIT) & (i1 >= f3.HIT); broken = (i0 >= f3.HIT) & (i1 < f3.HIT)
        print('δ=%.1f: bước dịch %d, bước nhảy %d | frame có dịch: %d (sửa %d, phá %d) | frame chỉ nhảy: %d (sửa %d, phá %d)' % (
            dl, nmove.sum(), njump.sum(), (nmove > 0).sum(), (fixed & (nmove > 0)).sum(), (broken & (nmove > 0)).sum(),
            ((njump > 0) & (nmove == 0)).sum(), (fixed & (nmove == 0)).sum(), (broken & (nmove == 0)).sum()))
        if dl == ck['delta']:
            for lab, m in (('A · δ=%.1f\ntrượt hẳn → đúng' % dl, (i0 == 0) & fixed), ('A · δ=%.1f\nlệch → đúng' % dl, (i0 > 0) & fixed),
                           ('A · δ=%.1f\nđúng → bị phá' % dl, broken)):
                rows.append((lab, pick_one(m, i1 - i0, rng), dl))
        else:
            for lab, m in (('B · δ=0\ndịch trên lưới\n→ sửa được', fixed & (nmove >= 2)),
                           ('B · δ=0\ndịch trên lưới\n→ làm hỏng', broken & (nmove >= 2)),
                           ('B · δ=0\ndịch trên lưới,\nca sai vẫn sai', (i0 < f3.HIT) & (i1 < f3.HIT) & (nmove >= 3))):
                rows.append((lab, pick_one(m, nmove.astype(float), rng), dl))
    rows = [r for r in rows if r[1] is not None]
    imgs = rv.load_images([vr.frame_paths()[d.te[j]] for _, j, _ in rows])
    h = 2.35 * len(rows) + 0.9
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(2.15 * NCOL + 1.3, h), dpi=120, squeeze=False)
    fig.patch.set_facecolor(rv.SURFACE)
    for r, (lab, j, dl) in enumerate(rows):
        trn, acts, _ = runs[dl]
        row(axes[r], imgs[r], att[j], gt[j], trn[:, j], acts[j], a['beta'], lab)
    hd = [Patch(fc='#e57373', label='lưới attention 24×24 (đỏ = xác suất tổn thương cao)'),
          Patch(fc='#9ecae1', label='ô của lưới dịch chuyển 28×28 mà box đang phủ'),
          Line2D([], [], color=rv.GT_C, lw=2, label='GT'), Line2D([], [], color=rv.START_C, lw=1.8, ls='--', label='box detector'),
          Line2D([], [], color=rv.STEP_C, lw=1, ls='--', label='box bước trước'), Line2D([], [], color=rv.FINAL_C, lw=2.2, label='box sau bước')]
    fig.legend(handles=hd, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1 - 0.3 / h))
    fig.suptitle('Hai lưới của agent %s trên TEST: attention 24×24 (Φ, reward) và lưới dịch chuyển 28×28 (hành động)' % tag,
                 fontsize=9.5, color=rv.INK, x=0.01, ha='left', y=1 - 0.05 / h, va='top')
    fig.subplots_adjust(left=1.3 / (2.15 * NCOL + 1.3), right=0.995, bottom=0.01, top=1 - 1.3 / h, wspace=0.06, hspace=0.5)
    out = 'fig_fixmiss3_grid_%s.png' % tag; fig.savefig(out, facecolor=rv.SURFACE); print('->', out)


if __name__ == '__main__':
    main()
