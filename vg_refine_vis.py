"""Vẽ quá trình học box refinement trên lưới (gọi từ vg_refine.py ở mỗi mốc train, hoặc chạy lại từ file snapshot).
  - fig_refine_progress_<method>.png : hàng = mốc train, cột = frame test cố định; GT, box detector ban đầu,
                                       từng bước dịch chuyển của agent, box cuối.
  - fig_refine_steps_<method>.png    : 1 episode của model cuối, mỗi cột là 1 bước.
  - fig_refine_curve.png             : TEST IoU sau refine theo số vòng train (DQN vs bắt chước) + mốc detector/oracle.
Chạy lại riêng: python vg_refine_vis.py  (đọc data/vg_refine_vis_<method>.npz)"""
from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

W = 336
GT_C, START_C, FINAL_C, STEP_C = '#1baf7a', '#eda100', '#2a78d6', '#ffffff'
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _box(ax: plt.Axes, b: np.ndarray, color: str, lw: float, ls: str = '-', alpha: float = 1.0) -> None:
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw, ls=ls, alpha=alpha))


def _panel(ax: plt.Axes, img: np.ndarray, gt: np.ndarray, traj: np.ndarray, steps: int, label: bool = True) -> None:
    """traj (T+1,4): box ở mỗi bước; steps = số bước agent thực sự đi."""
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    _box(ax, gt, GT_C, 2.2)
    _box(ax, traj[0], START_C, 1.8, '--')
    for k in range(1, steps):
        _box(ax, traj[k], STEP_C, 0.9, alpha=0.35 + 0.5 * k / max(steps, 1))
    if steps > 0:
        _box(ax, traj[steps], FINAL_C, 2.4)
    if label:
        ax.set_title('IoU %.2f→%.2f · %d bước' % (iou(traj[0], gt), iou(traj[steps], gt), steps), fontsize=7,
                     color=INK2, pad=2)


def _legend(fig: plt.Figure, y: float) -> None:
    h = [Line2D([], [], color=GT_C, lw=2.2), Line2D([], [], color=START_C, lw=1.8, ls='--'),
         Line2D([], [], color=MUTED, lw=0.9), Line2D([], [], color=FINAL_C, lw=2.4)]
    fig.legend(h, ['GT', 'box detector (bắt đầu)', 'các bước dịch chuyển', 'box cuối của agent'], loc='upper center',
               ncol=4, frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, y))


def load_images(paths: list[str]) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert('RGB').resize((W, W))) for p in paths]


def render_progress(snap: dict, imgs: list[np.ndarray], out: str, title: str, max_rows: int = 6,
                    metric: str = 'TEST IoU') -> None:
    """snap: its (K,), traj (K,T+1,B,4), steps (K,B), gt (B,4), iou_test (K,)."""
    k = len(snap['its']); rows = sorted(set(np.linspace(0, k - 1, min(max_rows, k)).round().astype(int).tolist()))
    b = snap['gt'].shape[0]; hgt = 2.2 * len(rows) + 0.8
    fig, axes = plt.subplots(len(rows), b, figsize=(1.9 * b + 1.5, hgt), dpi=110, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, ki in enumerate(rows):
        for c in range(b):
            _panel(axes[r, c], imgs[c], snap['gt'][c], snap['traj'][ki][:, c], int(snap['steps'][ki][c]))
        axes[r, 0].set_ylabel('vòng %d\n%s %.3f' % (snap['its'][ki], metric, snap['iou_test'][ki]), fontsize=8,
                              color=INK, rotation=0, ha='right', va='center', labelpad=8)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left', y=1 - 0.08 / hgt, va='top')
    _legend(fig, 1 - 0.35 / hgt)
    fig.subplots_adjust(left=1.5 / (1.9 * b + 1.5), right=0.995, bottom=0.01, top=1 - 0.8 / hgt, wspace=0.04, hspace=0.2)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def render_steps(snap: dict, imgs: list[np.ndarray], out: str, title: str, frames: tuple[int, ...] = (0, 1, 2),
                 max_steps: int = 8) -> None:
    """Model cuối: mỗi hàng 1 frame, mỗi cột 1 bước (bước 0 = box detector)."""
    traj = snap['traj'][-1]; steps = snap['steps'][-1]
    ncol = min(max_steps, int(max(steps[list(frames)]))) + 1; hgt = 2.1 * len(frames) + 0.8
    fig, axes = plt.subplots(len(frames), ncol, figsize=(1.75 * ncol, hgt), dpi=110, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, c in enumerate(frames):
        sc = int(steps[c])
        for s in range(ncol):
            ax = axes[r, s]; ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            _box(ax, snap['gt'][c], GT_C, 2.2)
            k = min(s, sc)
            _box(ax, traj[k, c], START_C if k == 0 else FINAL_C, 2.2, '--' if k == 0 else '-')
            ax.set_title('bước %d · IoU %.2f%s' % (s, iou(traj[k, c], snap['gt'][c]), ' (STOP)' if s > sc else ''),
                         fontsize=7, color=INK2 if s <= sc else MUTED, pad=2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left', y=1 - 0.08 / hgt, va='top')
    _legend(fig, 1 - 0.35 / hgt)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=1 - 0.8 / hgt, wspace=0.04, hspace=0.2)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def render_curve(logs: dict[str, dict], ref: dict[str, float], out: str) -> None:
    """logs[method] = {'its', 'iou_test'}; ref = {'detector': .., 'oracle': ..}."""
    colors = {'dqn': '#2a78d6', 'imitation': '#eb6834'}; names = {'dqn': 'DQN (RL)', 'imitation': 'Bắt chước oracle'}
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=140); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for m, lg in logs.items():
        ax.plot(np.array(lg['its']) / 1000, lg['iou_test'], color=colors[m], lw=2, marker='o', ms=4, label=names[m])
        ax.annotate('%.3f' % lg['iou_test'][-1], (lg['its'][-1] / 1000, lg['iou_test'][-1]), xytext=(6, 0),
                    textcoords='offset points', fontsize=8, color=INK2, va='center')
    for k, ls, c in (('detector', (0, (4, 3)), MUTED), ('oracle', (0, (1, 2)), INK)):
        ax.axhline(ref[k], color=c, lw=1.2, ls=ls)
        ax.text(0.0, ref[k] + 0.003, {'detector': 'box detector gốc', 'oracle': 'oracle (dùng GT)'}[k] + ' %.3f' % ref[k],
                fontsize=8, color=c)
    ax.set_xlabel('vòng train (nghìn)', fontsize=9, color=INK2); ax.set_ylabel('TEST IoU sau refine', fontsize=9, color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8); ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower right')
    ax.set_title('Box refinement trên lưới 28×28 — 1.068 frame bệnh, 5 video test', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def main() -> None:
    logs = {}; ref = None
    for m in ('dqn', 'imitation'):
        z = dict(np.load('data/vg_refine_vis_%s.npz' % m, allow_pickle=True)); imgs = load_images(list(z['paths']))
        render_progress(z, imgs, 'fig_refine_progress_%s.png' % m, 'Quá trình học — %s' % m)
        render_steps(z, imgs, 'fig_refine_steps_%s.png' % m, 'Từng bước dịch chuyển box — %s (model cuối)' % m)
        logs[m] = {'its': z['its'], 'iou_test': z['iou_test']}; ref = {'detector': float(z['ref_det']),
                                                                        'oracle': float(z['ref_oracle'])}
    render_curve(logs, ref, 'fig_refine_curve.png')


if __name__ == '__main__':
    main()
