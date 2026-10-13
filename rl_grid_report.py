"""Vẽ learning curve TEST IoU của 3 biến thể grid-attention (đọc data/rainbow_loc_grid_*_metrics.json)
-> fig_grid_learning_curve.png. Đường tham chiếu: Rainbow continuous tốt nhất cũ (BAOCAO_RAINBOW_LOC.md)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [('cursor', 6, 'Grid-cursor 6×6', '#2a78d6'),
        ('jump', 6, 'Grid-jump 6×6', '#eb6834'),
        ('jump', 8, 'Grid-jump 8×8', '#1baf7a')]
OLD_BEST = 0.116
SURFACE, GRID, AXIS, MUTED, INK2 = '#fcfcfb', '#e1e0d9', '#c3c2b7', '#898781', '#52514e'


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for mode, g, label, color in RUNS:
        m = json.load(open('data/rainbow_loc_grid_%s_g%d_metrics.json' % (mode, g)))
        ep = [c['ep'] / 1000 for c in m['curve']]; iou = [c['iou'] for c in m['curve']]
        ax.plot(ep, iou, color=color, lw=2, label=label, solid_capstyle='round')
        ax.plot(ep[-1], iou[-1], 'o', ms=6, color=color, mec=SURFACE, mew=2)
        dy = -14 if mode == 'cursor' else 0
        ax.annotate('%s — cuối %.3f' % (label, m['test']['iou']), (ep[-1], iou[-1]), xytext=(8, dy),
                    textcoords='offset points', va='center', fontsize=8, color=INK2)
    ax.axhline(OLD_BEST, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(37.8, OLD_BEST + 0.003, 'Rainbow continuous tốt nhất (cũ) 0.116', fontsize=8, color=MUTED,
            ha='right', va='bottom')
    fig.text(0.01, 0.015, 'Đường cong: 400 frame test lấy mẫu mỗi 1000 episode. Nhãn "cuối": đánh giá toàn bộ '
             '1068 frame test.', fontsize=7, color=MUTED)
    ax.set_xlabel('episode (nghìn)', color=INK2, fontsize=9); ax.set_ylabel('TEST IoU', color=INK2, fontsize=9)
    ax.set_title('Grid-attention RL: IoU trên 5 video test theo tiến trình học', fontsize=10, color='#0b0b0b',
                 loc='left')
    ax.set_ylim(0, 0.22); ax.set_xlim(0, 38)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('fig_grid_learning_curve.png', facecolor=SURFACE)


if __name__ == '__main__':
    main()
