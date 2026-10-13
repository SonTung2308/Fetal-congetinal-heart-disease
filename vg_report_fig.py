"""Hình Pareto cho BAOCAO_VIDEO_GRID.md: độ phủ đoạn bệnh (x) vs IoU trung bình của set frame (y), 3 ngân sách.
Đọc data/vg_select_l0.5_top1.json và data/vg_select_l0.0_top1.json -> fig_vg_pareto.png"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, GRID, AXIS, MUTED, INK, INK2 = '#fcfcfb', '#e1e0d9', '#c3c2b7', '#898781', '#0b0b0b', '#52514e'


def pick(rows: dict, prefix: str) -> dict:
    return next(v for k, v in rows.items() if k.startswith(prefix))


def main() -> None:
    a = json.load(open('data/vg_select_l0.5_top1.json'))['budgets']; b = json.load(open('data/vg_select_l0.0_top1.json'))['budgets']
    series = [('Conf top-k (cũ)', '#2a78d6', 'o', lambda k: pick(a[k], 'det-conf')),
              ('q̂ top-k', '#eb6834', 's', lambda k: pick(a[k], 'q̂ top-k')),
              ('q̂ + NMS thời gian (w=20)', '#1baf7a', '^', lambda k: pick(b[k], 'q̂ + NMS')),
              ('RL λ=0.5 (thưởng phủ đoạn)', '#eda100', 'D', lambda k: pick(a[k], 'RL')),
              ('RL λ=0', '#e87ba4', 'P', lambda k: pick(b[k], 'RL')),
              ('Random', MUTED, 'x', lambda k: pick(a[k], 'random')),
              ('Oracle', INK, '*', lambda k: pick(a[k], 'oracle'))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=150, sharey=True); fig.patch.set_facecolor(SURFACE)
    for ax, k, title in zip(axes, ('0.05', '0.1', '0.2'), ('Ngân sách 5%', 'Ngân sách 10%', 'Ngân sách 20%')):
        ax.set_facecolor(SURFACE)
        for name, color, mk, get in series:
            m = get(k)
            ax.scatter(100 * m['cov'], m['iou'], s=70 if mk != '*' else 140, marker=mk, color=color, label=name,
                       edgecolors=SURFACE if mk not in ('x',) else None, linewidths=1.5, zorder=3)
        ax.set_title(title, fontsize=9, color=INK, loc='left'); ax.set_xlim(20, 102); ax.set_ylim(0, 0.56)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8); ax.set_xlabel('phủ đoạn bệnh (%)', color=INK2, fontsize=8)
    axes[0].set_ylabel('IoU trung bình của set', color=INK2, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=7, frameon=False, fontsize=7.5, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, 'TB trên 5 video test có bệnh (split sạch). Box output = box top-1 của detector cho mọi phương pháp. '
             'Góc trên-phải là tốt.', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9)); fig.savefig('fig_vg_pareto.png', facecolor=SURFACE)


if __name__ == '__main__':
    main()
