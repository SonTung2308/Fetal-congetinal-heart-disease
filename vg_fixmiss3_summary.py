"""Tổng hợp ablation vg_fixmiss3.py (+ bám box 2 lượt vg_track.py nếu có) -> bảng markdown + hình.
Chạy: python vg_fixmiss3_summary.py  -> in bảng, fig_fixmiss3_ablation.png"""
from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

TAGS = [('A_base', 'A. gốc (v2 + mặt nạ đã thăm)'), ('B_t3', 'B. + thời gian ±3'),
        ('C_t3_attn', 'C. + thời gian ±3 + attention (state)'), ('D_t3_attn_b1', 'D. C + reward attention β=1'),
        ('E_t10_attn_b1', 'E. D nhưng thời gian ±10'), ('F_attn_b1', 'F. attention + reward β=1, không thời gian'),
        ('C_t3_attn_s1', 'C·s1. C, seed 1'), ('D_t3_attn_b1_s1', 'D·s1. D, seed 1')]


def load() -> list[tuple[str, str, dict]]:
    out = []
    for tag, name in TAGS:
        p = 'data/vg_fixmiss3_%s.json' % tag
        if os.path.exists(p):
            out.append((tag, name, json.load(open(p))))
    return out


def row(name: str, crit: str, s: dict) -> str:
    m = s['test']
    if s['fallback']:
        return '| %s | %s | — (tổng val %+.1f ≤ 0) | giữ detector | %.1f%% | 0 | 0 | **0** | 0/%d | — |' % (
            name, crit, s['score'], 100 * m['dr_before'], m['n_miss0'])
    return '| %s | %s | vòng %d, δ=%.2f (tổng val %+.1f) | %.1f%% → %.1f%% | %+.1f | %d | %d | **%+d** | %d/%d | %s |' % (
        name, crit, s['it'], s['delta'], s['score'], 100 * m['dr_before'], 100 * m['dr_after'],
        100 * (m['dr_after'] - m['dr_before']), m['fixed'], m['broken'], m['net'], m['fixed_miss0'], m['n_miss0'],
        ', '.join('%+d' % v for v in s['test_single_net']))


def main() -> None:
    res = load()
    print('| Cấu hình | Tiêu chí | Chọn trên val 4 fold | TEST DR@0.5 | Δ (điểm %) | Sửa | Phá | Ròng | IoU=0 cứu | Từng model (ròng) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for _, name, r in res:
        for crit in ('net', 'wnet'):
            print(row(name, crit, r['select'][crit]))
    print('\nTỉ lệ box sai ước lượng (EM, không nhãn) / thật trên test:')
    for tag, _, r in res:
        print('  %-14s train %.2f | test ước lượng %.2f | thật %.2f' % (tag, r['pi_train'], r['pi_test_hat'], r['pi_test_true']))
    for tag, _, _ in res:
        p = 'data/vg_track_%s.json' % tag
        if os.path.exists(p):
            t = json.load(open(p)); v1 = sum(x['pass1']['net'] for x in t['val'].values())
            v2 = sum(x['pass2']['net'] for x in t['val'].values())
            print('  bám box %-10s VAL ròng lượt1 %+d lượt2 %+d | TEST lượt1 %+d (sửa %d phá %d) lượt2 %+d (sửa %d phá %d)' % (
                tag, v1, v2, t['test']['pass1']['net'], t['test']['pass1']['fixed'], t['test']['pass1']['broken'],
                t['test']['pass2']['net'], t['test']['pass2']['fixed'], t['test']['pass2']['broken']))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.25, 1]})
    for ax, crit in zip(axes, ('net', 'wnet')):
        names = [n.split('. ')[0] for _, n, _ in res]; ms = [r['select'][crit]['test'] for _, _, r in res]
        fb = [r['select'][crit]['fallback'] for _, _, r in res]
        fixed = [0 if f else m['fixed'] for m, f in zip(ms, fb)]; broken = [0 if f else -m['broken'] for m, f in zip(ms, fb)]
        x = range(len(names))
        ax.bar(x, fixed, color='#2e7d32', label='sửa (sai→đúng)'); ax.bar(x, broken, color='#c62828', label='phá (đúng→sai)')
        for i, (a_, b_, f) in enumerate(zip(fixed, broken, fb)):
            ax.text(i, max(a_, 0) + 2, 'giữ\ndetector' if f else 'ròng %+d' % (a_ + b_), ha='center', fontsize=9)
        ax.axhline(0, color='k', lw=0.8); ax.set_xticks(list(x)); ax.set_xticklabels(names)
        ax.set_title('Chọn (vòng, δ) theo %s trên val 4 fold → TEST (ensemble 4 model)' % (
            'lợi ròng' if crit == 'net' else 'lợi ròng hiệu chỉnh tỉ lệ nền'), fontsize=10)
    ref = [('cổng logistic + attention (vg_attn.py)', json.load(open('data/vg_attn.json'))['gates']['+ attention']['test']['net'],
            '#6d6d6d')]
    if os.path.exists('data/vg_attn_rule.json'):
        ref.append(('luật attention peak_in (τ chọn trên train)', json.load(open('data/vg_attn_rule.json'))['peak_in']['test']['net'],
                    '#8e44ad'))
    for lab, v, col in ref:
        axes[0].axhline(v, color=col, ls='--', lw=1, label='%s: ròng %+d' % (lab, v))
    axes[0].set_ylim(-30, 55)
    axes[0].set_ylabel('số frame test (1068 frame, 249 box sai)'); axes[0].legend(loc='lower left', fontsize=8)
    fig.tight_layout(); fig.savefig('fig_fixmiss3_ablation.png', dpi=130); print('-> fig_fixmiss3_ablation.png')


if __name__ == '__main__':
    main()
