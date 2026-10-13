"""Hình giải thích RL dịch box trên lưới (cho BAOCAO_GRID_REFINE.md), dùng model đã train bởi vg_refine.py:
  fig_refine_actions.png   : lưới 28x28 (ô 12 px), 13 hành động, và vùng agent "nhìn" (state)
  fig_refine_anatomy.png   : giải phẫu 1 episode từng bước — box, hành động, IoU, ΔIoU, reward, tổng reward,
                             và Q-value của 13 hành động ở mỗi bước (agent chọn Q cao nhất trong hành động hợp lệ)
  fig_refine_dynamics.png  : động học train theo mốc (ε, reward TB, TEST IoU, % tốt lên/tệ đi, số bước, loss)
  fig_refine_bins.png      : ΔIoU theo mức lệch ban đầu của box detector (DQN / bắt chước / oracle)
Chạy: python vg_refine_explain.py [tag=_curr] [log_dir]  -> các hình trên + data/vg_refine_explain<tag>.json"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import torch

import vg_refine as vr
import vg_refine_vis as rv

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

DEV = vr.DEV
NAMES = ['dời trái', 'dời phải', 'dời lên', 'dời xuống', 'nới cạnh trái', 'nới cạnh phải', 'nới cạnh trên',
         'nới cạnh dưới', 'thu cạnh trái', 'thu cạnh phải', 'thu cạnh trên', 'thu cạnh dưới', 'STOP']
SHORT = ['←', '→', '↑', '↓', '+T', '+P', '+Tr', '+D', '−T', '−P', '−Tr', '−D', 'STOP']
BLUE, ORANGE, INK, INK2, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'


def load_net(path: str) -> vr.QNet:
    net = vr.QNet().to(DEV); net.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    return net.eval()


@torch.no_grad()
def episode(d: vr.Data, net: vr.QNet, i: int) -> list[dict]:
    """Chạy tham lam 1 frame, ghi lại mọi thứ ở mỗi bước."""
    idx = torch.tensor([i], device=DEV); box = d.start[idx].clone(); g = d.gt[idx]
    t = torch.zeros(1, dtype=torch.long, device=DEV); last = torch.full_like(t, -1); rec = []; ret = 0.0
    for _ in range(vr.T):
        x, sc = vr.state(d, idx, box, t, last); q = net(x, sc)[0]; ok = vr.valid(last)[0]
        a = int(q.masked_fill(~ok, -1e9).argmax()); nb = vr.apply(box, torch.tensor([a], device=DEV))
        i0 = float(vr.iou_t(box, g)); i1 = float(vr.iou_t(nb, g))
        r = 0.0 if a == vr.STOP else vr.R_SCALE * (i1 - i0) - vr.STEP_COST; ret += r
        rec.append({'box': box[0].tolist(), 'next': nb[0].tolist(), 'q': q.tolist(), 'valid': ok.tolist(), 'a': a,
                    'iou': i0, 'iou_next': i1, 'r': r, 'G': ret})
        if a == vr.STOP:
            break
        box = nb; t = t + 1; last = torch.tensor([a], device=DEV)
    return rec


def draw_grid(ax: plt.Axes) -> None:
    for k in range(vr.G + 1):
        ax.axhline(k * vr.CELL, color='white', lw=0.35, alpha=0.35); ax.axvline(k * vr.CELL, color='white', lw=0.35, alpha=0.35)


def box_patch(ax: plt.Axes, b: list[float], c: str, lw: float, ls: str = '-') -> None:
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw, ls=ls))


def crop(ax: plt.Axes, boxes: list[list[float]]) -> None:
    b = np.array(boxes); x0, y0 = b[:, 0].min(), b[:, 1].min(); x1, y1 = b[:, 2].max(), b[:, 3].max()
    s = max(x1 - x0, y1 - y0) * 0.75 + 3 * vr.CELL; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(max(0, cx - s), min(vr.W, cx + s)); ax.set_ylim(min(vr.W, cy + s), max(0, cy - s))


def fig_actions(img: np.ndarray, gt: list[float], start: list[float], out: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), dpi=130); fig.patch.set_facecolor(SURF)
    b = [5 * vr.CELL, 5 * vr.CELL, 9 * vr.CELL, 8 * vr.CELL]
    titles = ['Dời cả box (4 hành động)', 'Nới từng cạnh ra (4)', 'Thu từng cạnh vào (4)',
              'State: vùng box nới 1.6× → RoIAlign 10×10']
    for k, ax in enumerate(axes[:3]):
        ax.set_facecolor('#f3f2ee'); ax.set_xlim(2 * vr.CELL, 12 * vr.CELL); ax.set_ylim(11 * vr.CELL, 2 * vr.CELL)
        for j in range(vr.G + 1):
            ax.axhline(j * vr.CELL, color=GRID, lw=0.8); ax.axvline(j * vr.CELL, color=GRID, lw=0.8)
        box_patch(ax, b, BLUE, 2.4); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; c = vr.CELL
        if k == 0:
            arrows = [((cx, cy), (cx - 1.6 * c, cy)), ((cx, cy), (cx + 1.6 * c, cy)), ((cx, cy), (cx, cy - 1.6 * c)),
                      ((cx, cy), (cx, cy + 1.6 * c))]
            box_patch(ax, [b[0] - c, b[1], b[2] - c, b[3]], ORANGE, 1.4, '--')
        elif k == 1:
            arrows = [((b[0], cy), (b[0] - c, cy)), ((b[2], cy), (b[2] + c, cy)), ((cx, b[1]), (cx, b[1] - c)),
                      ((cx, b[3]), (cx, b[3] + c))]
            box_patch(ax, [b[0] - c, b[1], b[2], b[3]], ORANGE, 1.4, '--')
        else:
            arrows = [((b[0], cy), (b[0] + c, cy)), ((b[2], cy), (b[2] - c, cy)), ((cx, b[1]), (cx, b[1] + c)),
                      ((cx, b[3]), (cx, b[3] - c))]
            box_patch(ax, [b[0], b[1], b[2] - c, b[3]], ORANGE, 1.4, '--')
        for p0, p1 in arrows:
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=12, color=INK, lw=1.4))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles[k], fontsize=9, color=INK, loc='left')
        ax.text(2.2 * vr.CELL, 10.8 * vr.CELL, 'mỗi ô = 12 px · mỗi bước dịch đúng 1 ô\nnét đứt cam = ví dụ box sau 1 hành động',
                fontsize=7, color=INK2, va='bottom')
    ax = axes[3]; ax.imshow(img); draw_grid(ax)
    w, h = start[2] - start[0], start[3] - start[1]; cx, cy = (start[0] + start[2]) / 2, (start[1] + start[3]) / 2
    ex = [cx - 0.8 * w, cy - 0.8 * h, cx + 0.8 * w, cy + 0.8 * h]
    box_patch(ax, ex, 'white', 1.6, ':'); box_patch(ax, gt, '#1baf7a', 2.0); box_patch(ax, start, '#eda100', 2.0, '--')
    crop(ax, [ex, gt, start]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles[3], fontsize=9, color=INK, loc='left')
    ax.set_xlabel('trắng chấm = vùng agent nhìn (đặc trưng FPN)\nvàng đứt = box hiện tại · xanh lá = GT (agent KHÔNG thấy)',
                  fontsize=7, color=INK2)
    fig.text(0.01, 0.01, '13 hành động = 12 dịch chuyển + STOP. Hành động đảo ngược bước trước bị cấm (vd. vừa "nới cạnh trái" '
             'thì không được "thu cạnh trái").', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out, facecolor=SURF); plt.close(fig)


def fig_anatomy(examples: list[tuple[str, np.ndarray, list[float], list[dict]]], out: str) -> None:
    ncol = max(len(r) for *_, r in examples) + 1; nrow = 2 * len(examples)
    hgt = 2.45 * nrow + 1.0
    fig = plt.figure(figsize=(2.35 * ncol, hgt), dpi=120); fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(nrow, ncol, height_ratios=[1.35, 0.8] * len(examples), hspace=0.62, wspace=0.18,
                          left=0.035, right=0.99, top=1 - 1.05 / hgt, bottom=0.03)
    for e, (title, img, gt, rec) in enumerate(examples):
        allb = [gt] + [s['box'] for s in rec] + [rec[-1]['next']]
        for k, s in enumerate(rec):
            ax = fig.add_subplot(gs[2 * e, k]); ax.imshow(img); draw_grid(ax); box_patch(ax, gt, '#1baf7a', 2.0)
            box_patch(ax, s['box'], '#eda100' if k == 0 else BLUE, 2.2, '--' if k == 0 else '-')
            if s['a'] != vr.STOP:
                box_patch(ax, s['next'], 'white', 1.2, '--')
            crop(ax, allb); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title('bước %d: %s' % (k, NAMES[s['a']]), fontsize=8, color=INK, pad=3)
            ax.set_xlabel('IoU %.2f→%.2f   r = %+.2f   Σr = %+.2f' % (s['iou'], s['iou_next'], s['r'], s['G']),
                          fontsize=7.5, color=INK2, labelpad=2)
            if k == 0:
                ax.text(-0.12, 1.2, title, transform=ax.transAxes, fontsize=10.5, color=INK, ha='left', va='bottom',
                        weight='bold')
            qa = fig.add_subplot(gs[2 * e + 1, k]); q = np.array(s['q']); ok = np.array(s['valid'])
            cols = [BLUE if j == s['a'] else ('#c3c2b7' if ok[j] else '#ffffff') for j in range(13)]
            bars = qa.bar(range(13), np.where(ok, q, 0), color=cols, edgecolor=['#c3c2b7' if not ok[j] else 'none'
                                                                                  for j in range(13)], width=0.8)
            for j in range(13):
                if not ok[j]:
                    bars[j].set_hatch('////')
                    qa.text(j, 0, '×', ha='center', va='center', fontsize=9, color='#e34948')
            qa.axhline(0, color=MUTED, lw=0.6); qa.set_xticks(range(13)); qa.set_xticklabels(SHORT, fontsize=5.5, rotation=90)
            qa.tick_params(axis='y', labelsize=6, colors=MUTED); qa.set_facecolor(SURF)
            for sp in ('top', 'right'):
                qa.spines[sp].set_visible(False)
            if k == 0:
                qa.set_ylabel('Q(s, a)', fontsize=7, color=INK2)
        ax = fig.add_subplot(gs[2 * e, len(rec)]); ax.imshow(img); draw_grid(ax); box_patch(ax, gt, '#1baf7a', 2.0)
        box_patch(ax, rec[0]['box'], '#eda100', 1.6, '--'); box_patch(ax, rec[-1]['next'], BLUE, 2.4)
        crop(ax, allb); ax.set_xticks([]); ax.set_yticks([])
        n = sum(1 for s in rec if s['a'] != vr.STOP)
        ax.set_title('KẾT QUẢ', fontsize=8, color=INK, pad=3)
        ax.set_xlabel('IoU %.2f → %.2f · %d bước\nΣr = 10×(%.2f−%.2f) − 0.05×%d = %+.2f' % (
            rec[0]['iou'], rec[-1]['iou_next'], n, rec[-1]['iou_next'], rec[0]['iou'], n, rec[-1]['G']),
            fontsize=7.5, color=INK2, labelpad=2)
        fig.add_subplot(gs[2 * e + 1, len(rec)]).axis('off')
    fig.suptitle('Giải phẫu 1 episode của DQN đã học. Hàng ảnh: xanh lá = GT, vàng đứt = box detector, xanh dương = box '
                 'hiện tại, trắng đứt = box sau hành động; lưới trắng = các ô 12 px.\nHàng cột: Q(s, a) của 13 hành động '
                 '— xanh = được chọn (Q cao nhất trong hành động hợp lệ), × đỏ = bị cấm (đảo ngược bước trước). '
                 'r = 10·ΔIoU − 0.05 (STOP: r = 0), Σr = tổng reward tích luỹ.',
                 fontsize=9, color=INK, x=0.01, ha='left', y=1 - 0.1 / hgt, va='top')
    fig.savefig(out, facecolor=SURF); plt.close(fig)


def parse_log(path: str) -> dict[str, dict[str, list[float]]]:
    pat = re.compile(r'^(dqn|imitation)\s+vòng\s+(\d+) \| (.*?) \| TEST IoU ([\d.]+) -> ([\d.]+) \| DR ([\d.]+)% -> '
                     r'([\d.]+)% \| tốt lên ([\d.]+)% tệ đi ([\d.]+)% \| bước ([\d.]+)')
    out: dict[str, dict[str, list[float]]] = {}
    for line in open(path, encoding='utf-8'):
        m = pat.match(line)
        if not m:
            continue
        r = out.setdefault(m.group(1), {k: [] for k in ('it', 'eps', 'ret', 'loss', 'iou', 'dr', 'up', 'down', 'steps')})
        extra = m.group(3); r['it'].append(int(m.group(2)))
        e = re.search(r'eps ([\d.]+)', extra); rt = re.search(r'return TB ([-\d.]+)', extra); ls = re.search(r'loss ([\d.]+)', extra)
        r['eps'].append(float(e.group(1)) if e else (1.0 if m.group(1) == 'dqn' else np.nan))
        r['ret'].append(float(rt.group(1)) if rt else np.nan); r['loss'].append(float(ls.group(1)) if ls else np.nan)
        for k, gi in (('iou', 5), ('dr', 7), ('up', 8), ('down', 9), ('steps', 10)):
            r[k].append(float(m.group(gi)))
    return out


def fig_dynamics(lg: dict, ref: dict[str, float], out: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), dpi=130); fig.patch.set_facecolor(SURF)
    dq, bc = lg['dqn'], lg['imitation']; x = np.array(dq['it']) / 1000; xb = np.array(bc['it']) / 1000
    panels = [('(a) ε — xác suất chọn hành động ngẫu nhiên (DQN)', [(x, dq['eps'], BLUE, 'DQN')], None),
              ('(b) Tổng reward TB mỗi episode trên TRAIN (DQN)', [(x[1:], dq['ret'][1:], BLUE, 'DQN')], None),
              ('(c) TEST IoU sau refine', [(x, dq['iou'], BLUE, 'DQN'), (xb, bc['iou'], ORANGE, 'bắt chước')], ref),
              ('(d) % frame test tốt lên (liền) / tệ đi (đứt) — DQN',
               [(x, dq['up'], BLUE, 'tốt lên'), (x, dq['down'], BLUE, 'tệ đi')], None),
              ('(e) Số bước dịch TB trên test', [(x, dq['steps'], BLUE, 'DQN'), (xb, bc['steps'], ORANGE, 'bắt chước')], None),
              ('(f) Loss cross-entropy (bắt chước)', [(xb[1:], bc['loss'][1:], ORANGE, 'bắt chước')], None)]
    for ax, (title, series, rf) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURF)
        for j, (xx, yy, c, name) in enumerate(series):
            ls = '--' if 'tệ đi' in name else '-'
            ax.plot(xx, yy, color=c, lw=2, ls=ls, marker='o', ms=3.5, label=name)
        if rf:
            for k, ls_, c in (('detector', (0, (4, 3)), MUTED), ('oracle', (0, (1, 2)), INK)):
                ax.axhline(rf[k], color=c, lw=1.1, ls=ls_)
                ax.text(x[-1], rf[k] + 0.005, {'detector': 'box detector', 'oracle': 'oracle'}[k] + ' %.3f' % rf[k],
                        fontsize=7, color=c, ha='right', va='bottom')
            ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc='lower right')
        elif len(series) > 1:
            ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
        ax.set_title(title, fontsize=8.5, color=INK, loc='left'); ax.grid(axis='y', color=GRID, lw=0.7)
        ax.set_axisbelow(True); ax.tick_params(colors=MUTED, labelsize=7); ax.set_xlabel('vòng train (nghìn)', fontsize=7, color=INK2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('Động học quá trình học (xuất phát "curr") — đánh giá trên 1.068 frame bệnh test mỗi 2.000 vòng',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, facecolor=SURF); plt.close(fig)


def fig_bins(res: dict, out: str) -> None:
    labels = list(res['bins']); fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=140); fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF); w = 0.26
    for j, (m, c, name) in enumerate((('dqn', BLUE, 'DQN'), ('imitation', ORANGE, 'Bắt chước'), ('oracle', '#52514e', 'Oracle'))):
        vals = [res['bins'][b][m] for b in labels]
        bars = ax.bar(np.arange(len(labels)) + (j - 1) * w, vals, w * 0.92, color=c, label=name)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + (0.006 if v >= 0 else -0.006), '%+.2f' % v, ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=7, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(['IoU đầu %s\n(%d frame)' % (b, res['bins'][b]['n']) for b in labels],
                                                          fontsize=8, color=INK2)
    ax.set_ylabel('ΔIoU TB (cuối − đầu)', fontsize=8, color=INK2); ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title('Refine giúp box lệch nhiều nhưng làm hỏng box đã tốt (test, xuất phát "curr")', fontsize=9.5,
                 color=INK, loc='left')
    fig.tight_layout(); fig.savefig(out, facecolor=SURF); plt.close(fig)


def fig_miss(d: vr.Data, net: vr.QNet, paths: list[str], out: str, n: int = 4) -> dict:
    """Trường hợp detector khoanh TRƯỢT hẳn (IoU = 0): quỹ đạo DQN + đề xuất khác trong top-10 của detector đã đúng."""
    import h5py
    from vg_localize import box_iou
    te = d.te; tei = torch.from_numpy(te).to(DEV); i0 = vr.iou_t(d.start[tei], d.gt[tei]).cpu().numpy()
    props = h5py.File('data/vg_frames.h5', 'r')['boxes'][...][d.row[te]]; gt = d.gt[tei].cpu().numpy()
    piou = box_iou(props, gt[:, None, :]); best = piou.argmax(1)
    cand = np.where((i0 == 0) & (piou.max(1) >= 0.5))[0]; pick: list[int] = []
    for v in np.unique(d.vid[te][cand]):                      # mỗi video 1 ca trước, rồi lấp thêm
        pick.append(int(cand[d.vid[te][cand] == v][0]))
    pick += [int(c) for c in cand if int(c) not in pick]; pick = pick[:n]
    traj, steps = vr.trajectory(d, net, te[pick]); traj = traj.cpu().numpy(); steps = steps.cpu().numpy()
    imgs = rv.load_images([paths[te[j]] for j in pick])
    fig, axes = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 4.6), dpi=130); fig.patch.set_facecolor(SURF)
    info = []
    for c, (ax, j) in enumerate(zip(np.atleast_1d(axes), pick)):
        ax.imshow(imgs[c]); draw_grid(ax); box_patch(ax, gt[j].tolist(), '#1baf7a', 2.2)
        box_patch(ax, props[j, 0].tolist(), '#eda100', 2.0, '--')
        for k in range(1, int(steps[c])):
            box_patch(ax, traj[k, c].tolist(), 'white', 0.8)
        box_patch(ax, traj[int(steps[c]), c].tolist(), BLUE, 2.2)
        box_patch(ax, props[j, best[j]].tolist(), '#e87ba4', 2.0, ':')
        cd = np.hypot((gt[j, 0] + gt[j, 2] - props[j, 0, 0] - props[j, 0, 2]) / 2,
                      (gt[j, 1] + gt[j, 3] - props[j, 0, 1] - props[j, 0, 3]) / 2) / vr.CELL
        fi = float(box_iou(traj[int(steps[c]), c], gt[j]))
        ax.set_xlim(-0.5, vr.W - 0.5); ax.set_ylim(vr.W - 0.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('video %s · tâm GT cách box %.0f ô' % (d.vid[te][j][:8], cd), fontsize=8, color=INK, pad=3)
        ax.set_xlabel('detector IoU 0.00 → DQN %.2f (%d bước)\nđề xuất #%d của detector: IoU %.2f' % (
            fi, int(steps[c]), best[j] + 1, piou[j, best[j]]), fontsize=8, color=INK2)
        info.append({'frame': int(te[j]), 'dist_cells': float(cd), 'dqn_iou': fi, 'best_prop': int(best[j]) + 1,
                     'best_prop_iou': float(piou[j, best[j]])})
    fig.suptitle('Detector khoanh TRƯỢT (box không chồng GT): xanh lá = GT, vàng đứt = box top-1 detector, trắng = các bước DQN, '
                 'xanh dương = box cuối DQN, hồng chấm = đề xuất khác của detector đã trùng GT', fontsize=8.5, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(out, facecolor=SURF); plt.close(fig)
    return {'n_iou0': int((i0 == 0).sum()), 'n_iou0_with_good_prop': int(len(cand)), 'cases': info}


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[2] == 'miss':
        d = vr.Data(); res = fig_miss(d, load_net('data/vg_refine_dqn%s.pth' % sys.argv[1]), vr.frame_paths(),
                                      'fig_refine_miss.png')
        json.dump(res, open('data/vg_refine_miss%s.json' % sys.argv[1], 'w'), indent=1); print(json.dumps(res, indent=1))
        return
    tag = sys.argv[1] if len(sys.argv) > 1 else '_curr'
    logdir = sys.argv[2] if len(sys.argv) > 2 else None
    d = vr.Data(); paths = vr.frame_paths()
    dqn = load_net('data/vg_refine_dqn%s.pth' % tag); bc = load_net('data/vg_refine_imitation%s.pth' % tag)
    te = d.te; tei = torch.from_numpy(te).to(DEV); i0 = vr.iou_t(d.start[tei], d.gt[tei]).cpu().numpy()
    fin = {}
    for name, net in (('dqn', dqn), ('imitation', bc), ('oracle', None)):
        traj, steps = vr.trajectory(d, net, te, oracle=net is None)
        fin[name] = (vr.iou_t(traj[-1], d.gt[tei]).cpu().numpy(), steps.cpu().numpy())
    res: dict = {'bins': {}}
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, 1.01)):
        m = (i0 >= lo) & (i0 < hi); key = '%.1f–%.1f' % (lo, min(hi, 1.0))
        res['bins'][key] = {'n': int(m.sum()), **{k: float((fin[k][0][m] - i0[m]).mean()) for k in fin}}
    fig_bins(res, 'fig_refine_bins.png')
    # 2 episode minh hoạ: (A) box lệch được sửa tốt, (B) box đã tốt bị làm hỏng; ưu tiên 2–6 bước cho dễ xem
    dA = fin['dqn'][0] - i0; st = fin['dqn'][1]; ok = (st >= 2) & (st <= 5)
    a_ix = te[np.argmax(np.where(ok & (i0 < 0.6), dA, -9))]; b_ix = te[np.argmin(np.where(ok & (i0 >= 0.7), dA, 9))]
    ex = []
    for title, i in (('(A) Box detector lệch → agent sửa tốt', int(a_ix)), ('(B) Box detector đã tốt → agent làm hỏng', int(b_ix))):
        rec = episode(d, dqn, i); img = rv.load_images([paths[i]])[0]
        ex.append((title, img, d.gt[i].tolist(), rec)); res['episode_' + title[1]] = {'frame': i, 'steps': rec}
    fig_anatomy(ex, 'fig_refine_anatomy.png')
    fig_actions(ex[0][1], ex[0][2], ex[0][3][0]['box'], 'fig_refine_actions.png')
    ref = {'detector': float(i0.mean()), 'oracle': float(fin['oracle'][0].mean())}
    if logdir:
        lg = parse_log('%s/refine%s.log' % (logdir, tag)); json.dump(lg, open('data/vg_refine_trainlog%s.json' % tag, 'w'))
    else:
        lg = json.load(open('data/vg_refine_trainlog%s.json' % tag))
    fig_dynamics(lg, ref, 'fig_refine_dynamics.png')
    json.dump(res, open('data/vg_refine_explain%s.json' % tag, 'w'), indent=1, ensure_ascii=False)
    for k, v in res['bins'].items():
        print('IoU đầu %-8s n=%4d | ΔIoU DQN %+.3f | bắt chước %+.3f | oracle %+.3f' % (k, v['n'], v['dqn'], v['imitation'],
                                                                                       v['oracle']))
    for key in ('episode_A', 'episode_B'):
        e = res[key]; print(key, 'frame', e['frame'])
        for k, s in enumerate(e['steps']):
            print('  bước %d %-14s IoU %.3f -> %.3f | r %+.3f | G %+.3f | Q chọn %.3f | Q STOP %.3f' % (
                k, NAMES[s['a']], s['iou'], s['iou_next'], s['r'], s['G'], s['q'][s['a']], s['q'][12]))


if __name__ == '__main__':
    main()
