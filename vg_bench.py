"""Đo thời gian video -> output của pipeline video-grid trên 5 video test bệnh (TOÀN BỘ frame, gồm frame thường):
  (1) CŨ : đọc JPEG -> detector v3 -> sắp theo score top-1 -> chọn top-k frame + box top-1
  (2) MỚI: đọc JPEG -> detector v3 (top-10 đề xuất) + FPN 336 (lưới 8x8, ROI) -> re-ranker + head q̂
           -> chọn set frame (q̂ top-k / NMS thời gian / RL)
Re-ranker và head q̂ đo bằng mạng cùng kiến trúc (thời gian không phụ thuộc trọng số); policy RL nạp checkpoint.
Chạy: python vg_bench.py  -> in bảng + data/vg_bench.json"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

import vg_localize as vl
import vg_precompute as vp
import vg_select as vs

DEV = 'cuda'; BATCH = 16; BUDGET = 0.10


def sync() -> float:
    torch.cuda.synchronize(); return time.perf_counter()


def load(paths: list[str]) -> torch.Tensor:
    return torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((vp.W, vp.W))) for p in paths]).to(DEV)


@torch.no_grad()
def main() -> None:
    frames = vp.all_frames(); split, _ = vp.video_split(frames)
    te = sorted(v for v in frames if split[v] and any(f[2] for f in frames[v]))
    for v in te:
        for _, p, _, _, _ in frames[v]:
            open(p, 'rb').read()
    det = vp.load_det('data/kvasir_detector_v3.pth')
    rer = vl.Reranker(19, False).to(DEV).eval()
    qh = nn.Sequential(nn.Linear(19, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV).eval()
    pol = vs.Policy().to(DEV); pol.load_state_dict(torch.load('data/vg_select_l0.5_top1.pth', map_location=DEV,
                                                              weights_only=True)); pol.eval()
    t = {'io': 0.0, 'det': 0.0, 'feat': 0.0, 'rerank': 0.0, 'sel_old': 0.0, 'sel_new': 0.0}; n = 0
    warm = frames[te[0]][:BATCH]; vp.run_batch(det, load([p for _, p, _, _, _ in warm]))
    for v in te:
        fr = frames[v]; scores = []
        for s in range(0, len(fr), BATCH):
            t0 = sync(); x = load([p for _, p, _, _, _ in fr[s:s + BATCH]]); t1 = sync()
            outs = det(list(x)); t2 = sync()
            mean = torch.tensor(det.transform.image_mean, device=DEV).view(1, 3, 1, 1)
            std = torch.tensor(det.transform.image_std, device=DEV).view(1, 3, 1, 1)
            fm = det.backbone((x - mean) / std)['0']; grid = nn.functional.adaptive_avg_pool2d(fm, (8, 8)); t3 = sync()
            b = x.size(0); feed = {'grid': grid, 'roi': torch.zeros(b, 10, 1024, device=DEV),
                                   'scal': torch.rand(b, 10, 19, device=DEV), 'cover': torch.rand(b, 10, 8, 8, device=DEV),
                                   'valid': torch.ones(b, 10, dtype=torch.bool, device=DEV),
                                   'detlogit': torch.rand(b, 10, device=DEV)}
            rer(feed); qh(torch.rand(b, 19, device=DEV)); t4 = sync()
            scores += [float(o['scores'][0]) if len(o['scores']) else 0.0 for o in outs]
            t['io'] += t1 - t0; t['det'] += t2 - t1; t['feat'] += t3 - t2; t['rerank'] += t4 - t3
        k = max(1, int(len(fr) * BUDGET)); t5 = sync(); np.argsort(-np.array(scores))[:k]; t6 = sync()
        h = {'fn': np.array([f[0] for f in fr]), 'isp': np.array([f[2] for f in fr]),
             'hasgt': np.zeros(len(fr), np.int32), 'boxes': np.zeros((len(fr), 10, 4), np.float32),
             'scores': np.tile(np.array(scores, np.float32)[:, None], (1, 10)), 'gt': np.zeros((len(fr), 4), np.float32),
             'gfeat': np.random.randn(len(fr), 256).astype(np.float16)}
        loc = {'choice': np.zeros(len(fr), np.int64), 'qhat': np.array(scores, np.float32),
               'att': np.random.rand(len(fr), 8, 8).astype(np.float16)}
        t7 = sync(); vs.rollout(pol, vs.Video(h, np.arange(len(fr)), loc), BUDGET, 0.5, True); t8 = sync()
        t['sel_old'] += t6 - t5; t['sel_new'] += t8 - t7; n += len(fr)
    old = t['io'] + t['det'] + t['sel_old']; new = t['io'] + t['det'] + t['feat'] + t['rerank'] + t['sel_new']
    res = {'frames': n, 'videos': len(te), 'budget': BUDGET, 'ms_frame': {k: 1000 * val / n for k, val in t.items()},
           'old_ms_frame': 1000 * old / n, 'new_ms_frame': 1000 * new / n,
           'old_s_video': old / len(te), 'new_s_video': new / len(te), 'gpu': torch.cuda.get_device_name(0)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open('data/vg_bench.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
