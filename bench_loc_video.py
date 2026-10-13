"""Benchmark định vị end-to-end trên 5 VIDEO TEST (cùng split seed 0 với detector v3 và các bản Rainbow):
input = các frame JPEG của video -> đọc + resize 336 -> backbone -> (detector head | RL rollout) -> 1 box/frame.
Đo thời gian (CUDA đồng bộ, cache đĩa đã làm ấm, 1 lượt warm-up) + tính lại IoU / DR@0.5 cho phương pháp có
checkpoint. Phương pháp không còn checkpoint (Rainbow-A ImageNet, map-obs) chỉ đo thời gian với trọng số
ngẫu nhiên, chạy đủ 40 bước (đúng cấu hình navigation-only của chúng) — metric lấy từ BAOCAO_RAINBOW_LOC.md.
Mọi rollout RL giữ state trên GPU (triển khai kiểu deploy, như nhau cho mọi phương pháp).
Chạy: python bench_loc_video.py  -> in bảng + ghi data/bench_loc_video.json"""
from __future__ import annotations

import csv
import glob
import json
import os
import time
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align

import rl_localize_grid as rg

DEV = 'cuda'; W = 336; BATCH = 16; MAXSTEP = 40; HIST = 10; ALPHA = 0.2
PATH = {'Angiectasia', 'Blood - fresh', 'Blood - hematin', 'Erosion', 'Erythema', 'Foreign Body',
        'Lymphangiectasia', 'Polyp', 'Ulcer'}
ZC = torch.linspace(-12.0, 12.0, 51, device=DEV)


def test_videos() -> dict[str, list[tuple[str, list[float]]]]:
    paths = {os.path.basename(p): p for p in glob.glob('data/kvasir/labelled_images/**/*.jpg', recursive=True)}
    by_vid: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for r in csv.DictReader(open('data/kvasir/metadata.csv'), delimiter=';'):
        if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1', '').strip():
            xs = [int(r[k]) for k in ('x1', 'x2', 'x3', 'x4')]; ys = [int(r[k]) for k in ('y1', 'y2', 'y3', 'y4')]
            b = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(W, max(ys))]
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 4:
                by_vid[r['video_id']].append((paths[r['filename']], [float(v) for v in b]))
    o = sorted(by_vid); np.random.RandomState(0).shuffle(o)
    return {v: by_vid[v] for v in sorted(o[:max(3, int(0.25 * len(o)))])}


def load(paths: list[str]) -> torch.Tensor:
    return torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((W, W))) for p in paths])


def sync() -> float:
    torch.cuda.synchronize(); return time.perf_counter()


class ContNet(nn.Module):
    """Mạng MLP của rl_localize_rainbow.py (state = roi_align + box + lịch sử hành động)."""

    def __init__(self, sd: int, na: int) -> None:
        super().__init__(); self.na = na
        self.body = nn.Sequential(nn.Linear(sd, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.v1 = rg.Noisy(256, 256); self.v2 = rg.Noisy(256, 51)
        self.a1 = rg.Noisy(256, 256); self.a2 = rg.Noisy(256, na * 51)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x); v = self.v2(F.relu(self.v1(h))).view(-1, 1, 51)
        a = self.a2(F.relu(self.a1(h))).view(-1, self.na, 51)
        return F.log_softmax(v + a - a.mean(1, keepdim=True), 2)


class MapNet(nn.Module):
    """Mạng CNN của rl_localize_mapobs.py (state = toàn feature map + mask cửa sổ + toạ độ)."""

    def __init__(self, inc: int, na: int = 8) -> None:
        super().__init__(); self.na = na
        self.c = nn.Sequential(nn.Conv2d(inc, 64, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.v1 = rg.Noisy(64, 128); self.v2 = rg.Noisy(128, 51)
        self.a1 = rg.Noisy(64, 128); self.a2 = rg.Noisy(128, na * 51)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c(x).flatten(1); v = self.v2(F.relu(self.v1(h))).view(-1, 1, 51)
        a = self.a2(F.relu(self.a1(h))).view(-1, self.na, 51)
        return F.log_softmax(v + a - a.mean(1, keepdim=True), 2)


def cont_step(b: list[float], a: int) -> list[float]:
    w = b[2] - b[0]; h = b[3] - b[1]; aw = ALPHA * w; ah = ALPHA * h
    nb = [[b[0] + aw, b[1], b[2] + aw, b[3]], [b[0] - aw, b[1], b[2] - aw, b[3]],
          [b[0], b[1] + ah, b[2], b[3] + ah], [b[0], b[1] - ah, b[2], b[3] - ah],
          [b[0] - aw / 2, b[1] - ah / 2, b[2] + aw / 2, b[3] + ah / 2],
          [b[0] + aw / 2, b[1] + ah / 2, b[2] - aw / 2, b[3] - ah / 2],
          [b[0], b[1] + ah / 2, b[2], b[3] - ah / 2], [b[0] + aw / 2, b[1], b[2] - aw / 2, b[3]]][a]
    nb = [max(0.0, min(float(W), v)) for v in nb]
    return b if nb[2] - nb[0] < 8 or nb[3] - nb[1] < 8 else nb


@torch.no_grad()
def cont_rollout(net: ContNet, fm1: torch.Tensor, roig: int) -> tuple[list[float], int]:
    """Cửa sổ bắt đầu = cả ảnh; action 8 (nếu có) = trigger dừng."""
    fs = fm1.size(-1) / W; box = [0.0, 0.0, float(W), float(W)]; hist: list[int] = []; t = 0
    for t in range(1, MAXSTEP + 1):
        rb = torch.tensor([[0.0] + [v * fs for v in box]], device=DEV)
        rf = roi_align(fm1, rb, output_size=(roig, roig), aligned=True).reshape(-1)
        hv = torch.zeros(net.na * HIST, device=DEV)
        for i, a in enumerate(hist[-HIST:]):
            hv[i * net.na + a] = 1.0
        st = torch.cat([rf, torch.tensor([v / W for v in box], device=DEV), hv]).unsqueeze(0)
        a = int((net(st).exp() * ZC).sum(2).argmax(1)); hist.append(a)
        if a == 8:
            break
        box = cont_step(box, a)
    return box, t


@torch.no_grad()
def map_rollout(net: MapNet, fm1: torch.Tensor) -> tuple[list[float], int]:
    g = fm1.size(-1); cx = (torch.arange(g, device=DEV, dtype=torch.float32) + 0.5) * W / g
    coord = torch.stack([(cx / W).view(1, g).expand(g, g), (cx / W).view(g, 1).expand(g, g)]).unsqueeze(0)
    box = [0.0, 0.0, float(W), float(W)]
    for _ in range(MAXSTEP):
        mx = (cx >= box[0]) & (cx <= box[2]); my = (cx >= box[1]) & (cx <= box[3])
        mask = (my.view(g, 1) & mx.view(1, g)).float().view(1, 1, g, g)
        a = int((net(torch.cat([fm1, mask, coord], 1)).exp() * ZC).sum(2).argmax(1))
        box = cont_step(box, a)
    return box, MAXSTEP


def run(frames: list[tuple[str, list[float]]], feat_fn: Callable[[torch.Tensor], torch.Tensor],
        policy_fn: Callable[[torch.Tensor, int], tuple[list[float], int]]) -> dict:
    """Chạy 1 video, tách thời gian: đọc+resize / backbone(+head) / policy RL."""
    t_io = t_bb = t_pol = 0.0; ious: list[float] = []; steps: list[int] = []
    for s in range(0, len(frames), BATCH):
        chunk = frames[s:s + BATCH]
        t0 = sync(); x = load([p for p, _ in chunk]).to(DEV); t1 = sync()
        with torch.no_grad():
            feats = feat_fn(x)
        t2 = sync()
        for j, (_, gt) in enumerate(chunk):
            box, n = policy_fn(feats, j); ious.append(rg.iou(box, gt)); steps.append(n)
        t3 = sync(); t_io += t1 - t0; t_bb += t2 - t1; t_pol += t3 - t2
    return {'io': t_io, 'bb': t_bb, 'pol': t_pol, 'ious': ious, 'steps': steps}


def main() -> None:
    torch.manual_seed(0)
    vids = test_videos(); allf = [fr for v in vids.values() for fr in v]
    print('video test %d | frame %d | %s' % (len(vids), len(allf), {v: len(f) for v, f in vids.items()}))
    for p, _ in allf:
        open(p, 'rb').read()

    det = fasterrcnn_resnet50_fpn_v2(weights=None)
    det.roi_heads.box_predictor = FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features, 2)
    det.load_state_dict(torch.load('data/kvasir_detector_v3.pth', map_location='cpu')); det.eval().to(DEV)
    pool21 = nn.AdaptiveAvgPool2d((21, 21))

    def fpn21(x: torch.Tensor) -> torch.Tensor:
        return pool21(det.backbone(x)['0'])

    r18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    back18 = nn.Sequential(*list(r18.children())[:8]).eval().to(DEV)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    sd_b = torch.load('data/rainbow_loc.pth', map_location='cpu')
    na_b = sd_b['a2.wm'].shape[0] // 51; roig_b = 3
    net_b = ContNet(sd_b['body.0.weight'].shape[1], na_b).to(DEV); net_b.load_state_dict(sd_b); net_b.eval()
    net_a = ContNet(512 * 2 * 2 + 4 + 8 * HIST, 8).to(DEV).eval()
    net_bnav = ContNet(256 * roig_b * roig_b + 4 + 8 * HIST, 8).to(DEV).eval()
    net_m = MapNet(256 + 3).to(DEV).eval()

    def grid_method(path: str) -> tuple[str, Callable, Callable, bool]:
        ck = torch.load(path, map_location='cpu'); cfg = rg.Cfg(ck['G'], ck['budget'], ck['mode'])
        net_g = rg.make_net(ck['ch'], cfg).to(DEV); net_g.load_state_dict(ck['state']); net_g.eval()
        mu = ck['mu'].to(DEV).view(1, -1, 1, 1); sdv = ck['sd'].to(DEV).view(1, -1, 1, 1)

        def feat(x: torch.Tensor) -> torch.Tensor:
            return (rg.pool_grid(fpn21(x), cfg) - mu) / sdv

        def pol(f: torch.Tensor, j: int) -> tuple[list[float], int]:
            env = rg.greedy_rollout(net_g, f[j:j + 1], cfg); return env.box(), env.t

        return 'Grid-%s (MỚI, %dx%d, K=%d)' % (cfg.mode, cfg.G, cfg.G, cfg.K), feat, pol, True

    grid_paths = sorted(glob.glob('data/rainbow_loc_grid_*_g*.pth'))

    def det_feat(x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return det(list(x))

    def det_pol(o: list[dict[str, torch.Tensor]], j: int) -> tuple[list[float], int]:
        b = o[j]['boxes']; return (b[0].tolist() if len(b) else [0.0, 0.0, 0.0, 0.0]), 0

    methods: list[tuple[str, Callable, Callable, bool]] = [
        ('Detector Faster R-CNN v3 (giám sát)', det_feat, det_pol, True),
        ('Rainbow-A continuous (ResNet18-ImageNet)', lambda x: back18((x - mean) / std),
         lambda f, j: cont_rollout(net_a, f[j:j + 1], 2), False),
        ('Rainbow-B continuous (feat detector, ckpt)', fpn21,
         lambda f, j: cont_rollout(net_b, f[j:j + 1], roig_b), True),
        ('Rainbow-B continuous navigation-only (40 bước)', fpn21,
         lambda f, j: cont_rollout(net_bnav, f[j:j + 1], roig_b), False),
        ('Rainbow map-obs (toàn feature map)', fpn21, lambda f, j: map_rollout(net_m, f[j:j + 1]), False),
    ] + [grid_method(p) for p in grid_paths]
    results = []
    for name, ffn, pfn, has_metric in methods:
        run(allf[:BATCH], ffn, pfn)
        per_vid = [run(fr, ffn, pfn) for fr in vids.values()]
        io = sum(r['io'] for r in per_vid); bb = sum(r['bb'] for r in per_vid); pol = sum(r['pol'] for r in per_vid)
        ious = [v for r in per_vid for v in r['ious']]; n = len(ious); tot = io + bb + pol
        res = {'method': name, 'frames': n, 'total_s': tot, 'ms_frame': 1000 * tot / n,
               'io_ms': 1000 * io / n, 'backbone_ms': 1000 * bb / n, 'policy_ms': 1000 * pol / n,
               's_per_video': tot / len(vids), 'steps': float(np.mean([s for r in per_vid for s in r['steps']])),
               'max_video_s': max(r['io'] + r['bb'] + r['pol'] for r in per_vid),
               'iou': float(np.mean(ious)) if has_metric else None,
               'dr': float((np.array(ious) >= 0.5).mean()) if has_metric else None}
        results.append(res)
        print('%-45s | %6.1f ms/frame (io %.1f, backbone %.1f, policy %.1f) | %.1f s/video | steps %.1f | IoU %s DR %s'
              % (name, res['ms_frame'], res['io_ms'], res['backbone_ms'], res['policy_ms'], res['s_per_video'],
                 res['steps'], '%.3f' % res['iou'] if has_metric else '-',
                 '%.1f%%' % (100 * res['dr']) if has_metric else '-'), flush=True)
    json.dump({'videos': {v: len(f) for v, f in vids.items()}, 'gpu': torch.cuda.get_device_name(0),
               'results': results}, open('data/bench_loc_video.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
