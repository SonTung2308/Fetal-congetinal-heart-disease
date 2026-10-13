"""Video-grid pipeline — precompute SẠCH cho mọi frame có nhãn của 43 video.
Chọn mô hình theo split (không rò rỉ):
  - video bệnh thuộc 16 video train  -> detector OOF của fold giữ video đó lại (detect_kvasir_oof.py)
  - 5 video test bệnh + 22 video toàn bình thường -> detector v3 chính (chưa từng thấy các frame này)
Mỗi frame lưu: top-10 box đề xuất + score, lưới đặc trưng FPN '0' 8x8, đặc trưng ROI 2x2 của từng box,
đặc trưng toàn cục, GT box, nhãn bệnh, split. Input backbone được chuẩn hoá như bên trong detector.
Chạy: python vg_precompute.py  -> data/vg_frames.h5"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align

import detect_kvasir_oof as oof

DEV = 'cuda'; W = 336; KP = 10; GRID = 8; BATCH = 16; OUT = 'data/vg_frames.h5'


def all_frames() -> dict[str, list[tuple[int, str, int, list[float] | None, str]]]:
    """video -> [(frame_number, path, is_path, gt|None, class)] sắp theo frame."""
    paths = {os.path.basename(p): p for p in glob.glob(oof.ROOT + '/**/*.jpg', recursive=True)}
    by_vid: dict[str, list] = defaultdict(list)
    for r in csv.DictReader(open(oof.META), delimiter=';'):
        if r['filename'] not in paths:
            continue
        isp = int(r['finding_class'] in oof.PATH); gt = None
        if isp and r.get('x1', '').strip():
            xs = [int(r[k]) for k in ('x1', 'x2', 'x3', 'x4')]; ys = [int(r[k]) for k in ('y1', 'y2', 'y3', 'y4')]
            b = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(W, max(ys))]
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 4:
                gt = [float(v) for v in b]
        by_vid[r['video_id']].append((int(r['frame_number']), paths[r['filename']], isp, gt, r['finding_class']))
    return {v: sorted(fr, key=lambda t: (t[0], t[1])) for v, fr in by_vid.items()}


def video_split(frames: dict) -> tuple[dict[str, int], dict[str, int]]:
    """split: 0 train / 1 test. fold: detector OOF (0..3) cho MỌI video train, -1 = detector chính (chỉ video test).
    Video toàn-bình-thường ở train cũng đi qua detector OOF (chia đều 4 fold): nếu dùng detector chính, mô-đun học
    chồng sẽ học mẹo "đặc trưng kiểu detector chính => video không bệnh" (đã xảy ra: q̂ AUC test 0.358)."""
    lesion_te, fl = oof.folds(oof.lesion_items())
    normal = sorted(v for v in frames if v not in set(lesion_te) and not any(v in f for f in fl))
    np.random.RandomState(0).shuffle(normal); normal_te = set(normal[:max(3, int(0.25 * len(normal)))])
    split = {v: int(v in set(lesion_te) or v in normal_te) for v in frames}
    normal_fold = {v: i % len(fl) for i, v in enumerate(sorted(set(normal) - normal_te))}
    fold = {v: next((i for i, f in enumerate(fl) if v in f), normal_fold.get(v, -1)) for v in frames}
    return split, fold


def load_det(path: str) -> torch.nn.Module:
    det = fasterrcnn_resnet50_fpn_v2(weights=None)
    det.roi_heads.box_predictor = FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features, 2)
    det.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    det.roi_heads.score_thresh = 0.0; det.roi_heads.detections_per_img = KP
    return det.eval().to(DEV)


@torch.no_grad()
def run_batch(det: torch.nn.Module, x: torch.Tensor) -> tuple[np.ndarray, ...]:
    outs = det(list(x)); b = x.size(0)
    boxes = np.zeros((b, KP, 4), np.float32); sc = np.zeros((b, KP), np.float32)
    for j, o in enumerate(outs):
        n = len(o['scores']); boxes[j, :n] = o['boxes'].cpu().numpy(); sc[j, :n] = o['scores'].cpu().numpy()
    mean = torch.tensor(det.transform.image_mean, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(det.transform.image_std, device=DEV).view(1, 3, 1, 1)
    fm = det.backbone((x - mean) / std)['0']                                   # (B,256,84,84)
    grid = F.adaptive_avg_pool2d(fm, (GRID, GRID)); gfeat = fm.mean((2, 3))
    rb = torch.cat([torch.cat([torch.full((KP, 1), float(j), device=DEV), torch.from_numpy(boxes[j]).to(DEV)], 1)
                    for j in range(b)])
    roi = roi_align(fm, rb, output_size=(2, 2), spatial_scale=fm.size(-1) / W, aligned=True).view(b, KP, -1)
    return (boxes, sc, grid.cpu().numpy().astype(np.float16), roi.cpu().numpy().astype(np.float16),
            gfeat.cpu().numpy().astype(np.float16))


def main() -> None:
    frames = all_frames(); split, fold = video_split(frames)
    vids = sorted(frames, key=lambda v: (fold[v], v)); n = sum(len(frames[v]) for v in vids)
    print('video %d | frame %d | test video %d (lesion %d) | train lesion video %d' % (
        len(vids), n, sum(split.values()), sum(1 for v in vids if split[v] and any(f[2] for f in frames[v])),
        sum(1 for v in vids if fold[v] >= 0)), flush=True)
    h = h5py.File(OUT, 'w')
    ds = {'boxes': h.create_dataset('boxes', (n, KP, 4), np.float32),
          'scores': h.create_dataset('scores', (n, KP), np.float32),
          'grid': h.create_dataset('grid', (n, 256, GRID, GRID), np.float16),
          'roi': h.create_dataset('roi', (n, KP, 1024), np.float16),
          'gfeat': h.create_dataset('gfeat', (n, 256), np.float16)}
    meta = {'vid': [], 'fn': [], 'isp': [], 'hasgt': [], 'gt': [], 'split': [], 'fold': [], 'cls': []}
    cur = None; det = None; i0 = 0
    for v in vids:
        want = 'data/kvasir_detector_v3.pth' if fold[v] < 0 else 'data/kvasir_detector_v3_oof%d.pth' % fold[v]
        if want != cur:
            det = load_det(want); cur = want; print('model', want, flush=True)
        fr = frames[v]
        for s in range(0, len(fr), BATCH):
            chunk = fr[s:s + BATCH]
            x = torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((W, W))) for _, p, _, _, _ in chunk])
            for k, arr in zip(('boxes', 'scores', 'grid', 'roi', 'gfeat'), run_batch(det, x.to(DEV))):
                ds[k][i0 + s:i0 + s + len(chunk)] = arr
        for f_, _, isp, gt, cls in fr:
            meta['vid'].append(v); meta['fn'].append(f_); meta['isp'].append(isp); meta['hasgt'].append(int(gt is not None))
            meta['gt'].append(gt or [0.0] * 4); meta['split'].append(split[v]); meta['fold'].append(fold[v])
            meta['cls'].append(cls)
        i0 += len(fr); print('  %s split %d fold %2d | %4d frame | total %d/%d' % (v[:8], split[v], fold[v], len(fr), i0, n),
                             flush=True)
    for k, val in meta.items():
        if k in ('vid', 'cls'):
            h.create_dataset(k, data=np.array(val, dtype='S'))
        else:
            h.create_dataset(k, data=np.array(val, np.float32 if k == 'gt' else np.int32))
    h.close(); print('[*] xong ->', OUT, flush=True)


if __name__ == '__main__':
    main()
