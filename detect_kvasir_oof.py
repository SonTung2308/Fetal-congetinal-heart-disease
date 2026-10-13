"""Detector v3 OUT-OF-FOLD: cùng công thức detect_kvasir_v3.py, train trên 12/16 video train (4 fold theo video),
giữ lại 4 video để sinh dự đoán OOF sạch cho các mô-đun học chồng (re-rank box, chọn frame).
5 video test của detector v3 KHÔNG bao giờ được dùng. Lưu epoch cuối (không chọn epoch theo tập giữ lại).
Chạy: python detect_kvasir_oof.py <fold 0..3> [epochs=20]  -> data/kvasir_detector_v3_oof<fold>.pth"""
from __future__ import annotations

import csv
import glob
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

META = 'data/kvasir/metadata.csv'; ROOT = 'data/kvasir/labelled_images'; DEV = 'cuda'; W = 336; NFOLD = 4
PATH = {'Angiectasia', 'Blood - fresh', 'Blood - hematin', 'Erosion', 'Erythema', 'Foreign Body',
        'Lymphangiectasia', 'Polyp', 'Ulcer'}


def lesion_items() -> dict[str, list[tuple[str, list[float]]]]:
    paths = {os.path.basename(p): p for p in glob.glob(ROOT + '/**/*.jpg', recursive=True)}
    by_vid: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for r in csv.DictReader(open(META), delimiter=';'):
        if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1', '').strip():
            xs = [int(r[k]) for k in ('x1', 'x2', 'x3', 'x4')]; ys = [int(r[k]) for k in ('y1', 'y2', 'y3', 'y4')]
            b = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(W, max(ys))]
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 4:
                by_vid[r['video_id']].append((paths[r['filename']], b))
    return by_vid


def folds(by_vid: dict) -> tuple[list[str], list[list[str]]]:
    """Cùng split với detector v3: 5 video test đầu, 16 video train chia 4 fold xen kẽ."""
    order = sorted(by_vid); np.random.RandomState(0).shuffle(order)
    nte = max(4, int(0.25 * len(order))); tr = order[nte:]
    return order[:nte], [tr[i::NFOLD] for i in range(NFOLD)]


def jitter_scale(img: Image.Image, box: list[float]) -> tuple[Image.Image, list[float]]:
    s = 0.8 + 0.4 * random.random(); nw = int(W * s); img2 = img.resize((nw, nw))
    if nw <= W:
        canvas = Image.new('RGB', (W, W)); off = (W - nw) // 2; canvas.paste(img2, (off, off)); img = canvas
        box = [box[0] * s + off, box[1] * s + off, box[2] * s + off, box[3] * s + off]
    else:
        c = (nw - W) // 2; img = img2.crop((c, c, c + W, c + W))
        box = [box[0] * s - c, box[1] * s - c, box[2] * s - c, box[3] * s - c]
    return img, [max(0, min(W, v)) for v in box]


class DS(torch.utils.data.Dataset):
    def __init__(self, items: list, aug: bool) -> None:
        self.items = items; self.aug = aug

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p, box = self.items[i]; img = Image.open(p).convert('RGB').resize((W, W)); box = list(box)
        if self.aug:
            if random.random() < 0.5: img, box = jitter_scale(img, box)
            if random.random() < 0.5: img = TF.hflip(img); box = [W - box[2], box[1], W - box[0], box[3]]
            if random.random() < 0.3: img = TF.vflip(img); box = [box[0], W - box[3], box[2], W - box[1]]
            if random.random() < 0.5: img = TF.adjust_brightness(img, 0.7 + 0.6 * random.random())
            if random.random() < 0.5: img = TF.adjust_contrast(img, 0.7 + 0.6 * random.random())
        box = [max(0, min(W, v)) for v in box]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            box = list(self.items[i][1])
        t = {'boxes': torch.tensor([box], dtype=torch.float32), 'labels': torch.ones(1, dtype=torch.int64)}
        return TF.to_tensor(img), t


def collate(b: list) -> tuple:
    return tuple(zip(*b))


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0); ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def evaluate(net: torch.nn.Module, items: list) -> tuple[float, float]:
    net.eval(); ious = []
    dl = torch.utils.data.DataLoader(DS(items, False), batch_size=6, num_workers=4, collate_fn=collate)
    for imgs, tgts in dl:
        for o, t in zip(net([x.to(DEV) for x in imgs]), tgts):
            ious.append(iou(o['boxes'][0].cpu().tolist(), t['boxes'][0].tolist()) if len(o['boxes']) else 0.0)
    return float(np.mean(ious)), float((np.array(ious) >= 0.5).mean())


def main() -> None:
    fold = int(sys.argv[1]); epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    torch.manual_seed(fold); random.seed(fold); np.random.seed(fold)
    by_vid = lesion_items(); _, fl = folds(by_vid)
    held = set(fl[fold]); tr_items = [it for i, f in enumerate(fl) if i != fold for v in f for it in by_vid[v]]
    ho_items = [it for v in held for it in by_vid[v]]
    print('fold %d | train %d frame (%d video) | held-out %d frame %s' % (
        fold, len(tr_items), 12, len(ho_items), sorted(v[:8] for v in held)), flush=True)
    net = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=3)
    net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 2)
    net = net.to(DEV)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=5e-5, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = torch.utils.data.DataLoader(DS(tr_items, True), batch_size=6, shuffle=True, num_workers=6,
                                     collate_fn=collate, pin_memory=True)
    for ep in range(epochs):
        net.train(); tot = 0.0
        for imgs, tgts in dl:
            imgs = [x.to(DEV) for x in imgs]; tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            loss = sum(net(imgs, tgts).values()); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        msg = 'fold %d ep %2d loss %.3f' % (fold, ep + 1, tot / len(dl))
        if (ep + 1) % 5 == 0:
            mi, dr = evaluate(net, ho_items); msg += ' | held-out IoU %.3f DR %.1f%%' % (mi, dr * 100)
        print(msg, flush=True)
    torch.save(net.state_dict(), 'data/kvasir_detector_v3_oof%d.pth' % fold)
    print('[*] saved data/kvasir_detector_v3_oof%d.pth' % fold, flush=True)


if __name__ == '__main__':
    main()
