"""DETECTOR khoanh vùng tổn thương trong ảnh Kvasir (Faster R-CNN, 1 lớp 'lesion').
Nhãn bbox từ metadata (4 góc -> xmin,ymin,xmax,ymax). Chia theo video. Đo mAP@0.5 + IoU TB.
Đây là TẦNG 2 của pipeline: selector chọn frame bệnh -> detector này khoanh vị trí."""
import csv, glob, os, sys, numpy as np, torch
from collections import defaultdict
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

META = 'data/kvasir/metadata.csv'; ROOT = 'data/kvasir/labelled_images'; dev = 'cuda'
PATH = {'Angiectasia', 'Blood - fresh', 'Blood - hematin', 'Erosion', 'Erythema',
        'Foreign Body', 'Lymphangiectasia', 'Polyp', 'Ulcer'}
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
W = 336

paths = {os.path.basename(p): p for p in glob.glob(ROOT + '/**/*.jpg', recursive=True)}
by_vid = defaultdict(list)
for r in csv.DictReader(open(META), delimiter=';'):
    if r['finding_class'] in PATH and r['filename'] in paths and r.get('x1', '').strip():
        xs = [int(r[k]) for k in ('x1', 'x2', 'x3', 'x4')]
        ys = [int(r[k]) for k in ('y1', 'y2', 'y3', 'y4')]
        x0, y0, x1, y1 = max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(W, max(ys))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            by_vid[r['video_id']].append((paths[r['filename']], [x0, y0, x1, y1]))

vids = sorted(by_vid); rng = np.random.RandomState(0); order = list(vids); rng.shuffle(order)
nte = max(4, int(0.25 * len(order))); te = set(order[:nte]); tr = [v for v in order if v not in te]
tr_items = [it for v in tr for it in by_vid[v]]
te_items = [it for v in te for it in by_vid[v]]
print('ảnh có bbox: train %d | test %d (%d video test)' % (len(tr_items), len(te_items), len(te)))


class DS(torch.utils.data.Dataset):
    def __init__(s, items): s.items = items
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, box = s.items[i]
        img = TF.to_tensor(Image.open(p).convert('RGB').resize((W, W)))
        target = {'boxes': torch.tensor([box], dtype=torch.float32),
                  'labels': torch.ones(1, dtype=torch.int64)}
        return img, target


def collate(b): return tuple(zip(*b))


net = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 2)
net = net.to(dev)
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
dl = torch.utils.data.DataLoader(DS(tr_items), batch_size=8, shuffle=True, num_workers=8, collate_fn=collate)
for ep in range(EPOCHS):
    net.train(); tot = 0
    for imgs, tgts in dl:
        imgs = [x.to(dev) for x in imgs]; tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        loss = sum(net(imgs, tgts).values())
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print('epoch %d/%d loss %.3f' % (ep + 1, EPOCHS, tot / len(dl)), flush=True)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


net.eval(); ious = []; hit50 = 0
dlt = torch.utils.data.DataLoader(DS(te_items), batch_size=8, num_workers=8, collate_fn=collate)
with torch.no_grad():
    for imgs, tgts in dlt:
        out = net([x.to(dev) for x in imgs])
        for o, t in zip(out, tgts):
            gt = t['boxes'][0].tolist()
            if len(o['boxes']) == 0:
                ious.append(0); continue
            pb = o['boxes'][0].cpu().tolist()          # box điểm cao nhất
            v = iou(pb, gt); ious.append(v); hit50 += (v >= 0.5)
print('\n=== DETECTOR (test %d ảnh, %d video) ===' % (len(te_items), len(te)))
print('IoU trung bình: %.3f' % np.mean(ious))
print('Detection rate @IoU>=0.5: %.1f%%' % (hit50 / len(te_items) * 100))
torch.save(net.state_dict(), 'data/kvasir_detector.pth')
print('[*] lưu detector -> data/kvasir_detector.pth')
