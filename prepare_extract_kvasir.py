"""Trich dac trung Kvasir-Capsule cho pipeline tom tat RL.
Anh RGB -> ResNet18 ImageNet 512-d (khong 2.5D).
gtscore nhi phan: 1 = ton thuong benh ly, 0 = binh thuong/moc giai phau.
Moi video = chuoi frame co nhan, sap theo frame_number.
"""
import argparse, csv, glob, os
from collections import defaultdict
import cv2, h5py, numpy as np, torch, torch.nn as nn
from torchvision import models
from tqdm import tqdm

P = argparse.ArgumentParser()
P.add_argument('--root', default='data/kvasir/labelled_images')
P.add_argument('--meta', default='data/kvasir/metadata.csv')
P.add_argument('--out', default='data/kvasir_capsule.h5')
P.add_argument('--min-frames', type=int, default=10)
A = P.parse_args()

PATHOLOGY = {'Angiectasia', 'Blood - fresh', 'Blood - hematin', 'Erosion',
             'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Polyp', 'Ulcer'}

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
net.fc = nn.Identity(); net.eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

paths = {os.path.basename(p): p for p in glob.glob(A.root + '/**/*.jpg', recursive=True)}
paths.update({os.path.basename(p): p for p in glob.glob(A.root + '/**/*.png', recursive=True)})
print('[*] tim thay', len(paths), 'anh')

rows = list(csv.DictReader(open(A.meta), delimiter=';'))
by_vid = defaultdict(list)
for r in rows:
    if r['filename'] in paths:
        by_vid[r['video_id']].append((int(r['frame_number']), paths[r['filename']],
                                       1.0 if r['finding_class'] in PATHOLOGY else 0.0))
print('[*]', len(by_vid), 'video khop anh')


def feats_of(pl):
    imgs = []
    for p in pl:
        im = cv2.imread(p)
        if im is None:
            return None
        imgs.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)))
    x = torch.from_numpy(np.stack(imgs)).float().permute(0, 3, 1, 2).to(dev) / 255.0
    x = (x - mean) / std
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            out.append(net(x[i:i + 128]).cpu().numpy())
    return np.concatenate(out, 0)


kept = 0
with h5py.File(A.out, 'w') as f5:
    for vid, items in tqdm(by_vid.items(), desc='video'):
        items.sort(key=lambda t: t[0])
        if len(items) < A.min_frames:
            continue
        feat = feats_of([p for _, p, _ in items])
        if feat is None:
            continue
        score = np.array([s for _, _, s in items], dtype=np.float32)
        g = f5.create_group(vid)
        g.create_dataset('features', data=feat, compression='gzip')
        g.create_dataset('gtscore', data=score, compression='gzip')
        g.create_dataset('n_frames', data=feat.shape[0])
        g.create_dataset('picks', data=np.arange(feat.shape[0], dtype=np.int32))
        kept += 1
print('[*] xong', kept, 'video ->', A.out)
