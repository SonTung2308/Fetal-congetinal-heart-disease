"""
Dựng lại kiến trúc trích xuất đặc trưng theo đúng file .h5 của tác giả.

Khác biệt then chốt so với prepare.py + extract.py trong repo:

  - DYNAMIC : giữ TOÀN BỘ frame (h5 = NumberOfFrames - 2, chỉ bỏ 2 frame biên
              do ghép 2.5D). Không có bước lọc theo score.
  - PEDIATRIC: giữ ~33% frame (lọc `score > 0.01 or random < 0.1` như repo).

Gộp prepare + extract thành một bước, không ghi ảnh JPG trung gian
(nhanh hơn nhiều và không tốn ~800MB đĩa).

Cách dùng:
    python3 prepare_extract_khoi.py --dataset dynamic   --max-videos 200
    python3 prepare_extract_khoi.py --dataset pediatric --max-videos 200
"""
import argparse, os, random
import cv2, h5py, numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import models
from tqdm import tqdm

P = argparse.ArgumentParser()
P.add_argument('--dataset', choices=['dynamic', 'pediatric'], required=True)
P.add_argument('--max-videos', type=int, default=200)
P.add_argument('--weights', default='', help='.pth ResNet18 fine-tune; bỏ trống = ImageNet gốc')
P.add_argument('--out', default='')
P.add_argument('--sigma', type=float, default=2.0)
P.add_argument('--seed', type=int, default=42, help='seed cho bước lọc ngẫu nhiên (repo gốc thiếu)')
P.add_argument('--keep-all', type=int, default=-1,
               help='1 = giữ mọi frame, 0 = lọc, -1 = tự chọn theo dataset')
A = P.parse_args()

if A.dataset == 'dynamic':
    CSV = 'data/dynamic_echo_avi/VolumeTracings.csv'
    VID = 'data/dynamic_echo_avi/Videos'
    OUT = A.out or 'data/echonet_dynamic_repro.h5'
    KEEP_ALL = True if A.keep_all == -1 else bool(A.keep_all)
else:
    CSV = 'data/pediatric_echo_avi/A4C/VolumeTracings.csv'
    VID = 'data/pediatric_echo_avi/A4C/Videos'
    OUT = A.out or 'data/echonet_pediatric_repro.h5'
    KEEP_ALL = False if A.keep_all == -1 else bool(A.keep_all)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- ResNet18: fine-tune (nếu có) hoặc ImageNet gốc; lấy 512-d sau avgpool ----
net = models.resnet18(weights=None if A.weights else models.ResNet18_Weights.DEFAULT)
if A.weights:
    net.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
                           nn.Linear(256, 1), nn.Sigmoid())
    sd = torch.load(A.weights, map_location='cpu')
    net.load_state_dict({k.replace('base_model.', ''): v for k, v in sd.items()})
net.fc = nn.Identity()
net.eval().to(dev)

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

# ---- nhãn ED/ES từ VolumeTracings ----
df = pd.read_csv(CSV)
df['Frame'] = pd.to_numeric(df['Frame'], errors='coerce').astype('Int64')
targets = df.groupby('FileName')['Frame'].apply(
    lambda s: sorted({int(v) for v in s.dropna()})).to_dict()

random.seed(A.seed)
print(f'[*] {A.dataset} | giữ toàn bộ frame: {KEEP_ALL} | '
      f'backbone: {"fine-tune" if A.weights else "ImageNet gốc"}')

done = 0
with h5py.File(OUT, 'w') as f5:
    for vname, tf_list in tqdm(targets.items(), total=min(A.max_videos, len(targets))):
        if done >= A.max_videos:
            break
        path = os.path.join(VID, vname if vname.endswith('.avi') else vname + '.avi')
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        if len(frames) < 3:
            continue

        # ghép 2.5D [t-1, t, t+1] cho mọi t hợp lệ, kèm điểm Gaussian
        idxs, scores = [], []
        for i in range(1, len(frames) - 1):
            dmin = min(abs(i - t) for t in tf_list)
            sc = float(np.exp(-(dmin ** 2) / (2 * A.sigma ** 2)))
            if KEEP_ALL or sc > 0.01 or random.random() < 0.1:
                idxs.append(i)
                scores.append(sc)
        if not idxs:
            continue

        stack = np.stack([np.stack([frames[i - 1], frames[i], frames[i + 1]], 0) for i in idxs])
        x = torch.from_numpy(stack).float().to(dev) / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 128):
                out.append(net(x[i:i + 128]).cpu().numpy())
        feat = np.concatenate(out, 0)

        g = f5.create_group(vname.split('.')[0])
        g.create_dataset('features', data=feat, compression='gzip')
        g.create_dataset('gtscore', data=np.array(scores, np.float32), compression='gzip')
        g.create_dataset('n_frames', data=feat.shape[0])
        g.create_dataset('picks', data=np.arange(feat.shape[0], dtype=np.int32))
        done += 1

print(f'[*] xong {done} video -> {OUT}')
