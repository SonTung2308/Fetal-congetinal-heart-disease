"""Lưới attention "ý kiến thứ hai" — bước 1: đặc trưng patch DINOv2 ViT-B/14 (tự giám sát, KHÔNG huấn luyện trên dữ
liệu nội soi, nên không bị học thuộc video train như backbone detector) cho mọi frame bệnh có GT.
Ảnh 336 px -> lưới 24x24 patch (14 px/patch) x 768 chiều -> PCA 256 chiều (PCA khớp CHỈ trên frame train).
Cùng thứ tự frame với data/vg_refine_fm.h5.
Chạy: python vg_attn_prep.py  -> data/vg_dino.h5 (feat (M,24,24,256) fp16, row, pca_mean, pca_comp)"""
from __future__ import annotations

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import vg_refine as vr

DEV = 'cuda'; W = 336; P = 24; K = 256; BATCH = 32
HUB = '/home/iec/.cache/torch/hub/facebookresearch_dinov2_main'
CKPT = '/home/iec/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def main() -> None:
    row = h5py.File('data/vg_refine_fm.h5', 'r')['row'][...]
    split = h5py.File('data/vg_frames.h5', 'r')['split'][...][row]
    paths = vr.frame_paths(); assert len(paths) == len(row)
    m = torch.hub.load(HUB, 'dinov2_vitb14', source='local', pretrained=False)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True)); m = m.to(DEV).eval()
    feats = np.zeros((len(row), P * P, 768), np.float16)
    for s in range(0, len(row), BATCH):
        x = torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((W, W))) for p in paths[s:s + BATCH]])
        feats[s:s + len(x)] = m.forward_features(((x - MEAN) / STD).to(DEV))['x_norm_patchtokens'].half().cpu().numpy()
        if s % 640 == 0:
            print('  %d/%d' % (s, len(row)), flush=True)
    tr = np.where(split == 0)[0]; rng = np.random.RandomState(0)
    samp = torch.from_numpy(feats[rng.choice(tr, min(300, len(tr)), replace=False)].reshape(-1, 768).astype(np.float32)).to(DEV)
    mu = samp.mean(0); _, sv, v = torch.linalg.svd(samp - mu, full_matrices=False); comp = v[:K]
    print('PCA %d chiều giữ %.1f%% phương sai (khớp trên %d patch của frame train)' % (
        K, 100 * float((sv[:K] ** 2).sum() / (sv ** 2).sum()), len(samp)), flush=True)
    out = h5py.File('data/vg_dino.h5', 'w'); ds = out.create_dataset('feat', (len(row), P, P, K), np.float16)
    for s in range(0, len(row), 512):
        f = torch.from_numpy(feats[s:s + 512].astype(np.float32)).to(DEV)
        ds[s:s + len(f)] = ((f - mu) @ comp.T).view(len(f), P, P, K).half().cpu().numpy()
    out.create_dataset('row', data=row); out.create_dataset('pca_mean', data=mu.cpu().numpy())
    out.create_dataset('pca_comp', data=comp.cpu().numpy()); out.close()
    print('[*] xong -> data/vg_dino.h5', flush=True)


if __name__ == '__main__':
    main()
