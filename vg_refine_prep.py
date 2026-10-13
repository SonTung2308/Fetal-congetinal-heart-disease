"""Box refinement trên lưới — precompute lưới đặc trưng FPN '0' 28x28 (mỗi ô = 12 px của ảnh 336) cho mọi frame
bệnh có GT. Cùng nguồn detector với vg_precompute.py (không rò rỉ): video train -> backbone detector OOF của fold
giữ video đó lại; video test -> backbone detector v3 chính.
Chạy: python vg_refine_prep.py  -> data/vg_refine_fm.h5 (fm (M,256,28,28) fp16, row = chỉ số dòng trong vg_frames.h5)
Chẩn đoán: python vg_refine_prep.py test_oof <k>  -> lưới cho frame test bằng backbone OOF fold k"""
from __future__ import annotations

import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import vg_precompute as vp

DEV = 'cuda'; G = 28; BATCH = 16


@torch.no_grad()
@torch.no_grad()
def test_with_oof_backbone(k: int) -> None:
    """Chẩn đoán: lưới đặc trưng cho frame TEST tính bằng backbone OOF fold k (cùng 'loại' backbone với lúc train
    refiner) thay vì backbone detector chính. -> data/vg_refine_fm_test_oof<k>.h5"""
    frames = vp.all_frames(); split, fold = vp.video_split(frames)
    h = h5py.File('data/vg_frames.h5', 'r'); hasgt = h['hasgt'][...]; rows: list[tuple[int, str]] = []; i = 0
    for v in sorted(frames, key=lambda v: (fold[v], v)):
        for _, p, _, _, _ in frames[v]:
            if hasgt[i] and split[v] == 1:
                rows.append((i, p))
            i += 1
    det = vp.load_det('data/kvasir_detector_v3_oof%d.pth' % k)
    mean = torch.tensor(det.transform.image_mean, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(det.transform.image_std, device=DEV).view(1, 3, 1, 1)
    out = h5py.File('data/vg_refine_fm_test_oof%d.h5' % k, 'w')
    fm_ds = out.create_dataset('fm', (len(rows), 256, G, G), np.float16)
    out.create_dataset('row', data=np.array([r for r, _ in rows], np.int64))
    for s in range(0, len(rows), BATCH):
        x = torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((vp.W, vp.W))) for _, p in rows[s:s + BATCH]])
        fm = det.backbone((x.to(DEV) - mean) / std)['0']
        fm_ds[s:s + len(x)] = F.adaptive_avg_pool2d(fm, (G, G)).cpu().numpy().astype(np.float16)
    out.close(); print('[*] xong -> data/vg_refine_fm_test_oof%d.h5 (%d frame test)' % (k, len(rows)), flush=True)


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == 'test_oof':
        test_with_oof_backbone(int(sys.argv[2])); return
    frames = vp.all_frames(); split, fold = vp.video_split(frames)
    vids = sorted(frames, key=lambda v: (fold[v], v))
    h = h5py.File('data/vg_frames.h5', 'r')
    hv = h['vid'][...].astype(str); hfn = h['fn'][...]; hasgt = h['hasgt'][...]
    per_vid: dict[str, list[tuple[int, str]]] = {}; i = 0
    for v in vids:
        for fn, p, _, _, _ in frames[v]:
            assert hv[i] == v and hfn[i] == fn, 'thứ tự dòng không khớp vg_frames.h5'
            if hasgt[i]:
                per_vid.setdefault(v, []).append((i, p))
            i += 1
    rows = [r for v in vids for r in per_vid.get(v, [])]
    out = h5py.File('data/vg_refine_fm.h5', 'w')
    fm_ds = out.create_dataset('fm', (len(rows), 256, G, G), np.float16)
    out.create_dataset('row', data=np.array([r for r, _ in rows], np.int64))
    print('frame bệnh có GT: %d' % len(rows), flush=True)
    cur = None; det = None; j = 0
    for v in vids:
        if v not in per_vid:
            continue
        want = 'data/kvasir_detector_v3.pth' if fold[v] < 0 else 'data/kvasir_detector_v3_oof%d.pth' % fold[v]
        if want != cur:
            det = vp.load_det(want); cur = want
            mean = torch.tensor(det.transform.image_mean, device=DEV).view(1, 3, 1, 1)
            std = torch.tensor(det.transform.image_std, device=DEV).view(1, 3, 1, 1)
        rs = per_vid[v]
        for s in range(0, len(rs), BATCH):
            x = torch.stack([TF.to_tensor(Image.open(p).convert('RGB').resize((vp.W, vp.W))) for _, p in rs[s:s + BATCH]])
            fm = det.backbone((x.to(DEV) - mean) / std)['0']
            fm_ds[j:j + len(x)] = F.adaptive_avg_pool2d(fm, (G, G)).cpu().numpy().astype(np.float16); j += len(x)
    out.close(); print('[*] xong -> data/vg_refine_fm.h5 (%d frame)' % j, flush=True)


if __name__ == '__main__':
    main()
