"""Bám box qua frame (2 lượt) cho agent sửa box khoanh sai (vg_fixmiss3.py, cấu hình có --temporal).
  Lượt 1: agent sửa box trên từng frame (tín hiệu thời gian từ ĐỀ XUẤT GỐC của detector ở frame lân cận).
  Lượt 2: dựng lại bản đồ nhiệt + độ ủng hộ: ở mỗi frame lân cận có GT, đề xuất top-1 được THAY bằng BOX ĐÃ SỬA ở
          lượt 1 (score tương đối 1, các đề xuất khác giữ nguyên) -> chạy lại agent. Frame lân cận mà agent không động
          vào cho đặc trưng y hệt lượt 1, nên đầu vào không lệch phân phối so với lúc train.
Dùng lượt 2 hay không: quyết định trên VALIDATION (model của fold k chạy trên fold video k của nó, 4 fold), chỉ sau
đó mới áp lên TEST (ensemble 4 model, δ đã chọn). Không nhìn test để quyết định.
Chạy: python vg_track.py <tag> [net|wnet]  -> data/vg_track_<tag>.json"""
from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import torch

import vg_fixmiss as fx
import vg_fixmiss3 as f3

DEV = f3.DEV


def neighbor_rows(d: f3.Data3) -> tuple[np.ndarray, np.ndarray]:
    """Cho mỗi frame refine: chỉ số dòng (trong vg_frames.h5) của tối đa NB frame lân cận (±R), cùng thứ tự fx._temporal."""
    h = h5py.File('data/vg_frames.h5', 'r'); vid = h['vid'][...].astype(str); fn = h['fn'][...]
    start: dict[str, int] = {}; end: dict[str, int] = {}
    for i, v in enumerate(vid):
        start.setdefault(v, i); end[v] = i + 1
    nb = np.full((len(d.row), fx.NB), -1, np.int64)
    for r, i in enumerate(d.row):
        j = np.arange(max(start[vid[i]], i - 3 * fx.R_NB), min(end[vid[i]], i + 3 * fx.R_NB + 1))
        j = j[(np.abs(fn[j] - fn[i]) <= fx.R_NB) & (j != i)]; j = j[np.argsort(np.abs(fn[j] - fn[i]), kind='stable')][:fx.NB]
        nb[r, :len(j)] = j
    return nb, vid


def rebuild(d: f3.Data3, nbrow: np.ndarray, orig: dict, fixed: dict[int, np.ndarray], ix: np.ndarray) -> None:
    """Dựng lại nbp/nbs/heat/heat14/tsup cho các frame ix: đề xuất top-1 của frame lân cận -> box đã sửa (nếu có)."""
    pos = {int(r): i for i, r in enumerate(d.row)}; ti = torch.from_numpy(ix).to(DEV)
    nbp = orig['nbp'][ti].clone(); nbs = orig['nbs'][ti].clone()
    for a, r in enumerate(ix):
        for s_, g in enumerate(nbrow[r]):
            if g >= 0 and pos.get(int(g), -1) in fixed:
                nbp[a, s_, 0] = torch.from_numpy(fixed[pos[int(g)]]).to(DEV); nbs[a, s_, 0] = 1.0
    d.nbp[ti] = nbp; d.nbs[ti] = nbs
    for c0 in range(0, len(ix), 128):
        tj = ti[c0:c0 + 128]
        c = fx.cover28(d.nbp[tj]) * (d.nbs[tj] * d.nbm[tj][:, :, None])[..., None, None]
        hh = c.sum((1, 2)) / d.nbm[tj].sum(1).clamp_min(1)[:, None, None]
        d.heat[tj] = (hh / (hh.amax((1, 2), keepdim=True) + 1e-6)).half()[:, None]
        d.heat14[tj] = torch.nn.functional.avg_pool2d(d.heat[tj].float(), 2).half()
        d.tsup[tj] = fx.tsupport(d, tj, d.props[tj])


def two_pass(d: f3.Data3, nets: list, ix: np.ndarray, delta: float, nbrow: np.ndarray, orig: dict) -> tuple[dict, dict]:
    tr1, st1, jp1 = f3.rollout(d, nets, ix, delta); m1 = fx.metrics(d, ix, tr1[-1], st1, jp1)
    fixed = {int(i): tr1[-1][a].cpu().numpy() for a, i in enumerate(ix)}
    rebuild(d, nbrow, orig, fixed, ix)
    tr2, st2, jp2 = f3.rollout(d, nets, ix, delta); m2 = fx.metrics(d, ix, tr2[-1], st2, jp2)
    rebuild(d, nbrow, orig, {}, ix)                                   # khôi phục đặc trưng gốc
    return m1, m2


def main() -> None:
    tag = sys.argv[1]; crit = sys.argv[2] if len(sys.argv) > 2 else 'net'
    ck = torch.load('data/vg_fixmiss3_%s_%s.pth' % (tag, crit), map_location=DEV, weights_only=False); a = ck['args']
    assert a['temporal'] > 0, 'cần cấu hình có --temporal'
    d = f3.Data3(a['temporal'], a['attn'], a['beta']); nbrow, _ = neighbor_rows(d)
    orig = {'nbp': d.nbp.clone(), 'nbs': d.nbs.clone()}
    nets = []
    for sd in ck['states']:
        n = f3.Net3(d).to(DEV); n.load_state_dict(sd); nets.append(n.eval())
    res: dict = {'tag': tag, 'crit': crit, 'delta': ck['delta'], 'it': ck['it'], 'val': {}}
    v1 = v2 = 0
    for k, net in enumerate(nets):
        d.set_fold(k); m1, m2 = two_pass(d, [net], d.va, ck['delta'], nbrow, orig)
        res['val'][k] = {'pass1': m1, 'pass2': m2}; v1 += m1['net']; v2 += m2['net']
        print('[%s] VAL fold %d: lượt 1 sửa %d phá %d ròng %+d | lượt 2 (bám box) sửa %d phá %d ròng %+d' % (
            tag, k, m1['fixed'], m1['broken'], m1['net'], m2['fixed'], m2['broken'], m2['net']), flush=True)
    use2 = v2 > v1; res['use_pass2'] = bool(use2)
    print('[%s] VAL tổng ròng: lượt 1 %+d | lượt 2 %+d -> %s' % (tag, v1, v2, 'DÙNG lượt 2' if use2 else 'giữ lượt 1'))
    t1, t2 = two_pass(d, nets, d.te, ck['delta'], nbrow, orig); res['test'] = {'pass1': t1, 'pass2': t2}
    for lab, m in (('lượt 1', t1), ('lượt 2 (bám box)', t2)):
        print('[%s] TEST %-17s DR %.1f%% -> %.1f%% | sửa %d phá %d ròng %+d | IoU=0 cứu %d/%d' % (
            tag, lab, 100 * m['dr_before'], 100 * m['dr_after'], m['fixed'], m['broken'], m['net'], m['fixed_miss0'],
            m['n_miss0']), flush=True)
    json.dump(res, open('data/vg_track_%s.json' % tag, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
