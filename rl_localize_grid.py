"""Grid-attention RL localization — chia frame thành lưới G x G ô; agent chọn dần các ô nghi tổn thương rồi
STOP; output = bbox hợp các ô đã chọn. Thay cho tìm bbox liên tục bằng zoom/dịch (overfit train->test,
BAOCAO_RAINBOW_LOC.md mục 3b/3c). Hai chế độ "dịch chuyển" sự chú ý:
  - cursor : con trỏ đi từng ô (lên/xuống/trái/phải) + SELECT + STOP. State EGOCENTRIC (2G-1)x(2G-1) quanh
             con trỏ -> "tổn thương ở bên trái" trông như nhau ở mọi vị trí (bất biến tịnh tiến).
  - jump   : attention nhảy thẳng tới BẤT KỲ ô nào và chọn nó (G*G hành động) + STOP; Q của mọi ô tính bằng
             một head conv dùng chung (Q-map kiểu attention), ô đã chọn bị che (mask).
State: feature backbone detector v3 (chuẩn hoá theo kênh) + kênh đã-chọn + toạ độ (+ hợp-lệ cho cursor);
2 vô hướng: tỉ lệ ngân sách đã dùng, tỉ lệ thời gian đã dùng.
Reward:
  di chuyển (cursor): R_STEP + (d_trước - d_sau), d = Chebyshev tới ô tổn thương gần nhất CHƯA chọn
                      (shaping dạng thế năng -> đi vòng/lặp không kiếm được điểm).
  SELECT            : +1 ô tổn thương mới, -0.5 ô thường, -0.5 chọn lại (chọn lại KHÔNG tiêu ngân sách).
  kết thúc          : + R_TERM * IoU(bbox hợp các ô đã chọn, GT) — tối ưu thẳng metric cuối (không khả vi).
STOP chỉ hợp lệ khi đã chọn >=1 ô (chặn trigger-collapse). Hết ngân sách K hoặc MAXSTEP cũng kết thúc.
Khám phá: Noisy Nets + epsilon-greedy giảm dần. Rainbow: Double + Dueling + PER + N-step + Noisy + C51.
Chạy: python rl_localize_grid.py <episodes> <G> <budget_frac> <cursor|jump>"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import deque

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda'; W = 336; FEAT = 'data/loc_feat2.h5'
THRESH = 0.3                                     # ô là "tổn thương" nếu >=30% diện tích giao GT
R_HIT, R_MISS, R_DUP, R_STEP, R_TERM = 1.0, -0.5, -0.5, -0.05, 5.0
NATOMS = 51; VMIN, VMAX = -10.0, 10.0; GAMMA = 0.99; NSTEP = 3
ZSUP = torch.linspace(VMIN, VMAX, NATOMS, device=DEV)


class Cfg:
    """Kích thước lưới, ngân sách, chế độ hành động."""

    def __init__(self, g: int, budget: float, mode: str = 'cursor') -> None:
        self.G = g; self.K = max(1, round(budget * g * g)); self.mode = mode
        self.E = 2 * g - 1; self.CS = W / g
        self.NA = 6 if mode == 'cursor' else g * g + 1
        self.MAXSTEP = 5 * g if mode == 'cursor' else self.K + 1
        self.ckpt = 'data/rainbow_loc_grid_%s_g%d.pth' % (mode, g)


def iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cell_labels(gt: list[float], cfg: Cfg) -> np.ndarray:
    """(G,G) tỉ lệ diện tích mỗi ô giao với box GT."""
    g, cs = cfg.G, cfg.CS; lab = np.zeros((g, g), np.float32)
    for r in range(g):
        for c in range(g):
            ix = max(0.0, min((c + 1) * cs, gt[2]) - max(c * cs, gt[0]))
            iy = max(0.0, min((r + 1) * cs, gt[3]) - max(r * cs, gt[1]))
            lab[r, c] = ix * iy / (cs * cs)
    return lab


def cells_box(mask: np.ndarray, cfg: Cfg) -> list[float]:
    """Bbox hợp các ô có mask=1 (mask phẳng G*G), toạ độ px."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    rs, cs = idx // cfg.G, idx % cfg.G
    return [float(cs.min() * cfg.CS), float(rs.min() * cfg.CS),
            float((cs.max() + 1) * cfg.CS), float((rs.max() + 1) * cfg.CS)]


def nearest_dist(pos: list[int], remaining: set[tuple[int, int]]) -> int:
    return min(max(abs(pos[0] - r), abs(pos[1] - c)) for r, c in remaining) if remaining else 0


def valid_mask(sel: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Hành động hợp lệ (B,NA) từ mask đã chọn (B,G*G). cursor: mọi hành động; jump: ô chưa chọn + STOP."""
    if cfg.mode == 'cursor':
        return np.ones((sel.shape[0], cfg.NA), bool)
    return np.concatenate([sel == 0, sel.any(1, keepdims=True)], 1)


class GridEnv:
    """MDP chọn ô trên lưới. lab/gt chỉ dùng tính reward (None khi suy luận)."""

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg

    def reset(self, lab: np.ndarray | None = None, gt: list[float] | None = None) -> None:
        g = self.cfg.G; self.gt = gt; self.pos = [g // 2, g // 2]
        self.sel = np.zeros(g * g, np.uint8); self.nsel = 0; self.t = 0; self.hit = 0
        self.remaining = set() if lab is None else {
            (r, c) for r in range(g) for c in range(g) if lab[r, c] >= THRESH}
        self.npos = len(self.remaining)

    def scal(self) -> list[float]:
        return [self.nsel / self.cfg.K, self.t / self.cfg.MAXSTEP]

    def valid(self) -> np.ndarray:
        return valid_mask(self.sel[None], self.cfg)[0]

    def _move(self, a: int) -> float:
        g = self.cfg.G; r, c = self.pos; d0 = nearest_dist(self.pos, self.remaining)
        if a == 0: r = max(0, r - 1)
        elif a == 1: r = min(g - 1, r + 1)
        elif a == 2: c = max(0, c - 1)
        else: c = min(g - 1, c + 1)
        self.pos = [r, c]
        return R_STEP + (d0 - nearest_dist(self.pos, self.remaining))

    def _select(self) -> float:
        r, c = self.pos; j = r * self.cfg.G + c
        if self.sel[j]:
            return R_DUP
        self.sel[j] = 1; self.nsel += 1
        if (r, c) in self.remaining:
            self.remaining.discard((r, c)); self.hit += 1; return R_HIT
        return R_MISS

    def step(self, a: int) -> tuple[float, bool]:
        cfg = self.cfg; g = cfg.G; self.t += 1; done = False
        stop = a == 5 if cfg.mode == 'cursor' else a == g * g
        if stop:
            if self.nsel == 0:
                rw = R_DUP
            else:
                rw = 0.0; done = True
        elif cfg.mode == 'jump':
            self.pos = [a // g, a % g]; rw = self._select()
        elif a < 4:
            rw = self._move(a)
        else:
            rw = self._select()
        if self.nsel >= cfg.K or self.t >= cfg.MAXSTEP:
            done = True
        if done and self.gt is not None:
            rw += R_TERM * iou(self.box(), self.gt)
        return rw, done

    def box(self) -> list[float]:
        return cells_box(self.sel, self.cfg)


def make_input(feat: torch.Tensor, rs: list[int], cs: list[int], sels: list[np.ndarray],
               cfg: Cfg) -> torch.Tensor:
    """feat (B,CH,G,G) -> ảnh state. cursor: cắt egocentric (2G-1)x(2G-1); jump: toàn lưới G x G."""
    b = feat.size(0); g = cfg.G; e = cfg.E; d = feat.device
    sel = torch.as_tensor(np.asarray(sels), device=d).float().view(b, 1, g, g)
    u = (torch.arange(g, device=d).float() + 0.5) / g
    coord = torch.stack([u.view(1, g).expand(g, g), u.view(g, 1).expand(g, g)]).unsqueeze(0).expand(b, -1, -1, -1)
    if cfg.mode == 'jump':
        return torch.cat([feat, sel, coord], 1)
    m = torch.cat([feat, sel, torch.ones(b, 1, g, g, device=d), coord], 1)
    m = F.pad(m, (g - 1,) * 4).permute(0, 2, 3, 1)
    ar = torch.arange(e, device=d)
    rr = torch.as_tensor(rs, device=d).view(b, 1) + ar
    cc = torch.as_tensor(cs, device=d).view(b, 1) + ar
    out = m[torch.arange(b, device=d).view(b, 1, 1), rr.view(b, e, 1), cc.view(b, 1, e)]
    return out.permute(0, 3, 1, 2).contiguous()


class Noisy(nn.Module):
    """Factorised NoisyLinear."""

    def __init__(self, i: int, o: int, sig: float = 0.5) -> None:
        super().__init__(); self.i, self.o = i, o
        self.wm = nn.Parameter(torch.empty(o, i)); self.ws = nn.Parameter(torch.empty(o, i))
        self.bm = nn.Parameter(torch.empty(o)); self.bs = nn.Parameter(torch.empty(o))
        self.register_buffer('we', torch.empty(o, i)); self.register_buffer('be', torch.empty(o))
        mu = 1 / math.sqrt(i); self.wm.data.uniform_(-mu, mu); self.bm.data.uniform_(-mu, mu)
        self.ws.data.fill_(sig / math.sqrt(i)); self.bs.data.fill_(sig / math.sqrt(o)); self.reset_noise()

    @staticmethod
    def _n(n: int) -> torch.Tensor:
        x = torch.randn(n); return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        ei = self._n(self.i); eo = self._n(self.o); self.we.copy_(eo.ger(ei)); self.be.copy_(eo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, self.wm + self.ws * self.we, self.bm + self.bs * self.be)
        return F.linear(x, self.wm, self.bm)


class CursorNet(nn.Module):
    """CNN egocentric + dueling + C51 (6 hành động)."""

    def __init__(self, ch: int, cfg: Cfg) -> None:
        super().__init__(); e = cfg.E; self.na = cfg.NA
        self.c = nn.Sequential(nn.Conv2d(ch + 4, 64, 1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32 * e * e + 2, 256), nn.ReLU())
        self.v1 = Noisy(256, 128); self.v2 = Noisy(128, NATOMS)
        self.a1 = Noisy(256, 128); self.a2 = Noisy(128, self.na * NATOMS)

    def forward(self, x: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([self.c(x).flatten(1), sc], 1))
        v = self.v2(F.relu(self.v1(h))).view(-1, 1, NATOMS)
        a = self.a2(F.relu(self.a1(h))).view(-1, self.na, NATOMS)
        return F.log_softmax(v + a - a.mean(1, keepdim=True), dim=2)

    def reset_noise(self) -> None:
        for m in (self.v1, self.v2, self.a1, self.a2):
            m.reset_noise()


class JumpNet(nn.Module):
    """Q-map kiểu attention: advantage mỗi ô từ conv 1x1 dùng chung + STOP/value từ đặc trưng toàn cục."""

    def __init__(self, ch: int, cfg: Cfg) -> None:
        super().__init__(); self.g = cfg.G
        self.c = nn.Sequential(nn.Conv2d(ch + 5, 64, 1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.cell = nn.Conv2d(64, NATOMS, 1)
        self.fc = nn.Sequential(nn.Linear(64 + 2, 128), nn.ReLU())
        self.v = Noisy(128, NATOMS); self.stop = Noisy(128, NATOMS)

    def forward(self, x: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        b = x.size(0); g = self.g
        h = self.c(torch.cat([x, sc.view(b, 2, 1, 1).expand(-1, -1, g, g)], 1))
        z = self.fc(torch.cat([h.mean((2, 3)), sc], 1))
        a = torch.cat([self.cell(h).flatten(2).transpose(1, 2), self.stop(z).unsqueeze(1)], 1)
        return F.log_softmax(self.v(z).unsqueeze(1) + a - a.mean(1, keepdim=True), dim=2)

    def reset_noise(self) -> None:
        self.v.reset_noise(); self.stop.reset_noise()


def make_net(ch: int, cfg: Cfg) -> nn.Module:
    return CursorNet(ch, cfg) if cfg.mode == 'cursor' else JumpNet(ch, cfg)


def q_values(net: nn.Module, feat1: torch.Tensor, env: GridEnv) -> torch.Tensor:
    x = make_input(feat1, [env.pos[0]], [env.pos[1]], [env.sel], env.cfg)
    sc = torch.tensor([env.scal()], device=feat1.device)
    q = (net(x, sc).exp() * ZSUP).sum(2)
    return q.masked_fill(~torch.as_tensor(env.valid(), device=q.device).unsqueeze(0), -1e9)


@torch.no_grad()
def greedy_rollout(net: nn.Module, feat1: torch.Tensor, cfg: Cfg, lab: np.ndarray | None = None,
                   gt: list[float] | None = None) -> GridEnv:
    """Suy luận 1 frame: feat1 (1,CH,G,G) đã chuẩn hoá. Trả env sau episode (ô đã chọn, số bước)."""
    env = GridEnv(cfg); env.reset(lab, gt)
    for _ in range(cfg.MAXSTEP):
        _, done = env.step(int(q_values(net, feat1, env).argmax(1)))
        if done:
            break
    return env


def pool_grid(fm: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    return F.adaptive_avg_pool2d(fm, (cfg.G, cfg.G))


class Replay:
    """PER lưu dạng gọn (frame id, vị trí, mask đã chọn, vô hướng) cho s và s'."""

    def __init__(self, cap: int, cfg: Cfg) -> None:
        gg = cfg.G * cfg.G; self.cap = cap; self.n = 0; self.i = 0
        self.ID = np.zeros(cap, np.int64); self.A = np.zeros(cap, np.int64)
        self.R = np.zeros(cap, np.float32); self.D = np.zeros(cap, np.float32); self.P = np.zeros(cap, np.float32)
        self.POS = np.zeros((2, cap, 2), np.int64); self.SEL = np.zeros((2, cap, gg), np.uint8)
        self.SC = np.zeros((2, cap, 2), np.float32)

    def add(self, fid: int, o0: tuple, a: int, r: float, o1: tuple, d: bool) -> None:
        j = self.i; self.ID[j] = fid; self.A[j] = a; self.R[j] = r; self.D[j] = float(d)
        for k, o in enumerate((o0, o1)):
            self.POS[k, j] = o[0]; self.SEL[k, j] = o[1]; self.SC[k, j] = o[2]
        self.P[j] = self.P[:self.n].max() if self.n > 0 else 1.0
        self.i = (j + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.P[:self.n] ** 0.5; p = p / p.sum(); idx = np.random.choice(self.n, b, p=p)
        w = (self.n * p[idx]) ** (-beta); return idx, (w / w.max()).astype(np.float32)

    def update(self, idx: np.ndarray, pr: np.ndarray) -> None:
        self.P[idx] = pr + 1e-5


def obs(env: GridEnv) -> tuple:
    return (list(env.pos), env.sel.copy(), env.scal())


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cfg = Cfg(int(sys.argv[2]) if len(sys.argv) > 2 else 6, float(sys.argv[3]) if len(sys.argv) > 3 else 0.25,
              sys.argv[4] if len(sys.argv) > 4 else 'cursor')
    batch, buf, warm, train_every, tgt_every, lr = 64, 100000, 5000, 4, 2000, 1e-4
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    f = h5py.File(FEAT, 'r'); keys = list(f.keys()); ch = f[keys[0]]['fm'].shape[0]
    fm = torch.stack([pool_grid(torch.from_numpy(np.asarray(f[k]['fm'], np.float32)).unsqueeze(0), cfg)[0]
                      for k in keys])
    gts = [np.asarray(f[k]['gt'], np.float32).tolist() for k in keys]
    vidall = [f[k].attrs['vid'] for k in keys]
    o = sorted(set(vidall)); np.random.RandomState(0).shuffle(o); te = set(o[:max(3, int(0.25 * len(o)))])
    tr_ids = [i for i in range(len(keys)) if vidall[i] not in te]
    te_ids = [i for i in range(len(keys)) if vidall[i] in te]
    mu = fm[tr_ids].mean((0, 2, 3)); sd = fm[tr_ids].std((0, 2, 3)) + 1e-5
    fmg = ((fm - mu.view(1, -1, 1, 1)) / sd.view(1, -1, 1, 1)).to(DEV)
    labs = [cell_labels(g, cfg) for g in gts]
    print('mode %s | frame train %d | test %d | video test %d | grid %dx%d | K=%d | MAXSTEP %d | NA %d'
          % (cfg.mode, len(tr_ids), len(te_ids), len(te), cfg.G, cfg.G, cfg.K, cfg.MAXSTEP, cfg.NA), flush=True)

    net = make_net(ch, cfg).to(DEV); tgt = make_net(ch, cfg).to(DEV)
    tgt.load_state_dict(net.state_dict()); tgt.eval()
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1.5e-4)
    mem = Replay(buf, cfg); dz = (VMAX - VMIN) / (NATOMS - 1)

    def state(ids: np.ndarray, k: int, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        p = mem.POS[k, idx]
        x = make_input(fmg[torch.as_tensor(ids, device=DEV)], p[:, 0].tolist(), p[:, 1].tolist(),
                       list(mem.SEL[k, idx]), cfg)
        return x, torch.as_tensor(mem.SC[k, idx], device=DEV)

    def learn(beta: float) -> None:
        idx, w = mem.sample(batch, beta); ids = mem.ID[idx]
        s, ssc = state(ids, 0, idx); ns, nsc = state(ids, 1, idx)
        a = torch.as_tensor(mem.A[idx], device=DEV); r = torch.as_tensor(mem.R[idx], device=DEV)
        d = torch.as_tensor(mem.D[idx], device=DEV)
        nvalid = torch.as_tensor(valid_mask(mem.SEL[1, idx], cfg), device=DEV)
        with torch.no_grad():
            net.reset_noise()
            a_ = (net(ns, nsc).exp() * ZSUP).sum(2).masked_fill(~nvalid, -1e9).argmax(1)
            tgt.reset_noise(); tp = tgt(ns, nsc).exp()[range(batch), a_]
            tz = (r.unsqueeze(1) + (GAMMA ** NSTEP) * ZSUP.unsqueeze(0) * (1 - d.unsqueeze(1))).clamp(VMIN, VMAX)
            bj = ((tz - VMIN) / dz).clamp(0, NATOMS - 1); lo = bj.floor().long(); up = bj.ceil().long()
            lo[(up > 0) & (lo == up)] -= 1; up[(lo < NATOMS - 1) & (lo == up)] += 1
            lo = lo.clamp(0, NATOMS - 1); up = up.clamp(0, NATOMS - 1)
            m = torch.zeros(batch, NATOMS, device=DEV); off = torch.arange(batch, device=DEV).unsqueeze(1) * NATOMS
            m.view(-1).index_add_(0, (lo + off).view(-1), (tp * (up.float() - bj)).view(-1))
            m.view(-1).index_add_(0, (up + off).view(-1), (tp * (bj - lo.float())).view(-1))
        net.reset_noise(); logp = net(s, ssc)[range(batch), a]
        loss_i = -(m * logp).sum(1); loss = (torch.as_tensor(w, device=DEV) * loss_i).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        mem.update(idx, loss_i.detach().abs().cpu().numpy())

    def evaluate(ids: list[int], limit: int = 400) -> dict[str, float]:
        net.eval()
        ids = ids if len(ids) <= limit else list(np.random.RandomState(1).choice(ids, limit, replace=False))
        res: dict[str, list[float]] = {k: [] for k in ('iou', 'rec', 'prec', 'steps', 'rnd_iou', 'rnd_rec', 'orc_iou')}
        for i in ids:
            env = greedy_rollout(net, fmg[i:i + 1], cfg, labs[i], gts[i])
            res['iou'].append(iou(env.box(), gts[i])); res['steps'].append(env.t)
            res['rec'].append(env.hit / max(env.npos, 1)); res['prec'].append(env.hit / max(env.nsel, 1))
            env.reset(labs[i], gts[i])
            for _ in range(cfg.MAXSTEP):
                if env.step(int(random.choice(np.flatnonzero(env.valid()))))[1]:
                    break
            res['rnd_iou'].append(iou(env.box(), gts[i])); res['rnd_rec'].append(env.hit / max(env.npos, 1))
            res['orc_iou'].append(iou(cells_box((labs[i] >= THRESH).reshape(-1), cfg), gts[i]))
        net.train()
        out = {k: float(np.mean(v)) for k, v in res.items()}
        for k in ('', 'rnd_', 'orc_'):
            out[k + 'dr'] = float((np.array(res[k + 'iou']) >= 0.5).mean())
        return out

    nbuf: deque = deque(); env = GridEnv(cfg); step = 0; ep_r: list[float] = []; curve = []

    def push_n(force: bool) -> None:
        while len(nbuf) >= NSTEP or (force and nbuf):
            n = min(NSTEP, len(nbuf)); ret = sum((GAMMA ** j) * nbuf[j][3] for j in range(n))
            last = nbuf[n - 1]; mem.add(nbuf[0][0], nbuf[0][1], nbuf[0][2], ret, last[4], last[5])
            nbuf.popleft()
            if not force and len(nbuf) < NSTEP:
                break

    for ep in range(1, episodes + 1):
        i = random.choice(tr_ids); env.reset(labs[i], gts[i]); nbuf.clear(); tot = 0.0
        eps = max(0.02, 0.3 * (1 - ep / (0.3 * episodes)))
        for _ in range(cfg.MAXSTEP):
            net.reset_noise(); o0 = obs(env)
            if random.random() < eps:
                a = int(random.choice(np.flatnonzero(env.valid())))
            else:
                with torch.no_grad():
                    a = int(q_values(net, fmg[i:i + 1], env).argmax(1))
            r, d = env.step(a); nbuf.append((i, o0, a, r, obs(env), d)); push_n(d); tot += r; step += 1
            if mem.n >= warm and step % train_every == 0:
                learn(min(1.0, 0.4 + 0.6 * ep / episodes))
            if step % tgt_every == 0:
                tgt.load_state_dict(net.state_dict())
            if d:
                break
        ep_r.append(tot)
        if ep % 1000 == 0:
            m = evaluate(te_ids); m.update(ep=ep, epR=float(np.mean(ep_r[-1000:]))); curve.append(m)
            print('ep %6d | step %8d | epR %6.2f | TEST IoU %.3f DR %.1f%% rec %.1f%% prec %.1f%% steps %.1f'
                  ' | random IoU %.3f | oracle-grid IoU %.3f'
                  % (ep, step, m['epR'], m['iou'], m['dr'] * 100, m['rec'] * 100, m['prec'] * 100, m['steps'],
                     m['rnd_iou'], m['orc_iou']), flush=True)

    te_m = evaluate(te_ids, 10 ** 9); tr_m = evaluate(tr_ids, 800)
    print('\n=== GRID-ATTENTION RAINBOW — mode %s, grid %dx%d, K=%d ===' % (cfg.mode, cfg.G, cfg.G, cfg.K))
    for name, m in (('TEST', te_m), ('TRAIN', tr_m)):
        print('%-5s: IoU %.3f | DR@0.5 %.1f%% | recall ô %.1f%% | precision ô %.1f%% | steps %.1f'
              % (name, m['iou'], m['dr'] * 100, m['rec'] * 100, m['prec'] * 100, m['steps']))
    print('TEST random policy: IoU %.3f DR %.1f%% | oracle-grid (trần lượng tử hoá): IoU %.3f DR %.1f%%'
          % (te_m['rnd_iou'], te_m['rnd_dr'] * 100, te_m['orc_iou'], te_m['orc_dr'] * 100))
    torch.save({'state': net.state_dict(), 'G': cfg.G, 'budget': cfg.K / (cfg.G * cfg.G), 'mode': cfg.mode,
                'ch': ch, 'mu': mu, 'sd': sd}, cfg.ckpt)
    json.dump({'test': te_m, 'train': tr_m, 'curve': curve, 'G': cfg.G, 'K': cfg.K, 'mode': cfg.mode,
               'episodes': episodes, 'thresh': THRESH},
              open(cfg.ckpt.replace('.pth', '_metrics.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
