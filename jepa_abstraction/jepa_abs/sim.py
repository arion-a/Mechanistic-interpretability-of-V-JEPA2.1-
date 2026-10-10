"""Matched-state free-flight simulator and renderer (protocol 01, section 3).

One anchor state + nuisance bundle is shared by every gravity law; history is integrated
backward analytically and the future forward, so anchor state/pixels are identical across laws.
Gravity acts on +z (down = -z). Camera: fixed orthographic oblique view.
"""
from dataclasses import dataclass, field
import numpy as np

FPS = 30.0
ANCHOR = 31          # evaluation anchor frame index
N_FRAMES = 64
LAWS_K3 = (4.0, 9.0, 14.0)


@dataclass(frozen=True)
class RenderCfg:
    res: int = 32
    u_range: tuple = (-4.0, 4.0)
    v_range: tuple = (6.0, 22.0)
    radius_m: float = 0.10
    min_sigma_px: float = 0.9   # pilot deviation: sub-pixel Gaussian splat so tiny motions stay encoded


def sample_anchors(rng, n):
    p0 = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(12, 16, n)], 1)
    v0 = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(-2, 2, n)], 1)
    return p0, v0


def sample_nuisance(rng, n):
    return dict(color=rng.uniform(0.3, 1.0, (n, 3)), bg=rng.uniform(0.0, 0.3, (n, 1)),
                light=rng.uniform(0.7, 1.0, (n, 1)))


def positions(p0, v0, g, tau):
    """p(tau)=p0+v0 tau+0.5 a tau^2, a=(0,0,-g). p0,v0:[N,3]; g:[N]; tau:[T] seconds from anchor -> [N,T,3]."""
    tau = np.asarray(tau, float)[None, :, None]
    a = np.zeros_like(p0); a[:, 2] = -np.asarray(g, float)
    return p0[:, None] + v0[:, None] * tau + 0.5 * a[:, None] * tau ** 2


def frame_tau(frames):
    return (np.asarray(frames) - ANCHOR) / FPS


def project(p):
    """Orthographic oblique projection of [...,3] to image-plane (u,v) metres."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([x + 0.4 * y, 0.92 * z - 0.38 * y], -1)


def render(p, nuis, cfg=RenderCfg()):
    """p:[N,T,3] -> float32 RGB [N,T,3,R,R] in [0,1]."""
    R = cfg.res
    uv = project(p)
    (u0, u1), (v0, v1) = cfg.u_range, cfg.v_range
    col = (uv[..., 0] - u0) / (u1 - u0) * R - 0.5
    row = (v1 - uv[..., 1]) / (v1 - v0) * R - 0.5
    sig = max(cfg.min_sigma_px, cfg.radius_m / (u1 - u0) * R)
    grid = np.arange(R, dtype=np.float32)
    gc = np.exp(-0.5 * ((grid[None, None] - col[..., None]) / sig) ** 2)   # [N,T,R]
    gr = np.exp(-0.5 * ((grid[None, None] - row[..., None]) / sig) ** 2)
    blob = gr[..., :, None] * gc[..., None, :]                                # [N,T,R,R]
    bg = nuis["bg"][:, None, None, :, None]                                   # [N,1,1,1,1]
    rgb = bg + blob[:, :, None] * (nuis["color"] * nuis["light"])[:, None, :, None, None]
    return np.clip(rgb, 0, 1).astype(np.float32)


@dataclass
class Bundle:
    """Matched-state group set: G anchors x K laws. Arrays are law-major within group: index = a*K + k."""
    p0: np.ndarray; v0: np.ndarray; nuis: dict
    laws: np.ndarray                 # [K]
    K: int = 0
    def __post_init__(self): self.K = len(self.laws)
    @property
    def n_groups(self): return len(self.p0)
    def expand(self, arr): return np.repeat(arr, self.K, axis=0)
    def g(self): return np.tile(self.laws, self.n_groups)
    def group_id(self): return np.repeat(np.arange(self.n_groups), self.K)
    def state(self): return np.concatenate([self.expand(self.p0), self.expand(self.v0)], 1)

    def clips(self, c, offsets):
        """Context RGB [N,c,3,R,R] ending at anchor; future positions [N,len(offsets),3]."""
        N = self.n_groups * self.K
        p0, v0, g = self.expand(self.p0), self.expand(self.v0), self.g()
        ctx = positions(p0, v0, g, frame_tau(np.arange(ANCHOR - c + 1, ANCHOR + 1)))
        fut = positions(p0, v0, g, frame_tau(ANCHOR + np.asarray(offsets)))
        return ctx, fut


def make_bundle(seed, n_groups, laws=LAWS_K3):
    rng = np.random.default_rng(seed)
    p0, v0 = sample_anchors(rng, n_groups)
    return Bundle(p0, v0, sample_nuisance(rng, n_groups), np.asarray(laws, float))


def render_ctx(b, ctx_pos, cfg=RenderCfg(), idx=None):
    idx = np.arange(len(ctx_pos)) if idx is None else idx
    nuis = {k: b.expand(v)[idx] for k, v in b.nuis.items()}
    return render(ctx_pos[idx], nuis, cfg)
