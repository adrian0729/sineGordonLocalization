"""Model inputs ("views") rebuilt from the data pools: which quantities, which frames, when the
observation starts, sensor noise and sensor spacing. Clean trajectories are cached on disk.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import floquet, sim

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
SAMPLE = 0.5            # spacing of observation frames (time units), as in the paper
PERIOD_SAMPLES = 16     # random start times cover one swing period (T = 7.6-8.0 time units)
SPLIT_SEEDS = {"train": 101, "val": 102, "test": 103}


@dataclass(frozen=True)
class View:
    channels: tuple = ("E",)              # per frame: "theta", "omega", "E" (site energy), "dtheta" (theta - chain mean)
    frames: tuple = tuple(range(10))      # frame offsets in samples (0.5 time units each) from the start
    start: float = 0.0                    # observation start (time units since release)
    random_phase: bool = False            # start at a random time within the first swing period instead
    theta0: bool = False                  # add the first frame's angles as a static channel (v1's theta(0))
    amplitude: bool = False               # add the swing amplitude estimated from the chain-mean energy
    floquet: int = 0                      # add the Floquet-propagated field (this many periods); start must be 0
    noise: float = 0.0                    # Gaussian sensor noise on theta (rad) and omega (rad per time unit)
    stride: int = 1                       # instrument every stride-th pendulum (others masked) and add a mask channel
    layout: str = "channels"              # "channels": (runs, C, N); "frames": (runs, frames, C, N)
    calibration: bool = False             # add a measured map of each pendulum's gravity term (1 - onsite) as a channel

    def key(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def sample_index(view, runs, split):
    frames = np.asarray(view.frames, dtype=int)
    if view.random_phase:
        start = np.random.default_rng(SPLIT_SEEDS[split]).integers(0, PERIOD_SAMPLES, runs)
    else:
        start = np.full(runs, round(view.start / SAMPLE))
    return start[:, None] + frames[None, :]


def observe(theta0, index, params=sim.Params(), onsite=None):
    """theta and omega at per-run sample indices (runs, F); returns two float32 arrays (runs, F, N).
    `params` and `onsite` are the true dynamics (they can differ from the observer's model)."""
    theta = np.array(theta0, dtype=float)
    omega = np.zeros_like(theta)
    stride = round(SAMPLE / params.dt)
    out_t = np.empty((*index.shape, theta.shape[1]), dtype=np.float32)
    out_w = np.empty_like(out_t)
    last = int(index.max())
    for k in range(last + 1):
        rows, cols = np.nonzero(index == k)
        if len(rows):
            out_t[rows, cols], out_w[rows, cols] = theta[rows], omega[rows]
        if k < last:
            for _ in range(stride):
                theta, omega = sim.rk4_step(theta, omega, params.dt, params.g, params.h, onsite)
    return out_t, out_w


def _observe_chunk(args):
    return observe(*args)


def observe_cached(split, theta0, index, workers=8, chunk=2500, dynamics=None, onsite=None):
    """observe() over a whole split, in parallel, cached by the split, size, sample indices and
    (when not the defaults) the true dynamics."""
    key = index.tobytes() + theta0[:5].tobytes()
    if dynamics is not None:
        key += repr(dynamics).encode()
    if onsite is not None:
        key += onsite.tobytes()
    tag = hashlib.sha1(key).hexdigest()[:12]
    path = CACHE / f"{split}_{len(theta0)}_{tag}.npz"
    if path.exists():
        with np.load(path) as f:
            return f["theta"], f["omega"]
    dynamics = dynamics or sim.Params()
    jobs = [(theta0[i:i + chunk], index[i:i + chunk], dynamics, None if onsite is None else onsite[i:i + chunk])
            for i in range(0, len(theta0), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_observe_chunk, jobs))
    theta = np.concatenate([p[0] for p in parts])
    omega = np.concatenate([p[1] for p in parts])
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, theta=theta, omega=omega)
    tmp.rename(path)
    return theta, omega


def ensure(view, split, theta0, workers=8, dynamics=None, onsite=None):
    """Fill the trajectory cache for a view without building the inputs."""
    observe_cached(split, theta0, sample_index(view, len(theta0), split), workers, dynamics=dynamics, onsite=onsite)


def build(view, split, theta0, workers=8, params=sim.Params(), dynamics=None, onsite=None, calibration=None):
    """Inputs for one split. Masked (uninstrumented) sites are NaN so they can be standardized
    away; the caller replaces them with 0 after standardizing.

    `params` is the observer's model (used for the energy channel); `dynamics` and `onsite` are
    the true system when it differs. `calibration`, if given, is a per-site map appended as a
    static channel (e.g. measured pendulum frequencies)."""
    runs, n_sites = theta0.shape
    theta, omega = observe_cached(split, theta0, sample_index(view, runs, split), workers, dynamics=dynamics, onsite=onsite)
    if view.noise > 0:
        rng = np.random.default_rng(SPLIT_SEEDS[split] + 7919)
        theta = theta + np.float32(view.noise) * rng.standard_normal(theta.shape, dtype=np.float32)
        omega = omega + np.float32(view.noise) * rng.standard_normal(omega.shape, dtype=np.float32)
    energy = None
    if "E" in view.channels or view.amplitude:
        energy = sim.site_energy(theta.astype(float), omega.astype(float), params.g, params.h, "right").astype(np.float32)
    per_frame = {"theta": lambda: theta, "omega": lambda: omega, "E": lambda: energy,
                 "dtheta": lambda: theta - theta.mean(-1, keepdims=True)}
    x = np.stack([per_frame[c]() for c in view.channels], axis=2)  # (runs, F, C, N)

    static = []
    if view.theta0:
        static.append(theta[:, 0].copy())
    amp = None
    if view.amplitude or view.floquet:
        amp = np.arccos(np.clip(1 - energy[:, 0].mean(-1), -1, 1)) if energy is not None else theta[:, 0].mean(-1)
    if view.amplitude:
        static.append(np.repeat(amp[:, None], n_sites, axis=1).astype(np.float32))
    if calibration is not None:
        static.append(np.asarray(calibration, dtype=np.float32))
    if view.floquet:
        assert view.start == 0 and not view.random_phase, "the Floquet channel needs the chain at rest"
        static.append(floquet.propagated_field(theta[:, 0].astype(float), amp, view.floquet).astype(np.float32))
    mask = None
    if view.stride > 1:
        mask = (np.arange(n_sites) % view.stride == 0)
        x[..., ~mask] = np.nan
        for s in static:
            s[:, ~mask] = np.nan

    if view.layout == "channels":
        x = x.reshape(runs, -1, n_sites)
        extra = [s[:, None, :] for s in static]
        if mask is not None:
            extra.append(np.broadcast_to(mask.astype(np.float32), (runs, 1, n_sites)))
        return np.concatenate([x, *extra], axis=1) if extra else x
    frames = x.shape[1]
    extra = [np.repeat(s[:, None, None, :], frames, axis=1) for s in static]
    if mask is not None:
        extra.append(np.broadcast_to(mask.astype(np.float32), (runs, frames, 1, n_sites)))
    return np.concatenate([x, *extra], axis=2) if extra else x


def profile_target(split, theta0, workers=8, params=sim.Params(), tau=30.0, window=16):
    """Symmetric site energy averaged over `window` samples starting at `tau` (a dense auxiliary target)."""
    k0 = round(tau / SAMPLE)
    index = np.broadcast_to(np.arange(k0, k0 + window), (len(theta0), window)).copy()
    theta, omega = observe_cached(split, theta0, index, workers)
    return sim.site_energy(theta.astype(float), omega.astype(float), params.g, params.h, "sym").mean(1).astype(np.float32)
