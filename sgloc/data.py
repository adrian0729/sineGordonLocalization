"""Data pools (initial conditions + labels) and the model inputs built from them.

A pool stores only what is expensive to recompute: the initial angles and the labels. Every
input view (frames, channels, sensor noise, ...) is regenerated from the initial angles with
the same deterministic simulator, so new ablations never need a new dataset.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from . import sim

A_RANGE = (1.65, 1.85)
NOISE = 0.05  # theta_i(0) = A + NOISE * (U(0, 1) - 0.5), as in the paper
POOLS = {
    "train": {"size": 100_000, "seed": 1},
    "val": {"size": 20_000, "seed": 2},
    "test": {"size": 20_000, "seed": 3},
}


def initial_conditions(size, seed, n_sites=100, a_range=A_RANGE, noise=NOISE):
    """Uniform swing amplitude A per run plus independent uniform noise on every pendulum."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(a_range[0], a_range[1], size)
    theta0 = A[:, None] + noise * (rng.random((size, n_sites)) - 0.5)
    return A, theta0


def _localize_chunk(args):
    theta0, params, labels, onsite, gamma = args
    return sim.localize(theta0, params, labels, onsite, gamma)


def localize_parallel(theta0, params=sim.Params(), chunk=2000, workers=None, labels=sim.LABELS, onsite=None, gamma=0.0):
    """sim.localize over many runs, split into chunks across processes (order is preserved).
    `gamma` is the damping, a scalar or one value per run."""
    gamma = np.asarray(gamma, dtype=float)
    chunks = [(theta0[i:i + chunk], params, labels, None if onsite is None else onsite[i:i + chunk],
               gamma[i:i + chunk] if gamma.ndim else gamma) for i in range(0, len(theta0), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_localize_chunk, chunks))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def make_pool(name, path, params=sim.Params(), workers=None):
    spec = POOLS[name]
    A, theta0 = initial_conditions(spec["size"], spec["seed"])
    start = time.time()
    labels = localize_parallel(theta0, params, workers=workers)
    meta = {"name": name, **spec, "a_range": A_RANGE, "noise": NOISE, "params": asdict(params),
            "numpy": np.__version__, "seconds": round(time.time() - start, 1),
            "created": time.strftime("%Y-%m-%d %H:%M")}
    np.savez(path, A=A, theta0=theta0, meta=json.dumps(meta), **labels)
    return meta


def load_pool(path, size=None):
    """Load a pool; `size` keeps only the first `size` runs."""
    pool = {}
    with np.load(path) as f:
        for key in f.files:
            arr = f[key]
            pool[key] = arr[:size] if arr.ndim else arr
    pool["meta"] = json.loads(str(pool["meta"]))
    return pool


def v1_inputs(theta0, params=sim.Params(), n_frames=10, bond="right", chunk=10_000):
    """The paper's ML input: site energy at the first `n_frames` samples (tau = 0, 0.5, ...) plus
    theta(0) repeated in every frame. Shape (runs, n_frames, N, 2), float32."""
    out = np.empty((len(theta0), n_frames, theta0.shape[1], 2), dtype=np.float32)
    for i in range(0, len(theta0), chunk):
        theta, omega = sim.evolve(theta0[i:i + chunk], n_frames, params)
        out[i:i + chunk, ..., 0] = sim.site_energy(theta, omega, params.g, params.h, bond)
        out[i:i + chunk, ..., 1] = theta0[i:i + chunk, None, :]
    return out
