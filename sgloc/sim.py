"""Vectorized simulator for the sine-Gordon ring with a global term (paper Eq. 4).

Every function works on a batch of chains: theta has shape (runs, N), or (runs, frames, N) for
the energy functions. Time is dimensionless (tau = omega0 * t).

Optional linear damping adds -gamma * dtheta/dtau to Eq. (4): friction proportional to each
pendulum's angular velocity, with gamma in units of omega0 (an undriven pendulum's quality factor
is Q = 1 / gamma). The default gamma = 0 is the paper's conservative chain.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    g: float = 0.75          # coupling strength g~
    h: float = 0.5           # global (mean-field) strength h~; the paper's code uses 0.5 throughout
    dt: float = 0.05         # RK4 step
    sample: float = 0.5      # spacing of stored samples (the paper's dt)
    t_final: float = 150.0   # labels are searched over samples with tau < t_final
    threshold: float = 0.1   # localized once one site holds this fraction of the total energy
    avg_window: int = 16     # samples averaged by the "avg" label (8 time units, about one swing)


BONDS = ("right", "left", "sym")
LABELS = ("right", "left", "sym", "avg")


def accel(theta, g, h, onsite=None, gamma=0.0, omega=None):
    """Angular acceleration from Eq. (4) on a ring (periodic boundaries).

    `onsite` scales each pendulum's gravity term (1 everywhere for a pure chain; 1 - eps at an
    impurity, as in Sec. 3.5 of the paper). `gamma` adds the damping -gamma * omega; it is a
    scalar or one value per run with shape (runs, 1).
    """
    gravity = np.sin(theta) if onsite is None else onsite * np.sin(theta)
    a = (-gravity + g * (np.roll(theta, 1, axis=-1) + np.roll(theta, -1, axis=-1) - 2 * theta)
         + h * (theta.mean(axis=-1, keepdims=True) - theta))
    return a - gamma * omega if np.any(gamma) else a


def damping(gamma):
    """gamma as given, or one value per run (shape (runs,)) as a (runs, 1) column over the sites."""
    gamma = np.asarray(gamma, dtype=float)
    return gamma[:, None] if gamma.ndim == 1 else gamma


def rk4_step(theta, omega, dt, g, h, onsite=None, gamma=0.0):
    a1 = accel(theta, g, h, onsite, gamma, omega)
    v2 = omega + 0.5 * dt * a1
    a2 = accel(theta + 0.5 * dt * omega, g, h, onsite, gamma, v2)
    v3 = omega + 0.5 * dt * a2
    a3 = accel(theta + 0.5 * dt * v2, g, h, onsite, gamma, v3)
    v4 = omega + dt * a3
    a4 = accel(theta + dt * v3, g, h, onsite, gamma, v4)
    return (theta + dt / 6 * (omega + 2 * v2 + 2 * v3 + v4),
            omega + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4))


def site_energy(theta, omega, g, h, bond="right", onsite=None):
    """Energy of each site, Eq. (5).

    `bond` decides which site owns each spring: "right" gives site n the spring to n+1 (the
    paper's code), "left" the spring to n-1 (Eq. 5 as printed), "sym" half of each. All three
    have the same total. `onsite` scales the gravity term as in accel().
    """
    gravity = 1 - np.cos(theta)
    if onsite is not None:
        gravity = gravity * (onsite[..., None, :] if onsite.ndim < theta.ndim else onsite)
    stretch = theta - np.roll(theta, -1, axis=-1)  # spring between n and n+1
    if bond == "right":
        spring = stretch**2
    elif bond == "left":
        spring = np.roll(stretch, 1, axis=-1) ** 2
    elif bond == "sym":
        spring = 0.5 * (stretch**2 + np.roll(stretch, 1, axis=-1) ** 2)
    else:
        raise ValueError(f"unknown bond convention {bond!r}")
    return (0.5 * omega**2 + gravity + 0.5 * g * spring
            + 0.5 * h * (theta - theta.mean(axis=-1, keepdims=True)) ** 2)


def evolve(theta0, n_samples, params=Params(), omega0=None, onsite=None, gamma=0.0):
    """Return theta and omega at samples k = 0 .. n_samples-1, each of shape (runs, n_samples, N).
    `gamma` is the damping, a scalar or one value per run."""
    theta = np.array(theta0, dtype=float)
    omega = np.zeros_like(theta) if omega0 is None else np.array(omega0, dtype=float)
    gamma = damping(gamma)
    stride = round(params.sample / params.dt)
    thetas = np.empty((*theta.shape[:-1], n_samples, theta.shape[-1]))
    omegas = np.empty_like(thetas)
    for k in range(n_samples):
        thetas[..., k, :], omegas[..., k, :] = theta, omega
        if k < n_samples - 1:
            for _ in range(stride):
                theta, omega = rk4_step(theta, omega, params.dt, params.g, params.h, onsite, gamma)
    return thetas, omegas


def localize(theta0, params=Params(), labels=LABELS, onsite=None, gamma=0.0):
    """Run each chain (released from rest) until it localizes; return the site and time for each label.

    A chain localizes at the first sample where one site holds more than `threshold` of the
    total energy, and its label is that site (-1 and NaN if it never happens). The "right",
    "left" and "sym" labels use the instantaneous site energy with that bond convention. "avg"
    uses the symmetric energy averaged over the last `avg_window` samples, which removes the
    within-swing exchange between kinetic and spring energy; its time is the end of the window.

    With damping (`gamma`, a scalar or one value per run) the total energy decays, so the
    threshold is a fraction of the energy the chain holds at that sample (for "avg", averaged
    over the same window); without damping it is the conserved initial energy.
    """
    p = params
    theta = np.array(theta0, dtype=float)
    omega = np.zeros_like(theta)
    gamma = damping(gamma)
    runs, n_sites = theta.shape
    damped = np.broadcast_to(gamma != 0, (runs, 1))[:, 0]  # runs whose chain loses energy
    stride = round(p.sample / p.dt)
    n_samples = round(p.t_final / p.sample)
    limit = p.threshold * site_energy(theta, omega, p.g, p.h, "sym", onsite).sum(axis=1)
    site = {name: np.full(runs, -1, dtype=np.int16) for name in labels}
    when = {name: np.full(runs, np.nan, dtype=np.float32) for name in labels}
    window = np.zeros((p.avg_window, runs, n_sites)) if "avg" in labels else None

    for k in range(n_samples):
        energies = {b: site_energy(theta, omega, p.g, p.h, b, onsite) for b in BONDS
                    if b in labels or (b == "sym" and window is not None)}
        if window is not None:
            window[k % p.avg_window] = energies["sym"]
            if k >= p.avg_window - 1:
                energies["avg"] = window.mean(axis=0)
        for name in labels:
            if name not in energies:
                continue
            E = energies[name]
            now = np.where(damped, p.threshold * E.sum(axis=1), limit) if damped.any() else limit
            hit = (site[name] < 0) & (E.max(axis=1) > now)
            site[name][hit] = E[hit].argmax(axis=1)
            when[name][hit] = k * p.sample
        if all((site[name] >= 0).all() for name in labels) or k == n_samples - 1:
            break
        for _ in range(stride):
            theta, omega = rk4_step(theta, omega, p.dt, p.g, p.h, onsite, gamma)

    return {**{f"site_{n}": site[n] for n in labels}, **{f"time_{n}": when[n] for n in labels}}
