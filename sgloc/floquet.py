"""Linear (Floquet) stability of the uniform swing, and the training-free Floquet-filter predictor."""
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk


def period(A):
    """Period of the uniform swing released from rest at amplitude A (dimensionless time)."""
    return 4.0 * ellipk(np.sin(np.asarray(A) / 2.0) ** 2)


def monodromy(A, modes, g=0.75, h=0.5, n_sites=100):
    """One-period monodromy matrices, shape (len(modes), 2, 2), for ring modes m >= 1.

    About the uniform swing theta0(t), mode m obeys d'' = -[cos theta0(t) + h + 4 g sin^2(pi m / N)] d.
    """
    modes = np.asarray(modes)
    stiffness = h + 4.0 * g * np.sin(np.pi * modes / n_sites) ** 2
    count = len(modes)

    def rhs(t, y):
        P = y[2:].reshape(count, 2, 2)
        k = np.cos(y[0]) + stiffness
        dP = np.empty_like(P)
        dP[:, 0, :] = P[:, 1, :]
        dP[:, 1, :] = -k[:, None] * P[:, 0, :]
        return np.concatenate(([y[1], -np.sin(y[0])], dP.ravel()))

    T = float(period(A))
    y0 = np.concatenate(([A, 0.0], np.tile(np.eye(2).ravel(), count)))
    sol = solve_ivp(rhs, (0.0, T), y0, rtol=1e-10, atol=1e-12)
    return T, sol.y[2:, -1].reshape(count, 2, 2)


def growth_rates(A, modes, g=0.75, h=0.5, n_sites=100):
    """Floquet growth rate sigma_m (per unit time) of each ring mode; 0 means neutrally stable."""
    T, P = monodromy(A, modes, g, h, n_sites)
    return np.log(np.abs(np.linalg.eigvals(P)).max(axis=1)) / T


def threshold_amplitude(g=0.75, h=0.5, n_sites=100, lo=1.5, hi=1.7, tol=1e-5):
    """Smallest swing amplitude at which some long-wavelength ring mode starts to grow."""
    def unstable(A):
        return growth_rates(A, np.arange(1, 6), g, h, n_sites).max() > 1e-7

    if unstable(lo) or not unstable(hi):
        raise ValueError(f"the threshold is not inside [{lo}, {hi}]; widen the bracket")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if unstable(mid) else (mid, hi)
    return hi


@lru_cache(maxsize=512)
def _all_mode_monodromy(A, g, h, n_sites):
    return monodromy(A, np.arange(1, n_sites // 2 + 1), g, h, n_sites)[1]


def propagated_field(theta0, A, periods=6, g=0.75, h=0.5, a_grid=None):
    """Initial perturbation evolved through the linearized dynamics for `periods` swing periods
    (released from rest), shape (runs, N). A is the swing amplitude of each run."""
    theta0, A = np.atleast_2d(theta0), np.atleast_1d(A)
    n_sites = theta0.shape[1]
    if a_grid is None:
        a_grid = np.round(np.linspace(A.min(), A.max(), 41), 6) if np.ptp(A) > 0 else A[:1]
    gain = np.zeros((len(a_grid), n_sites // 2 + 1))
    for j, a in enumerate(a_grid):
        P = _all_mode_monodromy(float(a), g, h, n_sites)
        gain[j, 1:] = [np.linalg.matrix_power(p, periods)[0, 0] for p in P]
    nearest = np.abs(A[:, None] - np.asarray(a_grid)[None]).argmin(axis=1)
    spectrum = np.fft.rfft(theta0 - theta0.mean(axis=1, keepdims=True), axis=1)
    return np.fft.irfft(spectrum * gain[nearest], n=n_sites, axis=1)


def floquet_filter(theta0, A, periods=6, g=0.75, h=0.5, a_grid=None):
    """Predict localization sites with no training.

    Evolve the initial perturbation through the linearized dynamics for `periods` swing periods
    and return the site with the largest displacement. At whole periods the swing is at a
    turning point, where the local energy perturbation is sin(A) * delta_theta, so this is also
    the peak of the energy perturbation.
    """
    return propagated_field(theta0, A, periods, g, h, a_grid).argmax(axis=1)
