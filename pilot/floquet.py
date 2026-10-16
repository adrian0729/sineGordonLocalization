"""Floquet growth spectrum of the uniform swing in the sine-Gordon ring with a global term.

Linearizing Eq. (4) about the uniform orbit theta0(t) (theta0'' = -sin theta0) gives, for each
Fourier mode m != 0 of the ring,
    d'' = -[cos theta0(t) + h + 4 g sin^2(pi m / N)] d
a Hill equation. One-period monodromy matrices give the multipliers mu_m and the growth rate
sigma_m = ln max|mu_m| / T.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

N = 100
g = 0.75


def period(A):
    return 4.0 * ellipk(np.sin(A / 2.0) ** 2)


def spectrum(A, h, ms):
    T = period(A)
    q = h + 4.0 * g * np.sin(np.pi * ms / N) ** 2  # constant part of the stiffness per mode
    M = len(ms)

    def rhs(t, y):
        th, om = y[0], y[1]
        P = y[2:].reshape(M, 2, 2)  # fundamental matrices
        k = np.cos(th) + q  # time-dependent stiffness per mode
        dP = np.empty_like(P)
        dP[:, 0, :] = P[:, 1, :]
        dP[:, 1, :] = -k[:, None] * P[:, 0, :]
        return np.concatenate(([om, -np.sin(th)], dP.ravel()))

    y0 = np.concatenate(([A, 0.0], np.tile(np.eye(2).ravel(), M)))
    sol = solve_ivp(rhs, (0, T), y0, rtol=1e-11, atol=1e-12)
    P = sol.y[2:, -1].reshape(M, 2, 2)
    mu = np.array([np.max(np.abs(np.linalg.eigvals(p))) for p in P])
    return T, np.log(mu) / T, P


if __name__ == "__main__":
    ms = np.arange(1, 51)
    for h in (0.5, 0.05):
        print(f"\n=== h = {h} ===")
        print(" A      T      sigma_max  m*  #unstable  e-fold time  unstable m")
        for A in (1.50, 1.55, 1.58, 1.59, 1.60, 1.62, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90):
            T, sig, _ = spectrum(A, h, ms)
            uns = ms[sig > 1e-6]
            smax = sig.max()
            tf = 1 / smax if smax > 1e-6 else np.inf
            rng = f"{uns.min()}-{uns.max()}" if len(uns) else "-"
            print(f"{A:.2f}  {T:6.3f}  {smax:9.5f}  {ms[sig.argmax()]:2d}  {len(uns):9d}  {tf:11.2f}  {rng}")

    # Detailed growth spectrum at the paper's operating point
    A, h = 1.75, 0.5
    T, sig, P = spectrum(A, h, ms)
    print(f"\nA={A}, h={h}: period T={T:.4f}")
    print(" m  wavelength  sigma      gain over t=4.5  gain over t=25")
    for m, s in zip(ms[:15], sig[:15]):
        print(f"{m:2d}  {N/m:8.1f}  {s:9.5f}  {np.exp(s*4.5):10.3f}  {np.exp(s*25):12.2f}")
    np.save(sys.argv[1] if len(sys.argv) > 1 else "floquet_A175_h05.npy", np.c_[ms, sig])
