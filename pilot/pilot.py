"""Pilot: vectorized simulator, label robustness (twin runs), and physics-only predictors."""
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

N, g, h = 100, 0.75, 0.5
rng = np.random.default_rng(0)


def accel(th):
    return (-np.sin(th) + g * (np.roll(th, 1, 1) + np.roll(th, -1, 1) - 2 * th)
            + h * (th.mean(1, keepdims=True) - th))


def energy_code(th, om):
    # paper code: 2x Eq. (5), bond (n, n+1) assigned entirely to site n
    return (om**2 + 2 * (1 - np.cos(th)) + g * (th - np.roll(th, -1, 1))**2
            + h * (th - th.mean(1, keepdims=True))**2)


def energy_sym(th, om):
    d = th - np.roll(th, -1, 1)
    return (om**2 + 2 * (1 - np.cos(th)) + 0.5 * g * (d**2 + np.roll(d, 1, 1)**2)
            + h * (th - th.mean(1, keepdims=True))**2)


def simulate(th0, dt=0.05, tf=150.0, sample=0.5, nkeep=11):
    B = th0.shape[0]
    th, om = th0.copy(), np.zeros_like(th0)
    stride, nsteps = int(round(sample / dt)), int(round(tf / dt))
    E0c, E0s = energy_code(th, om).sum(1), energy_sym(th, om).sum(1)
    site_c, time_c = np.full(B, -1), np.full(B, np.nan)
    site_s, time_s = np.full(B, -1), np.full(B, np.nan)
    frames = np.zeros((nkeep, B, 2, N))
    for step in range(nsteps + 1):
        if step % stride == 0:
            k = step // stride
            if k < nkeep:
                frames[k, :, 0], frames[k, :, 1] = th, om
            Ec, Es = energy_code(th, om), energy_sym(th, om)
            hit = (Ec.max(1) > 0.1 * E0c) & (site_c < 0)
            site_c[hit], time_c[hit] = Ec[hit].argmax(1), step * dt
            hit = (Es.max(1) > 0.1 * E0s) & (site_s < 0)
            site_s[hit], time_s[hit] = Es[hit].argmax(1), step * dt
            if k >= nkeep and (site_c >= 0).all() and (site_s >= 0).all():
                break
        a1 = accel(th); v1 = om
        a2 = accel(th + 0.5 * dt * v1); v2 = om + 0.5 * dt * a1
        a3 = accel(th + 0.5 * dt * v2); v3 = om + 0.5 * dt * a2
        a4 = accel(th + dt * v3); v4 = om + dt * a3
        th = th + dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        om = om + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    return dict(site=site_c, time=time_c, site_sym=site_s, time_sym=time_s, frames=frames)


def cdist(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % N
    return np.minimum(d, N - d)


def boxcar(x, K):
    half = K // 2
    return sum(np.roll(x, j, 1) for j in range(-half, half + 1)) / K


def monodromy(A, ms):
    T = 4.0 * ellipk(np.sin(A / 2.0) ** 2)
    q = h + 4.0 * g * np.sin(np.pi * ms / N) ** 2
    M = len(ms)

    def rhs(t, y):
        P = y[2:].reshape(M, 2, 2)
        k = np.cos(y[0]) + q
        dP = np.empty_like(P)
        dP[:, 0, :] = P[:, 1, :]
        dP[:, 1, :] = -k[:, None] * P[:, 0, :]
        return np.concatenate(([y[1], -np.sin(y[0])], dP.ravel()))

    y0 = np.concatenate(([A, 0.0], np.tile(np.eye(2).ravel(), M)))
    sol = solve_ivp(rhs, (0, T), y0, rtol=1e-10, atol=1e-12)
    return T, sol.y[2:, -1].reshape(M, 2, 2)


def summarize(name, pred, true):
    d = cdist(pred, true)
    print(f"  {name:42s} MAE {d.mean():6.2f}  <=1: {np.mean(d <= 1)*100:5.1f}%  "
          f"<=3: {np.mean(d <= 3)*100:5.1f}%  <=5: {np.mean(d <= 5)*100:5.1f}%  >10: {np.mean(d > 10)*100:5.1f}%")
    return d.mean()


if __name__ == "__main__":
    out = sys.argv[1]
    B = 2000
    noise = lambda n: 0.05 * (rng.random((n, N)) - 0.5)

    # ---- 1. Main pilot set, matching the ML dataset (A ~ U(1.65, 1.85)) ----
    A = rng.uniform(1.65, 1.85, B)
    th0 = A[:, None] + noise(B)
    t0 = time.time(); main = simulate(th0); print(f"main set: {time.time()-t0:.1f}s for {B} runs")
    tc = main["time"]
    print(f"localized by t=150: {np.mean(main['site'] >= 0)*100:.2f}%   "
          f"localization time (real units): mean {np.nanmean(tc):.1f}, min {np.nanmin(tc):.1f}, max {np.nanmax(tc):.1f}"
          f"  -> in 0.5-sample index units: mean {np.nanmean(tc)/0.5:.0f}")
    dsym = (main["site_sym"] - main["site"] + N // 2) % N - N // 2
    vals, cnts = np.unique(dsym, return_counts=True)
    print("symmetric-energy label minus code label:",
          {int(v): round(c / B * 100, 2) for v, c in zip(vals, cnts) if c / B > 0.002})

    # ---- 2. Label robustness: integrator step and tiny perturbations ----
    S = 500
    print(f"\nlabel robustness on {S} runs (fraction whose site moves by >1 / MAE between twins):")
    base = main["site"][:S]
    runs = {"dt 0.05 -> 0.025": simulate(th0[:S], dt=0.025)}
    for eps in (1e-9, 1e-6, 1e-3):
        runs[f"IC + {eps:g} perturbation"] = simulate(th0[:S] + eps * (rng.random((S, N)) - 0.5))
    for k, r in runs.items():
        d = cdist(r["site"], base)
        print(f"  {k:28s} moved>1: {np.mean(d > 1)*100:5.1f}%   twin MAE {d.mean():.2f}")

    # ---- 3. Physics-only predictors on the main set ----
    true = main["site"]
    th_f, om_f = main["frames"][:, :, 0], main["frames"][:, :, 1]
    print("\npredictors on the main set (A random in [1.65, 1.85]):")
    for idx in (0, 5, 9):
        E = energy_code(th_f[idx], om_f[idx])
        res ={K: cdist(boxcar(E, K).argmax(1), true).mean() for K in (1, 9, 13, 15, 17, 19, 21, 25, 31)}
        Kb = min(res, key=res.get)
        print(f"  box-car on E at sample {idx} (t={idx*0.5}): " + "  ".join(f"K{K}:{v:.2f}" for K, v in res.items()))
        summarize(f"best box-car K={Kb}, sample {idx}", boxcar(E, Kb).argmax(1), true)

    ms = np.arange(1, 51)
    Agrid = np.linspace(1.65, 1.85, 41)
    mono = [monodromy(a, ms)[1] for a in Agrid]
    sig = np.array([np.log(np.abs(np.linalg.eigvals(Mm)).max(1)) for Mm in mono])
    ia = np.abs(A[:, None] - Agrid[None]).argmin(1)
    dth = th0 - th0.mean(1, keepdims=True)
    F = np.fft.rfft(dth, axis=1)
    print("\n  Floquet propagator filter: site = argmax of linearly evolved perturbation at t = p*T")
    best_p, best_mae = None, 1e9
    for p in (1, 2, 3, 4, 5, 6, 8):
        phi = np.zeros((len(Agrid), 51))
        for j, Mm in enumerate(mono):
            Pp = np.array([np.linalg.matrix_power(x, p) for x in Mm])
            phi[j, 1:] = Pp[:, 0, 0]
        pred = np.fft.irfft(F * phi[ia], n=N, axis=1).argmax(1)
        mae = summarize(f"Floquet filter, p={p} periods", pred, true)
        if mae < best_mae:
            best_p, best_mae = p, mae
    for mc in (3, 5, 6, 8):
        Fl = F.copy(); Fl[:, 0] = 0; Fl[:, mc + 1:] = 0
        summarize(f"ideal low-pass on dtheta(0), m<=%d" % mc, np.fft.irfft(Fl, n=N, axis=1).argmax(1), true)

    # ---- 4. Replication of the paper's box-car numbers (fixed A = 1.75) ----
    A2 = np.full(B, 1.75)
    rep = simulate(A2[:, None] + noise(B))
    E5 = energy_code(rep["frames"][5, :, 0], rep["frames"][5, :, 1])
    E10 = energy_code(rep["frames"][10, :, 0], rep["frames"][10, :, 1])
    print("\nreplication, A = 1.75 fixed (paper Fig. 5 says K17 -> 5.41 at 'timestamp 5'; K1 -> 22.5):")
    print("  sample 5 (t=2.5): " + "  ".join(f"K{K}:{cdist(boxcar(E5, K).argmax(1), rep['site']).mean():.2f}" for K in (1, 15, 17, 19)))
    print("  sample 10 (t=5):  " + "  ".join(f"K{K}:{cdist(boxcar(E10, K).argmax(1), rep['site']).mean():.2f}" for K in (1, 15, 17, 19)))

    np.savez_compressed(out, A=A, th0=th0, site=main["site"], time=main["time"],
                        site_sym=main["site_sym"], frames=main["frames"].astype(np.float32),
                        sigma=sig, Agrid=Agrid)
    print("saved", out)
