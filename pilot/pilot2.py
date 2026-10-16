"""Pilot 2: label-definition sensitivity, Fourier-band knockouts, and K*(A) prediction."""
import sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from pilot import N, accel, energy_code, energy_sym, simulate, cdist, boxcar  # noqa: E402

rng = np.random.default_rng(1)
d = np.load(sys.argv[2])
th0, site, A = d["th0"], d["site"], d["A"]


def trace(th0, dt=0.05, t_lo=30.0, t_hi=120.0, sample=0.5):
    """Return per-sample code and symmetric energies between t_lo and t_hi."""
    th, om = th0.copy(), np.zeros_like(th0)
    stride = int(round(sample / dt))
    Ec, Es, ts = [], [], []
    for step in range(int(round(t_hi / dt)) + 1):
        if step % stride == 0 and step * dt >= t_lo:
            Ec.append(energy_code(th, om)); Es.append(energy_sym(th, om)); ts.append(step * dt)
        a1 = accel(th); v1 = om
        a2 = accel(th + 0.5 * dt * v1); v2 = om + 0.5 * dt * a1
        a3 = accel(th + 0.5 * dt * v2); v3 = om + 0.5 * dt * a2
        a4 = accel(th + dt * v3); v4 = om + dt * a3
        th = th + dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        om = om + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    return np.array(ts), np.array(Ec), np.array(Es)


def first_cross(E, Etot, thr=0.1):
    hit = E.max(2) > thr * Etot[None]
    k = hit.argmax(0)
    ok = hit.any(0)
    s = E[k, np.arange(E.shape[1])].argmax(1)
    return np.where(ok, s, -1), np.where(ok, k, -1)


# ---- 1. Why do the code and symmetric labels disagree? ----
S = 600
ts, Ec, Es = trace(th0[:S])
Etot = energy_code(th0[:S], np.zeros_like(th0[:S])).sum(1)
sc, kc = first_cross(Ec, Etot)
ss, ks = first_cross(Es, Etot)
# period-averaged symmetric energy (running mean over ~one carrier period, 16 samples = 8 time units)
w = 16
Eavg = np.cumsum(Es, 0)
Eavg = (Eavg[w:] - Eavg[:-w]) / w
sa, ka = first_cross(Eavg, Etot)
diff = (ss - sc + N // 2) % N - N // 2
print("label definition check on", S, "runs")
print("  code vs symmetric: same site %.1f%%, |diff|<=1 %.1f%%" % (np.mean(diff == 0) * 100, np.mean(np.abs(diff) <= 1) * 100))
big = np.abs(diff) >= 2
print("  among |diff|>=2: mean crossing-time gap (sym - code) = %.2f time units" % ((ks[big] - kc[big]).mean() * 0.5))
print("  among |diff|<=1: mean crossing-time gap (sym - code) = %.2f time units" % ((ks[~big] - kc[~big]).mean() * 0.5))
# does the hotspot keep moving after it first appears? track argmax of period-averaged energy
drift = []
for i in range(S):
    if ka[i] < 0:
        continue
    k0 = ka[i]
    k1 = min(k0 + 20, Eavg.shape[0] - 1)  # 10 time units later
    drift.append(((Eavg[k1, i].argmax() - Eavg[k0, i].argmax() + N // 2) % N) - N // 2)
drift = np.array(drift)
print("  hotspot drift over the 10 time units after formation (period-averaged): "
      "|drift|<=1 %.1f%%, mean |drift| %.2f sites" % (np.mean(np.abs(drift) <= 1) * 100, np.abs(drift).mean()))
for name, lab in (("symmetric instantaneous", ss), ("symmetric period-averaged", sa)):
    dd = cdist(lab, sc)
    print(f"  {name:28s} vs code label: MAE {dd.mean():.2f}, <=1 {np.mean(dd <= 1)*100:.1f}%")

# ---- 2. Fourier-band knockouts of the initial noise ----
M = 1000
base = site[:M]
F = np.fft.rfft(th0[:M], axis=1)
print("\nband knockouts (re-simulate with part of the initial noise removed; shift of the site label):")


def knock(keep):
    keep = keep.copy()
    keep[0] = True  # never touch m = 0 (the swing amplitude itself)
    G = F.copy()
    G[:, ~keep] = 0
    return np.fft.irfft(G, n=N, axis=1)


ms = np.arange(F.shape[1])
for name, keep in (("keep m<=5 only (unstable band)", ms <= 5),
                   ("keep m<=12", ms <= 12),
                   ("keep m<=25", ms <= 25),
                   ("remove m<=5 (stable band only)", (ms == 0) | (ms > 5))):
    r = simulate(knock(keep))
    dd = cdist(r["site"], base)
    print(f"  {name:34s} MAE {dd.mean():6.2f}  <=1 {np.mean(dd <= 1)*100:5.1f}%  <=3 {np.mean(dd <= 3)*100:5.1f}%  >10 {np.mean(dd > 10)*100:5.1f}%")

print("\nband-limited perturbations, rms 1e-3 rad (like sensor error on the initial state):")
for name, band in (("only in m<=5", (ms >= 1) & (ms <= 5)), ("only in m>=6", ms >= 6), ("white (all m)", ms >= 1)):
    P = np.fft.rfft(rng.standard_normal((M, N)), axis=1)
    P[:, ~band] = 0
    p = np.fft.irfft(P, n=N, axis=1)
    p *= 1e-3 / p.std(1, keepdims=True)
    r = simulate(th0[:M] + p)
    dd = cdist(r["site"], base)
    print(f"  {name:16s} moved>1 {np.mean(dd > 1)*100:5.1f}%  moved>3 {np.mean(dd > 3)*100:5.1f}%  MAE {dd.mean():.2f}")

# ---- 3. Does the Floquet band edge predict the best box-car width? ----
print("\nbest box-car width vs amplitude (prediction K* = N/(m_max+1)):")
for Aval, pred in ((1.65, 25.0), (1.75, 16.7), (1.85, 14.3)):
    r = simulate(Aval + 0.05 * (rng.random((2000, N)) - 0.5))
    E0 = energy_code(r["frames"][0, :, 0], r["frames"][0, :, 1])
    Ks = np.arange(5, 40, 2)
    mae = np.array([cdist(boxcar(E0, K).argmax(1), r["site"]).mean() for K in Ks])
    print(f"  A={Aval}: predicted K*={pred:.1f}, measured K*={Ks[mae.argmin()]} (MAE {mae.min():.2f}); "
          + " ".join(f"{K}:{m:.1f}" for K, m in zip(Ks, mae)))
