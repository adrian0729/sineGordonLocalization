"""How readable is the localization site from a single snapshot at later times?

    python scripts/diagnose_late_frames.py

v1 as described (frames at tau <= 4.5) reached 61.7% within ±1 on the test pool, against the
paper's 94.4%. This script checks how late an observation has to be before simple readouts
(the raw energy peak, or a short box-car) reach the paper's numbers, as a clue to whether the
paper's inputs extended later than stated. Writes results/late_frames.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, metrics, sim  # noqa: E402


def boxcar(x, K):
    return sum(np.roll(x, j, axis=-1) for j in range(-(K // 2), K // 2 + 1)) / K


def main():
    test = data.load_pool(ROOT / "data" / "pool_test.npz", 5000)
    p = sim.Params()
    taus = [0, 4.5, 10, 20, 30, 35, 40, 45, 50, 55, 60]
    stride = round(p.sample / p.dt)
    theta, omega = test["theta0"].copy(), np.zeros_like(test["theta0"])
    out = []
    for k in range(round(max(taus) / p.sample) + 1):
        tau = k * p.sample
        if tau in taus:
            E = sim.site_energy(theta, omega, p.g, p.h, "right")
            row = {"tau": tau, "localized_already": float(np.mean(test["time_right"] <= tau))}
            for name, pred in (("peak", E.argmax(1)), ("boxcar5", boxcar(E, 5).argmax(1)),
                               ("boxcar17", boxcar(E, 17).argmax(1))):
                m = metrics.site_metrics(pred, test["site_right"])
                row[name] = {"mae": round(m["mae"], 2), "acc_1": round(m["acc_1"], 4), "acc_3": round(m["acc_3"], 4)}
            out.append(row)
            print(f"tau {tau:4.1f}  already localized {row['localized_already']:6.1%}  "
                  + "  ".join(f"{n}: mean {row[n]['mae']:5.2f} ±1 {row[n]['acc_1']:6.1%}"
                              for n in ("peak", "boxcar5", "boxcar17")), flush=True)
        for _ in range(stride):
            theta, omega = sim.rk4_step(theta, omega, p.dt, p.g, p.h)
    (ROOT / "results" / "late_frames.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
