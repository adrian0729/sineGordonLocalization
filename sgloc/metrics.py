"""Evaluation metrics. Distances are measured around the ring unless the name says "linear"."""
import numpy as np
from scipy.stats import rankdata


def ring_distance(a, b, n_sites=100):
    """Distance between sites a and b measured around the ring."""
    d = np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)) % n_sites
    return np.minimum(d, n_sites - d)


def site_metrics(pred, true, n_sites=100):
    """Location metrics over runs that localized (true >= 0).

    `acc_n` is the fraction within n sites around the ring; the `_linear` variants use plain
    |pred - true|, which is how the paper's tolerance metric appears to be defined.
    """
    pred, true = np.asarray(pred, dtype=int), np.asarray(true, dtype=int)
    keep = true >= 0
    if not keep.any():  # e.g. a strongly damped chain: there is no site to score
        return {"n": 0}
    ring = ring_distance(pred[keep], true[keep], n_sites)
    linear = np.abs(pred[keep] - true[keep])
    out = {"n": int(keep.sum()), "mae": float(ring.mean()), "mae_linear": float(linear.mean()),
           "over_10": float((ring > 10).mean())}
    for n in (0, 1, 2, 3, 4):
        out[f"acc_{n}"] = float((ring <= n).mean())
    for n in (1, 2, 4):
        out[f"acc_{n}_linear"] = float((linear <= n).mean())
    return out


def localizes_metrics(prob, localizes):
    """The "does it localize" head against the truth (by t_final): accuracy at p = 0.5, the area
    under the ROC curve, the share of localizing runs called non-localizing (missed) and of
    non-localizing runs called localizing (false alarms)."""
    prob, truth = np.asarray(prob, dtype=float), np.asarray(localizes, dtype=bool)
    call = prob > 0.5
    out = {"n": int(truth.size), "base_rate": float(truth.mean()), "accuracy": float((call == truth).mean())}
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if n_pos and n_neg:
        out["auc"] = float((rankdata(prob)[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    if n_pos:
        out["missed"] = float((~call[truth]).mean())
    if n_neg:
        out["false_alarm"] = float(call[~truth].mean())
    return out
