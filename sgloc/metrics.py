"""Evaluation metrics. Distances are measured around the ring unless the name says "linear"."""
import numpy as np


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
    ring = ring_distance(pred[keep], true[keep], n_sites)
    linear = np.abs(pred[keep] - true[keep])
    out = {"n": int(keep.sum()), "mae": float(ring.mean()), "mae_linear": float(linear.mean()),
           "over_10": float((ring > 10).mean())}
    for n in (0, 1, 2, 3, 4):
        out[f"acc_{n}"] = float((ring <= n).mean())
    for n in (1, 2, 4):
        out[f"acc_{n}_linear"] = float((linear <= n).mean())
    return out
