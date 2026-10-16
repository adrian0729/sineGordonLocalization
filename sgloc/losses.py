"""Loss functions."""
import math

import torch
import torch.nn.functional as F


def v1_loss(outputs, y_localizes, y_site, w_localizes=0.5, w_site=1.0):
    """The paper's loss: 0.5 x binary cross-entropy on "does it localize" + 1.0 x sparse
    categorical cross-entropy on the site, ignoring runs labeled -1 (never localized).

    Returns the total and the two parts (detached, still on the device).
    """
    localizes_logit, site_logits = outputs
    localizes = F.binary_cross_entropy_with_logits(localizes_logit, y_localizes)
    site = F.cross_entropy(site_logits, y_site, ignore_index=-1)
    total = w_localizes * localizes + w_site * site
    return total, {"localizes": localizes.detach(), "site": site.detach()}


def ring_distance_matrix(n_sites, device=None):
    i = torch.arange(n_sites, device=device)
    d = (i[:, None] - i[None, :]).abs()
    return torch.minimum(d, n_sites - d).float()


def soft_targets(y, n_sites, sigma):
    """Circular Gaussian over sites centred on the true site (sigma in sites)."""
    d = ring_distance_matrix(n_sites, y.device)[y]
    w = torch.exp(-0.5 * (d / sigma) ** 2)
    return w / w.sum(dim=1, keepdim=True)


def circular_mean_site(scores):
    """Site at the circular mean of softmax(scores): what a circular regression predicts."""
    n_sites = scores.shape[-1]
    phase = 2 * math.pi * torch.arange(n_sites, device=scores.device) / n_sites
    p = scores.softmax(dim=-1)
    angle = torch.atan2((p * phase.sin()).sum(-1), (p * phase.cos()).sum(-1))
    return angle, torch.remainder(torch.round(angle * n_sites / (2 * math.pi)), n_sites).long()


def _mean(values, weight):
    """Batch mean; with per-run weights (1 for runs that localize, 0 for the rest), the mean over the
    runs that count."""
    return values.mean() if weight is None else (values * weight).sum() / weight.sum().clamp(min=1.0)


class SiteLoss:
    """Configurable site loss for the v2 models.

    kind="ce": cross-entropy, optionally with circular-Gaussian soft labels (soft_sigma > 0)
    and an expected ring-distance term (weight `distance`). kind="angle": circular regression,
    1 - cos(predicted angle - true angle) with the angle at the circular mean of the scores.
    Optional auxiliary terms: "time" (MSE on standardized log localization time), "profile"
    (MSE on a later energy profile), "localizes" (binary cross-entropy, as in v1).

    A batch with a "site_weight" entry (added when a config sets ignore_unlocalized, as the
    damping group does) leaves runs that never localize (site -1) out of the site and time
    terms, as v1's loss did. Without it such runs are scored as site -1, i.e. the last site.
    """

    def __init__(self, kind="ce", soft_sigma=0.0, distance=0.0, time=0.0, profile=0.0, localizes=0.0):
        self.kind, self.soft_sigma, self.distance = kind, soft_sigma, distance
        self.weights = {"time": time, "profile": profile, "localizes": localizes}
        self._dist = None

    def __call__(self, out, batch):
        scores, y = out["site"], batch["site"]
        weight = batch.get("site_weight")
        if weight is not None:
            y = y.clamp(min=0)  # any valid index; these runs get weight 0
        n_sites = scores.shape[-1]
        if self.kind == "angle":
            angle, _ = circular_mean_site(scores)
            loss = _mean(1 - torch.cos(angle - 2 * math.pi * y / n_sites), weight)
        else:
            logp = scores.log_softmax(dim=-1)
            if self.soft_sigma > 0:
                loss = _mean(-(soft_targets(y, n_sites, self.soft_sigma) * logp).sum(-1), weight)
            elif weight is None:
                loss = F.nll_loss(logp, y)
            else:
                loss = _mean(F.nll_loss(logp, y, reduction="none"), weight)
            if self.distance > 0:
                if self._dist is None or self._dist.device != scores.device:
                    self._dist = ring_distance_matrix(n_sites, scores.device)
                loss = loss + self.distance * _mean((logp.exp() * self._dist[y]).sum(-1), weight)
        if self.weights["time"]:
            if weight is None:
                loss = loss + self.weights["time"] * F.mse_loss(out["time"], batch["log_time"])
            else:
                loss = loss + self.weights["time"] * _mean((out["time"] - batch["log_time"]) ** 2, weight)
        if self.weights["profile"]:
            loss = loss + self.weights["profile"] * F.mse_loss(out["profile"], batch["profile"])
        if self.weights["localizes"]:
            loss = loss + self.weights["localizes"] * F.binary_cross_entropy_with_logits(out["localizes"], batch["localizes"])
        return loss

    def predict(self, out):
        if self.kind == "angle":
            return circular_mean_site(out["site"])[1]
        return out["site"].argmax(dim=-1)
