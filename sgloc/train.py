"""Training loops (v1: Adam + early stopping like Keras; v2: fixed-step schedules) and evaluation."""
import math
import time

import numpy as np
import torch

from . import metrics


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023): step along the sign of an interpolated momentum."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                m = self.state[p].setdefault("exp_avg", torch.zeros_like(p))
                p.add_(torch.sign(m * beta1 + p.grad * (1 - beta1)), alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)


def make_optimizer(params, name="adamw", lr=3e-3, weight_decay=1e-4, **_):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, eps=1e-7)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def make_schedule(optimizer, kind, steps, lr, warmup=0):
    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    if kind == "cosine":
        def factor(s):
            if s < warmup:
                return (s + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    raise ValueError(kind)


def predict_sites(model, loss_fn, x, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            preds.append(loss_fn.predict(model(x[i:i + batch_size])).cpu())
    return torch.cat(preds).numpy()


def fit_v2(model, loss_fn, train_split, val_split, *, steps, batch_size=256, optim=None, eval_every=None,
           seed=0, patience=None, log=print):
    """Mini-batch training for a fixed number of steps with the given optimizer and schedule.

    The validation mean distance is checked every `eval_every` steps and the best weights are
    restored at the end. With `patience` (in evaluations), training also stops early.
    Returns the history and the best step.
    """
    optim = dict(optim or {})
    warmup = int(optim.pop("warmup_fraction", 0.0) * steps)
    schedule = optim.pop("schedule", "cosine")
    clip = optim.pop("clip", None)
    optimizer = make_optimizer(model.parameters(), **optim)
    scheduler = make_schedule(optimizer, schedule, steps, optim.get("lr", 3e-3), warmup)
    shuffle = torch.Generator().manual_seed(seed)
    device = train_split["x"].device
    n = len(train_split["x"])
    eval_every = eval_every or max(1, steps // 30)
    y_val = val_split["site"].cpu().numpy()
    best_mae, best_state, best_step, waited, step, running, history = float("inf"), None, 0, 0, 0, 0.0, []
    start = time.time()
    while step < steps:
        order = torch.randperm(n, generator=shuffle).to(device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            model.train()
            loss = loss_fn(model(train_split["x"][idx]), {k: v[idx] for k, v in train_split.items()})
            if not torch.isfinite(loss):
                log(f"step {step}: loss is not finite, stopping")
                steps = step
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
            if step % eval_every == 0 or step == steps:
                mae = float(metrics.ring_distance(predict_sites(model, loss_fn, val_split["x"]), y_val).mean())
                if mae < best_mae:
                    best_mae, best_step, waited = mae, step, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    waited += 1
                history.append({"step": step, "train_loss": running / eval_every, "val_mae": mae,
                                "seconds": round(time.time() - start, 1)})
                log(f"step {step:6d}  train loss {running / eval_every:.4f}  val mean distance {mae:.3f}  "
                    f"best {best_mae:.3f} @ {best_step}  {time.time() - start:.0f}s")
                running = 0.0
                if patience is not None and waited >= patience:
                    steps = step
            if step >= steps:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_step


def lr_range_test(model, loss_fn, train_split, *, batch_size=256, optim=None, lr_min=1e-5, lr_max=1.0,
                  steps=300, seed=0):
    """Increase the learning rate geometrically each step and record the smoothed training loss."""
    optim = {k: v for k, v in (optim or {}).items() if k not in ("schedule", "warmup_fraction", "clip")}
    optim["lr"] = lr_min
    optimizer = make_optimizer(model.parameters(), **optim)
    gamma = (lr_max / lr_min) ** (1 / steps)
    shuffle = torch.Generator().manual_seed(seed)
    n = len(train_split["x"])
    out, smooth, best = [], None, float("inf")
    for s in range(steps):
        lr = lr_min * gamma ** s
        for group in optimizer.param_groups:
            group["lr"] = lr
        idx = torch.randint(0, n, (batch_size,), generator=shuffle).to(train_split["x"].device)
        model.train()
        loss = loss_fn(model(train_split["x"][idx]), {k: v[idx] for k, v in train_split.items()})
        if not torch.isfinite(loss):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
        out.append((lr, smooth))
        best = min(best, smooth)
        if smooth > 4 * best and s > 20:
            break
    return out


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_loss(model, loss_fn, split, batch_size=1024):
    model.eval()
    total, n = 0.0, len(split["x"])
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = slice(i, i + batch_size)
            loss, _ = loss_fn(model(split["x"][batch]), *(t[batch] for t in split["targets"]))
            total += loss.item() * len(split["x"][batch])
    return total / n


def fit(model, loss_fn, train, val, *, max_epochs=100, patience=10, batch_size=64, lr=1e-3,
        eps=1e-7, seed=0, log=print):
    """Train with Adam, stop after `patience` epochs without a lower validation loss, and
    restore the weights from the best epoch.

    `train` and `val` are dicts with "x" (a tensor) and "targets" (a tuple of tensors), already
    on the model's device. Returns the per-epoch history and the best epoch.
    """
    shuffle = torch.Generator().manual_seed(seed)
    device = train["x"].device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
    n = len(train["x"])
    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        order = torch.randperm(n, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            loss, _ = loss_fn(model(train["x"][idx]), *(t[idx] for t in train["targets"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.detach() * len(idx)
        train_loss = running.item() / n
        val_loss = mean_loss(model, loss_fn, val)
        if val_loss < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            waited += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": round(time.time() - start, 1)})
        log(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"best {best_loss:.4f} (epoch {best_epoch})  {time.time() - start:.0f}s")
        if waited >= patience:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def fit_lbfgs(score_fn, params, y, l2=0.0, max_iter=1000):
    """Minimize mean site cross-entropy + l2 * sum of squared parameters with full-batch L-BFGS.

    For RingLinear the problem is convex, so this reaches the global optimum. `score_fn()`
    returns the site scores for the whole training set. Returns the final training loss.
    """
    params = list(params)
    optimizer = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(score_fn(), y)
        if l2 > 0:
            loss = loss + l2 * sum((p ** 2).sum() for p in params)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.nn.functional.cross_entropy(score_fn(), y).item()


def predict(model, x, batch_size=1024):
    """Return P(localizes) and the site log-probabilities as numpy arrays."""
    model.eval()
    probs, logps = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            localizes_logit, site_logits = model(x[i:i + batch_size])
            probs.append(torch.sigmoid(localizes_logit).cpu())
            logps.append(torch.log_softmax(site_logits, dim=1).cpu())
    return torch.cat(probs).numpy(), torch.cat(logps).numpy()


def evaluate(model, x, y_site, batch_size=1024):
    """Location metrics plus localization accuracy and the site negative log-likelihood."""
    p_localizes, logp = predict(model, x, batch_size)
    y_site = np.asarray(y_site, dtype=int)
    pred = logp.argmax(axis=1)
    out = metrics.site_metrics(pred, y_site)
    out["acc_localizes"] = float(((p_localizes > 0.5) == (y_site >= 0)).mean())
    keep = y_site >= 0
    out["nll"] = float(-logp[keep, y_site[keep]].mean())
    return out, pred
