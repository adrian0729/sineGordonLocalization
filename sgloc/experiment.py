"""One experiment = one config dict -> train, evaluate, append a record to results/runs.jsonl.

Config keys: phase, group, name, note (why the run exists), seed, n_train, view (sgloc.views.View
fields), test_view (optional overrides for the test inputs), model (sgloc.models.build spec),
loss (sgloc.losses.SiteLoss fields), optim (optimizer, schedule, steps or epochs, batch_size).
Optional: label (sgloc.sim.LABELS, or "hump" once scripts/add_labels.py has added it to the
pools), pools (other pool files), ignore_unlocalized (leave runs that never localize out of the
site and time losses; for damped pools), augment ("mirror": add the mirror image of every
training run).
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data, losses, metrics, models, sim, train, views

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "runs.jsonl"
TIME_MEAN, TIME_STD = math.log(57.0), 0.2  # standardization of log localization time


def _jsonable(cfg):
    return json.loads(json.dumps(cfg, default=list))


LABEL_KEYS = ("phase", "group", "name", "note")


def config_id(cfg):
    """Stable id of everything that affects the result. Phase, group, name and note are labels
    only, so the same configuration listed in two ablation groups runs once."""
    key = {k: v for k, v in _jsonable(cfg).items() if k not in LABEL_KEYS}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def completed_ids():
    if not RESULTS.exists():
        return set()
    with open(RESULTS) as f:
        return {json.loads(line).get("config_id") for line in f if line.strip()}


def load_pools(n_train, names=None):
    """The train/val/test pools; `names` can swap any of them for another file in data/."""
    names = names or {}
    path = {s: ROOT / "data" / (names.get(s, f"pool_{s}") + ".npz") for s in ("train", "val", "test")}
    return {"train": data.load_pool(path["train"], n_train), "val": data.load_pool(path["val"]),
            "test": data.load_pool(path["test"])}


def true_system(pool):
    """The dynamics a pool was simulated with, when they differ from the nominal model: other g or
    h, a per-site gravity map (impurities) and a per-run damping coefficient."""
    p = pool["meta"].get("params", {})
    dynamics = sim.Params(**p) if p and (p["g"], p["h"]) != (sim.Params().g, sim.Params().h) else None
    return dynamics, pool.get("onsite"), pool.get("gamma")


def build_inputs(view, split, pool, workers=1):
    dynamics, onsite, gamma = true_system(pool)
    calibration = None
    if view.calibration:
        calibration = (1 - onsite) if onsite is not None else np.zeros_like(pool["theta0"], dtype=np.float32)
    return views.build(view, split, pool["theta0"], workers, dynamics=dynamics,
                       onsite=None if onsite is None else onsite.astype(float), calibration=calibration, gamma=gamma)


def prebuild(cfg, pools, workers=8):
    """Fill the trajectory cache for every split of a config."""
    split_views = views_for(cfg)
    for s in ("train", "val", "test"):
        dynamics, onsite, gamma = true_system(pools[s])
        views.ensure(split_views[s], s, pools[s]["theta0"], workers, dynamics=dynamics,
                     onsite=None if onsite is None else onsite.astype(float), gamma=gamma)
        if cfg.get("loss", {}).get("profile"):
            views.profile_target(s, pools[s]["theta0"], workers)


def views_for(cfg):
    view = views.View(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["view"].items()})
    test = dict(cfg["view"], **cfg.get("test_view", {}))
    return {"train": view, "val": view,
            "test": views.View(**{k: tuple(v) if isinstance(v, list) else v for k, v in test.items()})}


# each label's source under reflection
MIRROR_LABEL = {"right": "left", "left": "right", "sym": "sym", "avg": "avg", "hump": "hump"}


def mirror_pool(pool):
    """The pool of the mirror-image chains (site n -> N-1-n), which the ring's reflection symmetry makes
    valid runs. The one-sided labels swap: v1's label of a mirrored chain (each site owns the spring to
    its right) is the mirror of the original's left-bond label. "sym" and "avg" are simply mirrored."""
    n_sites = pool["theta0"].shape[1]
    out = dict(pool, theta0=pool["theta0"][:, ::-1])
    if "onsite" in pool:
        out["onsite"] = pool["onsite"][:, ::-1]
    for name, source in MIRROR_LABEL.items():
        if f"site_{source}" in pool:
            site = pool[f"site_{source}"]
            out[f"site_{name}"] = np.where(site >= 0, n_sites - 1 - site, site).astype(site.dtype)
            out[f"time_{name}"] = pool[f"time_{source}"]
    return out


def mirror_inputs(view, x):
    """Inputs of the mirror-image chains. Reversing the site axis mirrors angles, velocities and the static
    per-site channels exactly, but not one-sided site energies or a sparse sensor layout."""
    if "E" in view.channels or view.stride > 1:
        raise ValueError("mirror augmentation needs inputs that reverse exactly: no site-energy channel "
                         "(each site owns the spring to its right) and every pendulum observed")
    return np.ascontiguousarray(x[..., ::-1])


def prepare(cfg, pools, workers=1):
    """Standardized input tensors and targets for train/val/test. With cfg["augment"] = "mirror", the
    training split also holds the mirror image of every run (twice the runs; same input scaling)."""
    split_views = views_for(cfg)
    xs = {s: build_inputs(split_views[s], s, pools[s], workers) for s in pools}
    channel_axis = 1 if split_views["train"].layout == "channels" else 2
    axes = tuple(a for a in range(xs["train"].ndim) if a != channel_axis)
    mean = np.nanmean(xs["train"], axis=axes, keepdims=True)
    std = np.nanstd(xs["train"], axis=axes, keepdims=True) + 1e-12
    profiles = None
    if cfg.get("loss", {}).get("profile", 0):
        profiles = {s: views.profile_target(s, pools[s]["theta0"], workers) for s in pools}
        p_mean, p_std = profiles["train"].mean(), profiles["train"].std()
    augment = cfg.get("augment")
    if augment == "mirror":  # treated as a fourth split below, then appended to the training runs
        xs["train_mirror"] = mirror_inputs(split_views["train"], xs["train"])
        pools = dict(pools, train_mirror=mirror_pool(pools["train"]))
        if profiles is not None:  # the profile uses the symmetric energy, so it simply mirrors
            profiles["train_mirror"] = np.ascontiguousarray(profiles["train"][:, ::-1])
    elif augment is not None:
        raise ValueError(f"unknown augmentation {augment!r}")
    label = cfg.get("label", "right")  # which localization-site convention to learn (sim.LABELS)
    out = {}
    for s, x in xs.items():
        x = np.nan_to_num((x - mean) / std, nan=0.0).astype(np.float32)
        t = pools[s][f"time_{label}"].astype(np.float64)
        split = {"x": torch.from_numpy(x),
                 "site": torch.from_numpy(pools[s][f"site_{label}"].astype(np.int64)),
                 "localizes": torch.from_numpy((pools[s][f"site_{label}"] >= 0).astype(np.float32)),
                 "log_time": torch.from_numpy(((np.log(np.nan_to_num(t, nan=150.0)) - TIME_MEAN) / TIME_STD).astype(np.float32))}
        if profiles is not None:
            split["profile"] = torch.from_numpy(((profiles[s] - p_mean) / p_std).astype(np.float32))
        if cfg.get("ignore_unlocalized"):  # runs that never localize (site -1) carry no site or time loss
            split["site_weight"] = split["localizes"]
        out[s] = split
    if "train_mirror" in out:
        mirrored = out.pop("train_mirror")
        out["train"] = {k: torch.cat([v, mirrored[k]]) for k, v in out["train"].items()}
    return out, (mean, std)


def run(cfg, workers=1, threads=None, device="cpu", log=print):
    cfg = copy.deepcopy(cfg)
    cid = config_id(cfg)
    if threads:
        torch.set_num_threads(threads)
    run_dir = ROOT / "runs" / f"{cfg['name']}_s{cfg['seed']}_{cid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "log.txt", "w")

    def say(message):
        log_file.write(message + "\n")
        log_file.flush()

    start = time.time()
    pools = load_pools(cfg["n_train"], cfg.get("pools"))
    splits, (scale_mean, scale_std) = prepare(cfg, pools, workers)
    loss_spec = dict(cfg.get("loss", {}))
    model_spec = dict(cfg["model"])
    aux = [k for k in ("time", "profile", "localizes") if loss_spec.get(k)]
    if aux and model_spec.get("kind", "ring") != "v1":
        model_spec["aux"] = aux
    torch.manual_seed(cfg["seed"])
    model = models.build(model_spec, tuple(splits["train"]["x"].shape[1:])).to(device)
    splits = {s: {k: v.to(device) for k, v in split.items()} for s, split in splits.items()}
    loss_fn = losses.SiteLoss(**loss_spec)
    optim = dict(cfg["optim"])
    batch_size = optim.pop("batch_size", 256)
    steps = optim.pop("steps", None) or optim.pop("epochs", 30) * math.ceil(cfg["n_train"] / batch_size)
    optim.pop("epochs", None)
    patience = optim.pop("patience", None)
    eval_every = optim.pop("eval_every", None)
    say(f"{cfg['name']} seed {cfg['seed']} ({cid}): {cfg['note']}\nparams {models.count_parameters(model)}, "
        f"inputs {tuple(splits['train']['x'].shape)}, steps {steps}, data ready in {time.time() - start:.0f}s")
    train_start = time.time()
    val_check = {k: v[:5000] for k, v in splits["val"].items()}  # checkpoint selection; final metrics use all of val
    history, best_step = train.fit_v2(model, loss_fn, splits["train"], val_check, steps=steps,
                                      batch_size=batch_size, optim=optim, eval_every=eval_every,
                                      seed=cfg["seed"], patience=patience, log=say)
    train_seconds = time.time() - train_start

    results = {}
    for s in ("val", "test"):
        pred = train.predict_sites(model, loss_fn, splits[s]["x"])
        label = cfg.get("label", "right")
        results[s] = metrics.site_metrics(pred, pools[s][f"site_{label}"])
        if label != "right":  # also score against v1's label, for comparison
            results[s]["right_label"] = metrics.site_metrics(pred, pools[s]["site_right"])
        if loss_spec.get("kind", "ce") == "ce":
            model.eval()
            with torch.no_grad():
                logp = torch.cat([model(splits[s]["x"][i:i + 512])["site"].log_softmax(-1)
                                  for i in range(0, len(splits[s]["x"]), 512)])
            y, masked = splits[s]["site"], "site_weight" in splits[s]
            nll = -logp[torch.arange(len(y), device=y.device), y.clamp(min=0) if masked else y]
            results[s]["nll"] = float(nll[y >= 0].mean() if masked else nll.mean())
        if loss_spec.get("localizes"):
            results[s]["localizes"] = metrics.localizes_metrics(localizes_probability(model, splits[s]["x"]),
                                                                pools[s][f"site_{label}"] >= 0)
        if s == "test":
            np.save(run_dir / "pred_test.npy", pred.astype(np.int16))
    (run_dir / "history.json").write_text(json.dumps(history))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "model.pt")  # with the scaler, enough to re-evaluate on new test sets
    np.savez(run_dir / "scaler.npz", mean=scale_mean, std=scale_std)
    (run_dir / "config.json").write_text(json.dumps(_jsonable(cfg), indent=1))
    (run_dir / "model_spec.json").write_text(json.dumps({"spec": _jsonable(model_spec),
                                                         "input_shape": list(splits["train"]["x"].shape[1:])}))
    record = {"config_id": cid, "phase": cfg["phase"], "group": cfg["group"], "name": cfg["name"],
              "note": cfg["note"], "seed": cfg["seed"], "config": _jsonable(cfg),
              "params": models.count_parameters(model), "steps": steps, "best_step": best_step,
              "train_seconds": round(train_seconds), "device": str(device), "val": results["val"], "test": results["test"],
              "run_dir": str(run_dir.relative_to(ROOT)), "created": time.strftime("%Y-%m-%d %H:%M")}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    say(f"test: mean {results['test']['mae']:.3f}  ±1 {results['test']['acc_1']:.2%}  "
        f"±3 {results['test']['acc_3']:.2%}  >10 {results['test']['over_10']:.2%}  ({train_seconds:.0f}s)")
    return record


def localizes_probability(model, x, batch_size=512):
    """The "does it localize" head's probability for every run."""
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(x[i:i + batch_size])["localizes"]).cpu()
                          for i in range(0, len(x), batch_size)]).numpy()


def evaluate_saved(run_dir, pool_name, view_override=None, split="test", workers=4, label_pool=None):
    """Re-evaluate a saved run on another pool (e.g. a robustness pool), optionally with a
    different test view (e.g. more sensor noise). Inputs are standardized with the run's scaler.
    Sites are scored against the label the run was trained on; `label_pool` takes the labels from
    another pool with the same initial conditions (inputs from one chain, outcome from another)."""
    run_dir = ROOT / run_dir
    if (run_dir / "config.json").exists():
        cfg = json.loads((run_dir / "config.json").read_text())
    else:  # runs recorded before config.json was saved
        with open(RESULTS) as f:
            cfg = next(json.loads(l)["config"] for l in f if json.loads(l).get("run_dir") == str(run_dir.relative_to(ROOT)))
    spec = json.loads((run_dir / "model_spec.json").read_text())
    scaler = np.load(run_dir / "scaler.npz")
    view_cfg = dict(cfg["view"], **(view_override or {}))
    view = views.View(**{k: tuple(v) if isinstance(v, list) else v for k, v in view_cfg.items()})
    pool = data.load_pool(ROOT / "data" / f"{pool_name}.npz")
    x = build_inputs(view, split, pool, workers)
    x = torch.from_numpy(np.nan_to_num((x - scaler["mean"]) / scaler["std"], nan=0.0).astype(np.float32))
    model = models.build(spec["spec"], tuple(x.shape[1:]))
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    loss_fn = losses.SiteLoss(**cfg.get("loss", {}))
    pred = train.predict_sites(model, loss_fn, x)
    if label_pool is not None:
        other = data.load_pool(ROOT / "data" / f"{label_pool}.npz")
        assert np.array_equal(other["theta0"], pool["theta0"]), "label pool must share the initial conditions"
        pool = other
    truth = pool[f"site_{cfg.get('label', 'right')}"]
    out = metrics.site_metrics(pred, truth, n_sites=pool["theta0"].shape[1])
    if "localizes" in spec["spec"].get("aux", []):
        out["localizes"] = metrics.localizes_metrics(localizes_probability(model, x), truth >= 0)
    return out
