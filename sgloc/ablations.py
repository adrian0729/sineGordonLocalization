"""The ablation tables of the study, one function per phase (written up in REPORT.md).

Each config's `note` records the hypothesis it tests, so results/runs.jsonl says why every run
exists. Defaults for later phases are set from earlier results; each such choice is commented.
"""
import copy

STEPS = 7050  # 30 epochs of 60k runs at batch 256; fixed so data-size comparisons use the same budget
SCREEN_STEPS = 3525  # 15-epoch screening budget for Phases 3-5; Phase 6 retrains the final choices at STEPS

V1_INFO = {"channels": ["E"], "frames": list(range(10)), "theta0": True, "layout": "channels"}

BASE = {
    "phase": 2, "group": "", "name": "", "note": "", "seed": 0, "n_train": 60_000,
    "view": V1_INFO,
    "model": {"kind": "ring", "front": "early", "width": 32, "depth": 2, "kernel": 5, "modes": 6,
              "local": True, "spectral": True, "padding": "circular", "head": "site"},
    "loss": {"kind": "ce"},
    "optim": {"name": "adamw", "lr": 3e-3, "weight_decay": 1e-4, "schedule": "cosine",
              "warmup_fraction": 0.03, "steps": STEPS, "batch_size": 256},
}


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Table:
    def __init__(self, base, phase):
        self.base, self.phase, self.configs = base, phase, []

    def add(self, group, name, note, seeds=(0, 1), **override):
        for seed in seeds:
            cfg = merge(self.base, override)
            cfg.update(phase=self.phase, group=group, name=name, note=note, seed=seed)
            self.configs.append(cfg)


def phase2():
    t = Table(BASE, 2)
    three = (0, 1, 2)
    # H2.1: how should the 10 frames be combined?
    t.add("time", "early_fusion", "H2.1: frames stacked as input channels, so every kernel sees all 10 frames at once", three)
    t.add("time", "frame_gru", "H2.1: per-site GRU over the 10 frames (a recurrent model of time) before the ring backbone",
          three, view={"layout": "frames"}, model={"front": "gru"})
    t.add("time", "conv2d", "H2.1: one 2D convolution over (time, site) before the ring backbone",
          three, view={"layout": "frames"}, model={"front": "conv2d"})
    t.add("time", "single_frame", "H2.1: only the tau = 0 frame (E and theta(0)); do later frames add anything?",
          three, view={"frames": [0]})
    t.add("time", "pointwise_mlp", "H2.1 follow-up: a per-site MLP over the stacked frames (no recurrence) before the "
          "ring backbone; does it match the GRU at lower cost?", three, model={"front": "mlp"})
    # follow-up after the first results: a deep dilated stack that sees the whole ring was best;
    # combine it with the other winning ideas before fixing the architecture
    ring = {"spectral": False, "kernel": 3, "dilations": [1, 2, 4, 8, 16, 32, 1]}
    t.add("combine", "dilated129_plus_fourier", "H2 follow-up: whole-ring dilated stack plus the Fourier branch",
          three, model={**ring, "spectral": True})
    t.add("combine", "dilated129_mlp_front", "H2 follow-up: whole-ring dilated stack behind the per-site MLP front end",
          three, model={**ring, "front": "mlp"})
    t.add("combine", "dilated129_gru_front", "H2 follow-up: whole-ring dilated stack behind the per-site GRU front end",
          three, view={"layout": "frames"}, model={**ring, "front": "gru"})
    t.add("combine", "dilated129_w64", "H2 follow-up: whole-ring dilated stack, width 64", three, model={**ring, "width": 64})
    # H2.2: how much of the ring must a site see? (local-only stacks, kernel 3, growing dilations)
    for rf, dil in [(9, [1, 1, 1, 1]), (17, [1, 2, 4, 1]), (33, [1, 2, 4, 8, 1]), (65, [1, 2, 4, 8, 16, 1]),
                    (129, [1, 2, 4, 8, 16, 32, 1])]:
        t.add("receptive_field", f"local_rf{rf}", f"H2.2: local-only circular CNN seeing {rf} sites (kernel 3, dilations {dil})",
              (0, 1, 2) if rf == 129 else (0, 1), model={"spectral": False, "kernel": 3, "dilations": dil})
    # H2.3: global (Fourier) branch, local branch, or both; how many ring modes?
    t.add("scales", "global_only", "H2.3: Fourier branch only (ring modes 1-6), no local convolutions", model={"local": False})
    t.add("scales", "local_only", "H2.3: local branch only (2 layers, kernel 5)", model={"spectral": False})
    for modes in (3, 6, 12, 25):
        t.add("scales", f"both_modes{modes}", f"H2.3: both branches, Fourier branch keeps ring modes 1-{modes}", model={"modes": modes})
    # H2.4: kernel size and width
    for k in (3, 5, 9, 17):
        for w in (8, 16, 32, 64):
            t.add("size", f"k{k}_w{w}", f"H2.4: kernel {k}, width {w}, depth 2", (0,), model={"kernel": k, "width": w})
    for k, w in ((5, 16), (5, 32), (3, 32)):
        t.add("size", f"k{k}_w{w}_d4", f"H2.4: kernel {k}, width {w}, depth 4", (0,), model={"kernel": k, "width": w, "depth": 4})
    # H2.5: does respecting the ring's symmetry save data?
    for n in (1000, 3000, 10_000, 30_000, 60_000):
        for name, over, why in (("circular", {}, "circular padding, one score per site (shift-equivariant)"),
                                ("zero_pad", {"padding": "zeros"}, "zero padding at sites 0 and 99"),
                                ("dense_head", {"head": "dense"}, "flatten + dense head like v1 (not shift-equivariant)")):
            t.add("symmetry", f"{name}_n{n}", f"H2.5: {why}, {n} training runs", n_train=n, model=over)
    # H2.6: does feeding in the linear-physics prediction help, especially with little data?
    for n in (3000, 60_000):
        t.add("prior", f"floquet_channel_n{n}", f"H2.6: add the Floquet-propagated field (6 periods) as an input channel, {n} training runs",
              n_train=n, view={"floquet": 6})
    return t.configs


# ---- choices carried forward; each is set from the previous phase's results (see REPORT.md) ----
# Phase 2 winner: the whole-ring dilated CNN (local_rf129). It ties the best validation score (1.561 vs
# 1.570 for dilated + Fourier and 1.571 for the per-site GRU, all within one standard error) and is the
# cheapest and simplest of the three: 23k parameters, no Fourier layer, no recurrence.
ARCH = merge(BASE["model"], {"spectral": False, "kernel": 3, "dilations": [1, 2, 4, 8, 16, 32, 1]})
# Phase 3 winner: angles only at tau = 0..4.5 (I3). Best mean distance and best ±1 of the noise-free input
# sets (1.56 ± 0.01, 76.7%), and the cheapest to measure: one camera, no velocities, no model parameters.
VIEW = {"channels": ["theta"], "frames": list(range(10)), "theta0": False, "layout": "channels"}
# Phase 4 winner: soft labels (sigma 1) + 0.1 x expected distance + 0.1 x log-time head (L5). It ties the
# profile-head variant within one standard error (val 1.54 vs 1.52) and is simpler: its extra target comes free.
LOSS = {"kind": "ce", "soft_sigma": 1.0, "distance": 0.1, "time": 0.1}
# Phase 5: no optimizer beat AdamW + cosine beyond seed noise at the screening budget (Adam constant 1.68 ± 0.05,
# AdamW one-cycle 1.65 ± 0.08, AdamW cosine 1.75 ± 0.09; SGD failed to train), so the final model keeps the
# recipe used for every full-budget run in Phase 2.
OPTIM = BASE["optim"]


def fit_view(view, arch=None):
    """Give a view the layout the architecture needs: multi-frame front ends read (frames, C, N)."""
    arch = arch or ARCH
    view = dict(view)
    view["layout"] = "frames" if arch.get("front") in ("gru", "conv2d") else "channels"
    return view


def fit_arch(view, arch=None):
    """A 2D (time x site) front end needs at least 3 frames; use early fusion for shorter views."""
    arch = dict(arch or ARCH)
    if arch.get("front") == "conv2d" and len(view.get("frames", [0])) < 3:
        arch["front"] = "early"
    return arch


def sensor_view(kind, start, noise=0.0, stride=1):
    """Real-world sensing setups: S1 camera (angles, 8 frames = 4 time units), S2 camera + gyros
    (angles and angular velocities, one frame), S3 energies computed from S2's measurements."""
    channels, frames = {"S1": (["theta"], list(range(8))), "S2": (["theta", "omega"], [0]),
                        "S3": (["E"], [0])}[kind]
    return {"channels": channels, "frames": frames, "start": start, "noise": noise, "stride": stride,
            "theta0": False, "layout": "channels"}


class FittedTable(Table):
    """A Table whose views get the layout the current architecture needs."""

    def add(self, group, name, note, seeds=(0, 1), **override):
        view = merge(self.base["view"], override.get("view", {}))
        override = dict(override, view=fit_view(view), model=merge(fit_arch(view), override.get("model", {})))
        super().add(group, name, note, seeds, **override)


def phase3():
    t = FittedTable(merge(BASE, {"model": ARCH, "optim": {"steps": SCREEN_STEPS}}), 3)
    three = (0, 1, 2)
    # H1.1 / H1.6: which quantities carry the information (noise-free, observed from release)?
    sets = {
        "I0_v1_inputs": ("v1's inputs: E at tau = 0..4.5 plus theta(0)", V1_INFO),
        "I1_theta0": ("theta(0) alone, one frame, one channel", {"channels": ["theta"], "frames": [0], "theta0": False}),
        "I2_E0": ("E(0) alone", {"channels": ["E"], "frames": [0], "theta0": False}),
        "I3_theta_frames": ("theta at tau = 0..4.5", {"channels": ["theta"], "frames": list(range(10)), "theta0": False}),
        "I4_E_frames": ("E at tau = 0..4.5", {"channels": ["E"], "frames": list(range(10)), "theta0": False}),
        "I5_theta_omega_random_start": ("theta and omega at one random time within the first swing",
                                        {"channels": ["theta", "omega"], "frames": [0], "random_phase": True, "theta0": False}),
        "I6_dtheta0_amplitude": ("theta(0) minus the chain mean, plus the amplitude as a separate channel",
                                 {"channels": ["dtheta"], "frames": [0], "amplitude": True, "theta0": False}),
    }
    for name, (why, view) in sets.items():
        t.add("information", name, f"H1.1/H1.6: {why}", three, view=dict(view, layout="channels"))
    # H1.2: observation starting at an unknown point of the swing
    for w in (1, 2, 4, 8, 16):
        t.add("phase", f"theta_random_start_w{w}", f"H1.2: angles only, {w} frame(s) ({w * 0.5:g} time units) from a random start within the first swing",
              view={"channels": ["theta"], "frames": list(range(w)), "random_phase": True, "theta0": False})
    t.add("phase", "E_random_start_w1", "H1.2: energy, one frame at a random start within the first swing",
          view={"channels": ["E"], "frames": [0], "random_phase": True, "theta0": False})
    t.add("phase", "theta_omega_random_start_w1", "H1.2: angles and velocities, one frame at a random start",
          view={"channels": ["theta", "omega"], "frames": [0], "random_phase": True, "theta0": False})
    # H1.3 / H1.4: sensor noise against how long we watch
    for kind in ("S1", "S2", "S3"):
        for start in (0, 10, 20, 30):
            for noise in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                t.add("noise_time", f"{kind}_t{start}_noise{noise:g}",
                      f"H1.3/H1.4: sensing setup {kind} starting at tau = {start}, sensor noise {noise:g}",
                      (0,), view=sensor_view(kind, start, noise))
    # H1.5: only every s-th pendulum instrumented (camera angles)
    for kind, start in (("S1", 0), ("S1", 20)):
        for noise in (0.0, 1e-3):
            for stride in (1, 2, 4, 8):
                t.add("sparse", f"{kind}_t{start}_noise{noise:g}_every{stride}",
                      f"H1.5: angles from every {stride}th pendulum, start tau = {start}, noise {noise:g}",
                      (0,), view=sensor_view(kind, start, noise, stride))
    return t.configs


def phase4():
    # one factor at a time around a fixed base (v1's inputs), so Phases 3-5 can run in parallel;
    # Phase 6 combines the winners and checks for interactions with a leave-one-out pass
    t = Table(merge(BASE, {"model": ARCH, "view": V1_INFO, "optim": {"steps": SCREEN_STEPS}}), 4)
    three = (0, 1, 2)
    t.add("loss", "L1_ce", "H3: baseline cross-entropy on the site (v1's site loss)", three, loss={"kind": "ce"})
    t.add("loss", "L2_angle", "H3.1: circular regression (1 - cos of the angle error); should average competing humps",
          three, loss={"kind": "angle"})
    for sigma in (0.5, 1.0, 2.0):
        t.add("loss", f"L3_soft{sigma:g}", f"H3.2: circular-Gaussian soft labels, sigma {sigma:g} site(s)",
              three, loss={"kind": "ce", "soft_sigma": sigma})
    for lam in (0.03, 0.1, 0.3):
        t.add("loss", f"L4_soft1_dist{lam:g}", f"H3.3: soft labels (sigma 1) + {lam:g} x expected ring distance",
              three, loss={"kind": "ce", "soft_sigma": 1.0, "distance": lam})
    t.add("loss", "L5_soft1_dist0.1_time", "H3.4: + 0.1 x log-localization-time head",
          three, loss={"kind": "ce", "soft_sigma": 1.0, "distance": 0.1, "time": 0.1})
    t.add("loss", "L6_soft1_dist0.1_profile", "H3.4: + 0.1 x energy-profile head (period-averaged energy at tau = 30)",
          three, loss={"kind": "ce", "soft_sigma": 1.0, "distance": 0.1, "profile": 0.1})
    t.add("loss", "L7_soft1_dist0.1_localizes", "H3.5: + v1's binary 'does it localize' head (weight 0.5)",
          three, loss={"kind": "ce", "soft_sigma": 1.0, "distance": 0.1, "localizes": 0.5})
    return t.configs


def phase5(lrs=None):
    """Optimizer ablations. Learning rates for O3-O5 come from scripts/lr_range_test.py."""
    lrs = lrs or LR_FROM_RANGE_TEST
    t = Table(merge(BASE, {"model": ARCH, "view": V1_INFO, "loss": {"kind": "ce"}}), 5)
    three = (0, 1, 2)
    adamw = {"name": "adamw", "lr": 3e-3, "weight_decay": 1e-4, "schedule": "cosine", "warmup_fraction": 0.03,
             "steps": SCREEN_STEPS, "batch_size": 256}
    t.add("optimizer", "O1_adam_constant", "H4.1: v1's optimizer: Adam, constant learning rate 1e-3, no weight decay",
          three, optim={**adamw, "name": "adam", "lr": 1e-3, "schedule": "constant", "warmup_fraction": 0.0})
    t.add("optimizer", "O2_adamw_cosine", "H4.1: AdamW + cosine decay with 3% warmup (the default so far)", three, optim=adamw)
    t.add("optimizer", "O3_adamw_onecycle", "H4.1: AdamW + one-cycle schedule", three,
          optim={**adamw, "schedule": "onecycle", "lr": lrs["adamw_max"], "warmup_fraction": 0.0})
    t.add("optimizer", "O4_sgd_nesterov_cosine", "H4.1: SGD with Nesterov momentum + cosine (gradients clipped at 1;"
          " the range test was inconclusive for SGD)", three, optim={**adamw, "name": "sgd", "lr": lrs["sgd"], "clip": 1.0})
    t.add("optimizer", "O5_lion_cosine", "H4.1: Lion + cosine", three,
          optim={**adamw, "name": "lion", "lr": lrs["lion"], "weight_decay": 1e-2})
    for batch in (64, 1024):  # same number of epochs; learning rate scaled with the square root of the batch size
        t.add("batch", f"adamw_batch{batch}", f"H4.2: batch size {batch} (lr scaled by sqrt(batch/256), same epochs)",
              three, optim={**adamw, "batch_size": batch, "lr": 3e-3 * (batch / 256) ** 0.5, "steps": SCREEN_STEPS * 256 // batch})
    for wd in (0.0, 1e-2):
        t.add("weight_decay", f"adamw_wd{wd:g}", f"H4: AdamW weight decay {wd:g}", three, optim={**adamw, "weight_decay": wd})
    return t.configs


# from scripts/lr_range_test.py (results/lr_range.json): loss minimum at 6.8e-3 for AdamW and 7.9e-3 for Lion;
# peak = minimum / 3 for one-cycle, start = minimum / 10 for Lion with cosine. SGD never improved before
# diverging in the range test, so it gets a conventional 0.03 with gradient clipping.
LR_FROM_RANGE_TEST = {"adamw_max": 2.3e-3, "sgd": 0.03, "lion": 7.9e-4}

PLAIN = {"model": BASE["model"], "view": V1_INFO, "loss": {"kind": "ce"}, "optim": BASE["optim"]}


def final_recipe():
    """The assembled v2 model: the winners of Phases 2-5, back at the full 30-epoch budget."""
    full = dict(OPTIM, steps=STEPS * 256 // OPTIM.get("batch_size", 256))
    return merge(BASE, {"model": ARCH, "view": VIEW, "loss": LOSS, "optim": full})


def phase6():
    t = Table(final_recipe(), 6)
    three = (0, 1, 2)
    t.add("final", "v2_final", "Q5: the assembled v2 model (winners of Phases 2-5)", (0, 1, 2, 3, 4))
    # leave one choice out at a time (only choices the final model actually makes)
    removals = [
        ("no_fourier_branch", "without the global Fourier branch", {"model": {"spectral": False}}, ARCH.get("spectral", True)),
        ("no_dilation", "the same layers without dilation (sees 17 sites instead of the whole ring)",
         {"model": {"dilations": [1] * len(ARCH.get("dilations") or [])}}, bool(ARCH.get("dilations"))),
        ("zero_padding", "zero padding instead of circular", {"model": {"padding": "zeros"}}, ARCH.get("padding") == "circular"),
        ("dense_head", "flatten + dense head instead of one score per site", {"model": {"head": "dense"}}, ARCH.get("head") == "site"),
        ("plain_cross_entropy", "plain cross-entropy instead of the chosen loss",
         {"loss": {"kind": "ce", "soft_sigma": 0.0, "distance": 0.0, "time": 0.0, "profile": 0.0}}, LOSS != {"kind": "ce"}),
        ("no_soft_labels", "hard labels instead of soft (other loss terms kept)", {"loss": {"soft_sigma": 0.0}},
         bool(LOSS.get("soft_sigma"))),
        ("no_time_head", "without the auxiliary time head", {"loss": {"time": 0.0}}, bool(LOSS.get("time"))),
        ("v1_inputs", "v1's inputs (energy frames + theta(0)) instead of the chosen inputs", {"view": V1_INFO},
         VIEW != V1_INFO),
        ("no_floquet_channel", "without the Floquet input channel", {"view": {"floquet": 0}}, bool(VIEW.get("floquet"))),
        ("adam_constant", "v1's optimizer (Adam, constant 1e-3)", {"optim": {"name": "adam", "lr": 1e-3, "schedule": "constant",
                                                                             "warmup_fraction": 0.0}}, OPTIM.get("name") != "adam"),
    ]
    for name, why, override, applies in removals:
        if applies:
            t.add("leave_one_out", name, f"Q5 leave-one-out: final model {why}", three, **override)
    for n in (1000, 3000, 10_000, 30_000, 100_000):
        t.add("data", f"final_n{n}", f"Q5: final model trained on {n} runs (data efficiency)", (0, 1), n_train=n)
    t.add("noise", "final_trained_noise1e-3", "Q5: final model trained and tested with 1e-3 rad sensor noise",
          three, view={"noise": 1e-3})
    impurity_pools = {"train": "robust_impurity_train", "val": "robust_impurity_val", "test": "robust_impurity_test"}
    t.add("impurity", "impurity_trained", "Q5: trained on chains with a random impurity (0-2%), no map of the chain",
          three, pools=impurity_pools)
    t.add("impurity", "impurity_trained_with_map", "Q5: trained on chains with a random impurity, given a measured map of each pendulum",
          three, pools=impurity_pools, view={"calibration": True})
    add_deploy(t)
    add_label(t)
    add_damping(t)
    return t.configs


def add_label(t):
    """Which pendulum crosses 10% first depends on the phase of the swing at the crossing; the
    hotspot's period-averaged position (the "avg" label) does not. Is that target easier to predict?"""
    t.add("label", "final_avg_label", "Q5: final recipe trained and tested on the period-averaged, symmetric label",
          (0, 1, 2), label="avg")
    t.add("label", "final_sym_label", "Q5: final recipe trained and tested on the symmetric label (springs split "
          "between both pendulums, not period-averaged)", (0, 1, 2), label="sym")


def add_deploy(t):
    """The final recipe on the sensing setups Phase 3 recommends for a real chain."""
    three = (0, 1, 2)
    t.add("deploy", "deploy_camera_every2_noise1e-3", "Q1/Q5: final recipe, camera on every 2nd pendulum with 1e-3 rad noise, from release",
          three, view={"noise": 1e-3, "stride": 2})
    t.add("deploy", "deploy_camera_noise1e-2_t20", "Q1/Q5: final recipe, a coarse camera (1e-2 rad) watching from tau = 20",
          three, view={"noise": 1e-2, "start": 20})
    t.add("deploy", "deploy_energy_t30", "Q1/Q5: final recipe, energies at tau = 30 (one frame, noise-free): the late-warning ceiling",
          three, view={"channels": ["E"], "frames": [0], "start": 30})


def add_damping(t):
    """Damped chains, gamma log-uniform in [1e-4, 1e-2] per time unit (quality factor 100 to 10,000).
    The swing decays toward A_c, so some runs never localize by tau = 150 (site -1); they are left out of
    the site and time losses. Can the final recipe still warn early, and tell which runs will localize?"""
    three = (0, 1, 2)
    pools = {"train": "robust_damped_train", "val": "robust_damped_val", "test": "robust_damped_test"}
    damped = {"pools": pools, "ignore_unlocalized": True}
    t.add("damping", "damped_trained", "Q5/damping: final recipe trained and tested on damped chains (gamma log-uniform "
          "in 1e-4..1e-2); runs that never localize are left out of the site and time losses", three, **damped)
    t.add("damping", "damped_trained_localizes_head", "Q5/damping: as damped_trained, plus v1's 'does it localize' head "
          "(weight 0.5), which damping makes non-trivial", three, loss={"localizes": 0.5}, **damped)
    t.add("damping", "damped_trained_avg_label", "Q5/damping: as damped_trained_localizes_head, on the swing-averaged "
          "label, which a shift in the localization time moves less", three, loss={"localizes": 0.5}, label="avg", **damped)
    t.add("damping", "damped_deploy_camera_every2_noise1e-3", "Q1/Q5/damping: the recommended camera (every 2nd "
          "pendulum, 1e-3 rad, from release) on damped chains, with the localizes head", three,
          loss={"localizes": 0.5}, view={"noise": 1e-3, "stride": 2}, **damped)


def phase6_deploy():
    """Only the deploy group of Phase 6, so it can run beside the main Phase 6 sweep."""
    return [c for c in phase6() if c["group"] == "deploy"]


def phase6_label():
    """Only the label group of Phase 6, so it can run beside the main Phase 6 sweep."""
    return [c for c in phase6() if c["group"] == "label"]


def phase6_damping():
    """Only the damping group of Phase 6 (added after the main Phase 6 sweep)."""
    return [c for c in phase6() if c["group"] == "damping"]


class BothLabels(Table):
    """A Table that adds every config twice: on the swing-averaged label, and on v1's label as the matched
    control. The control has no label key, so a v1-label run Phase 6 already made has the same config id
    and is reused, not repeated."""

    def add(self, group, name, note, seeds=(0, 1), **override):
        super().add(group, f"{name}_avg", f"{note}; trained and tested on the swing-averaged label", seeds,
                    label="avg", **override)
        super().add(group, f"{name}_v1label", f"{note}; v1's label (matched control)", seeds, **override)


def phase7():
    """Phases 2-6 chose inputs, architecture and loss on v1's label, which jitters by 1-3 sites within a swing
    (Phase 6). Do those choices still hold on the swing-averaged label? Final recipe, full budget, 3 seeds."""
    t = BothLabels(final_recipe(), 7)
    three = (0, 1, 2)
    # Q1: in clean data every input set carries the same initial state, and on v1's label the inputs cost
    # little (v1's inputs +0.02). With the jitter gone the error floor drops from 1.49 to 0.58, so smaller
    # differences between input sets can show.
    t.add("inputs", "v1_inputs", "Phase 7 / Q1: final recipe on v1's inputs (energy frames + theta(0))", three,
          view=V1_INFO)
    t.add("inputs", "angle_frames", "Phase 7 / Q1: final recipe (angles at tau = 0..4.5)", three)
    t.add("inputs", "theta0_only", "Phase 7 / Q1: final recipe on theta(0) alone, one frame", three,
          view={"channels": ["theta"], "frames": [0], "theta0": False})
    # H2.2: the growing modes are the ring's five longest waves whichever label marks the hotspot, so a site
    # should still need to see the whole ring. Same dilation patterns as Phase 2's receptive_field group.
    for rf, dil in ((17, [1, 2, 4, 1]), (33, [1, 2, 4, 8, 1]), (129, ARCH["dilations"])):
        t.add("receptive_field", f"rf{rf}", f"Phase 7 / H2.2: final recipe, circular stack seeing {rf} sites "
              f"(kernel 3, dilations {dil})", three, model={"dilations": dil})
    # H3.2: on v1's label soft labels raised +-1 from 65% to 77%, by placing predictions between the sites the
    # label jitters between. Without the jitter that gain should shrink.
    t.add("soft_labels", "soft_on", "Phase 7 / H3.2: final recipe (soft labels, sigma 1)", three)
    t.add("soft_labels", "soft_off", "Phase 7 / H3.2: final recipe with hard labels (other loss terms kept)", three,
          loss={"soft_sigma": 0.0})
    add_deploy(t)
    return t.configs


FINE_EVAL = 47  # checkpoint every 47 steps instead of every 235 (= 5 x 47, so the default checkpoints are kept)
BEST_STEPS = {1000: 235, 3000: 235, 10_000: 705}  # Phase 8: best schedule on validation without mirror images


def phase8():
    """Regularization with little data. In Phase 6's data group (final recipe, 7,050 steps at every size), the
    best checkpoint at 1,000 and 3,000 training runs is the first one evaluated (step 235), and the validation
    error then rises by about 50%. Does regularization rescue v2 there? Final recipe, same 7,050 steps."""
    t = Table(final_recipe(), 8)
    fine = {"eval_every": FINE_EVAL}
    for n in (1000, 3000, 10_000):
        g = f"n{n}"
        # the matched control is Phase 6's final_n{n} (same config id, so it is reused, not rerun)
        t.add(g, f"n{n}_control", f"Phase 8: final recipe on {n} runs, checkpoint every 235 steps (Phase 6's final_n{n})",
              (0, 1), n_train=n)
        t.add(g, f"n{n}_fine_eval", f"Phase 8: as the control, checkpoint every {FINE_EVAL} steps: how much of the "
              "small-data loss is a late first checkpoint?", (0, 1, 2), n_train=n, optim=fine)
        # each regularizer below also checkpoints every 47 steps, so it is compared with n{n}_fine_eval
        for wd in (1e-2, 1e-1):
            t.add(g, f"n{n}_wd{wd:g}", f"Phase 8: AdamW weight decay {wd:g} (final recipe: 1e-4)", (0, 1),
                  n_train=n, optim=dict(fine, weight_decay=wd))
        for p in (0.1, 0.2):
            t.add(g, f"n{n}_dropout{p:g}", f"Phase 8: dropout {p:g} after every hidden activation", (0, 1),
                  n_train=n, model={"dropout": p}, optim=fine)
        t.add(g, f"n{n}_mirror", "Phase 8: add the mirror image of every training run (the ring's reflection symmetry; "
              "v1's one-sided label becomes the mirrored left-bond label)", (0, 1, 2), n_train=n, augment="mirror", optim=fine)
        # follow-ups after the first results at 1,000 runs (REPORT.md, Sec. 10). There the best checkpoint comes
        # at steps 141-188, still in the learning-rate warmup (211 steps). By then AdamW, which shrinks the weights by
        # lr x wd per step, has shrunk them by only 1.4-2.5% at wd 0.1, so 1e-2 and 1e-1 change nothing. Does wd 1 help?
        t.add(g, f"n{n}_wd1", "Phase 8 follow-up: AdamW weight decay 1", (0, 1, 2), n_train=n,
              optim=dict(fine, weight_decay=1.0))
        # The cosine decay starts long after the model has overfit. Does a shorter schedule, which anneals the learning
        # rate before that, do better? (Same 3% warmup; checkpoints every steps / 30 as by default.)
        for steps in (235, 705, 2350):
            t.add(g, f"n{n}_steps{steps}", f"Phase 8 follow-up: a {steps}-step cosine schedule instead of 7,050",
                  (0, 1, 2), n_train=n, optim={"steps": steps})
        # The best combination, from the results above (validation pool). The mirror images help at every size. Weight
        # decay 1 helps at 10,000 runs, where training runs long enough for the decay to act, and is neutral below. A
        # shorter schedule helps at 3,000 and 10,000 runs. Weight decay needs the long schedule, and at 10,000 runs the
        # images' gain comes late (best checkpoints at steps 4,089-4,700), so the two strategies are tried separately.
        t.add(g, f"n{n}_mirror_wd1", "Phase 8 combination: mirror images + weight decay 1 (7,050 steps)", (0, 1, 2),
              n_train=n, augment="mirror", optim=dict(fine, weight_decay=1.0))
        steps = BEST_STEPS[n]
        t.add(g, f"n{n}_mirror_steps{steps}", f"Phase 8 combination: mirror images + the {steps}-step schedule (the "
              "shortest within 0.01 of the best on validation without images)", (0, 1, 2), n_train=n, augment="mirror",
              optim={"steps": steps})
    # follow-up after the results at 1k-10k runs: the mirror images are worth about a doubling of the training runs at
    # every size. Do they help the final model at 60,000 runs too? Checkpoints as in v2_final, so the images are the
    # only change (its best checkpoints come late, at steps 4,465-7,050).
    t.add("n60000", "n60000_control", "Phase 8: the final model (Phase 6's v2_final)", (0, 1, 2))
    t.add("n60000", "n60000_mirror", "Phase 8 follow-up: the final model with the mirror image of every training run",
          (0, 1, 2), augment="mirror")
    return t.configs


def phase3_avg():
    """Phase 3's noise-and-timing grid again, on the swing-averaged label. On v1's label waiting also helps without
    noise, because that label fires at a moment in the swing; on the averaged label it gains only 0.03 (Phase 7). So
    this grid measures the Floquet trade-off without the label's own dependence on time. Each config is a Phase 3
    noise_time config plus label="avg", with every other setting unchanged (Phase 3 base, screening budget, seed 0).
    Its control is the Phase 3 run itself: listed here without the label key, it has the same config id and is
    reused, not rerun."""
    # Predictions, written before the runs started:
    # - Camera (S1): the noise threshold is set by when the growing modes rise above the sensor noise, which does not
    #   depend on the label. So the error should again depend only on sigma_s * exp(-sigma_1 * tau), and a 10x
    #   noisier sensor should cost about ln 10 / sigma_1 = 19.5 time units of waiting. With the label's own time
    #   dependence gone, the noise-free cells should be flat in tau, so the collapse should reach down to them.
    # - Noise should cost a larger share of the error than on v1's label (at full budget x2.0-2.2 against x1.3-1.4,
    #   Phase 7): its misses land on the wrong hump, which no label removes, while the floor drops from ~1.5 to ~0.6.
    # - S2 and S3 (one frame) should still degrade faster from release (no frames to average the noise) and gain less
    #   from waiting (velocity noise enters at first order once the chain moves, and a fixed later frame lands at an
    #   amplitude-dependent point of the swing). Neither cause involves the label.
    # Refined prediction, added after re-fitting Phase 3's cells (scripts/noise_tradeoff.py) and before any tau > 0
    # cell of this grid finished. On v1's label a fit of all 24 camera cells gives k = 0.087 (26.5 time units per 10x
    # noise, 95% CI 23-30), not 0.118. Split by the test runs' amplitude, k follows the mean growth rate of the
    # growing modes, sigma_bar(A), rather than the fastest one: k = 0.070, 0.077, 0.111, 0.115 for A in 1.65-1.70,
    # 1.70-1.75, 1.75-1.80, 1.80-1.85, against sigma_bar = 0.070, 0.085, 0.101, 0.116 and sigma_1 = 0.087, 0.109,
    # 0.128, 0.145. White sensor noise is the same in every ring mode, and the site depends on all the growing modes.
    # So on the averaged label: k per amplitude bin ~ sigma_bar(A), and about ln 10 / 0.093 = 25 time units per 10x
    # noise for the whole test pool.
    configs = []
    for cfg in phase3():
        if cfg["group"] == "noise_time":
            configs.append(merge(cfg, {"name": f"{cfg['name']}_avg", "label": "avg",
                                       "note": f"Phase 3 on the swing-averaged label: {cfg['note']}"}))
            configs.append(merge(cfg, {"name": f"{cfg['name']}_v1label",
                                       "note": f"{cfg['note']}; v1's label (matched control, Phase 3's run)"}))
    return configs


def deploy_start():
    """When should a coarse camera start watching? The Phase 3 addendum found that waiting divides the sensor noise at
    about the mean growth rate of the growing modes (k = 0.09-0.10), not sigma_1, and changed the report's rule from
    8.5 ln(sigma_s / 1e-3) to 10 ln(sigma_s / 1e-3) time units after release: tau ~ 23 for a 1e-2 rad camera, not 20.
    Full-budget check: the final recipe with a 1e-2 rad camera on every pendulum from tau = 20 (Phase 6's setup), 24 and
    28, against the camera the rule aims to match (1e-3 rad on every pendulum, from release). Both labels, 3 seeds.
    Configs that Phase 6 or 7 already trained have the same config id and are reused, not rerun."""
    t = BothLabels(final_recipe(), 6)
    three = (0, 1, 2)
    # Predictions, written before the runs started. In the screening grid a 1e-2 rad camera matched the 1e-3 rad one from
    # release when it started at tau ~ 21.3 (v1's label) and 23.5 (averaged label). So at the full budget:
    # - From tau = 24 the coarse camera should be level with the 1e-3 rad camera from release, within the seed spread, on
    #   both labels. From tau = 20 it should be worse, clearly so on the averaged label; from tau = 28, better.
    # - On the averaged label the 1e-3 rad camera from release should land at about 1.0 (every 2nd pendulum: 1.18).
    t.add("deploy_start", "final_recipe", "Q1/Q5 reference: the final recipe, every pendulum, noise-free, from release", three)
    t.add("deploy_start", "camera_noise1e-3", "Q1/Q5: final recipe, camera on every pendulum with 1e-3 rad noise, from "
          "release: the camera the start-time rule aims to match", three, view={"noise": 1e-3})
    t.add("deploy_start", "camera_every2_noise1e-3", "Q1/Q5: final recipe, camera on every 2nd pendulum with 1e-3 rad "
          "noise, from release (the recommended setup)", three, view={"noise": 1e-3, "stride": 2})
    for start in (20, 24, 28):
        t.add("deploy_start", f"camera_noise1e-2_t{start}", f"Q1/Q5: final recipe, a coarse camera (1e-2 rad) on every "
              f"pendulum watching from tau = {start}", three, view={"noise": 1e-2, "start": start})
    return t.configs


def phase9():
    """Is one pendulum the right target? When v1's label fires, the hotspot is a hump about 8 pendulums wide, and the
    label jitters by 1-3 sites within a swing. The swing-averaged label removes the jitter but fires later (median tau
    60 against 55), once one pendulum's averaged energy passes 10%. The hump label marks the event at the hump's scale:
    the first sample where the swing-averaged energy of 3 neighbouring pendulums passes 15% of the total, which fires at
    about v1's median time (tau 54.5 against 55); its site is the window's centre (sim.localize). Final recipe, full
    budget, 3 seeds."""
    t = Table(final_recipe(), 9)
    # Predictions, written before the runs started:
    # - The hump label has no within-swing jitter (it is averaged over a swing like the "avg" label), so it should be
    #   about as predictable as that label (0.58, 96.4% within +-1) and far more than v1's (1.49, 77.2%).
    # - Summing 3 pendulums finds the hump's centre even when its top is flat, which should help a little. Firing 5
    #   time units earlier, when rival humps are closer in size, should hurt a little. Expect 0.5-0.7 and 95-97%.
    # - The two labels agree within +-1 in 93% of runs (validation pool), so this is a different target, not a copy.
    t.add("label", "final_hump_label", "Phase 9: final recipe trained and tested on the hump label: the first sample where "
          "the swing-averaged energy of 3 neighbouring pendulums passes 15% of the total; the label is the window's centre",
          (0, 1, 2), label="hump")
    # the matched controls are Phase 6's runs on the other two labels (same config ids, so they are reused, not rerun)
    t.add("label", "final_avg_label", "Phase 9 control: Phase 6's final_avg_label (the swing-averaged label)", (0, 1, 2),
          label="avg")
    t.add("label", "final_v1_label", "Phase 9 control: Phase 6's v2_final (v1's label)", (0, 1, 2, 3, 4))
    return t.configs


class ThreeLabels(Table):
    """A Table that adds every config on the hump label, beside the same config on the swing-averaged label and on v1's
    label as matched controls. The controls are Phase 7's runs (same config ids), so they are reused, not repeated."""

    def add(self, group, name, note, seeds=(0, 1), **override):
        super().add(group, f"{name}_hump", f"{note}; trained and tested on the hump label", seeds, label="hump",
                    **override)
        super().add(group, f"{name}_avg", f"{note}; the swing-averaged label (Phase 7's run)", seeds, label="avg",
                    **override)
        super().add(group, f"{name}_v1label", f"{note}; v1's label (matched control)", seeds, **override)


def phase10():
    """Phase 9's hump label is the best target so far (final recipe: 0.27, 99.4% within +-1). Is its gain the moment
    it is read or the 3-pendulum sum? And do the choices made on v1's label, which Phase 7 re-tested on the swing-
    averaged label, still hold on it? Final recipe, full budget, 3 seeds; Phase 7's runs are the controls."""
    three = (0, 1, 2)
    # Predictions, written before the runs started:
    # - Timing: hump_top fires with the hump label but names the top pendulum. At that moment the two agree within +-1
    #   in 99.7% of test runs, and the hump model already scores 0.41 and 98.7% on hump_top without being trained on
    #   it. So the timing should explain the +-1 gain: about 0.3-0.4 and 99%. The top pendulum flips between two
    #   neighbours when the hump's top is flat, so fewer exact hits than on the hump label (75-80% against 84.7%).
    # - Inputs and receptive field: what is hard is reading the initial state, whichever label marks the hump. So
    #   about the same absolute costs as on the averaged label (v1's inputs +0.03-0.06, theta(0) alone +0.08-0.15,
    #   33 sites +0.1-0.15), a larger share of the lower error. 17 sites cannot see the five longest waves at all:
    #   1.5-1.8, five to six times the final recipe.
    # - Soft labels: the hump label has no jitter, so they should not matter (within 0.02, as on the averaged label).
    # - Sensing: noise picks the wrong hump, and no label removes those misses, so the noisy setups lose about as many
    #   runs as on the averaged label: about 0.9-1.1 for both cameras, three to four times the final recipe. Energies
    #   at tau = 30 should tie with the final recipe (0.25-0.3). A second site should help the noisy setups more than
    #   the clean ones, since their misses are on the wrong hump.
    timing = Table(final_recipe(), 10)
    timing.add("timing", "final_hump_top_label", "Phase 10: final recipe on the hump_top label, which fires with the "
               "hump label but names the top pendulum of the swing-averaged energy (no 3-pendulum sum)", three,
               label="hump_top")
    timing.add("timing", "final_hump_label", "Phase 10 control: Phase 9's final_hump_label", three, label="hump")
    timing.add("timing", "final_avg_label", "Phase 10 control: Phase 6's final_avg_label", three, label="avg")
    t = ThreeLabels(final_recipe(), 10)
    t.add("inputs", "v1_inputs", "Phase 10 / Q1: final recipe on v1's inputs (energy frames + theta(0))", three,
          view=V1_INFO)
    t.add("inputs", "angle_frames", "Phase 10 / Q1: final recipe (angles at tau = 0..4.5)", three)
    t.add("inputs", "theta0_only", "Phase 10 / Q1: final recipe on theta(0) alone, one frame", three,
          view={"channels": ["theta"], "frames": [0], "theta0": False})
    for rf, dil in ((17, [1, 2, 4, 1]), (33, [1, 2, 4, 8, 1]), (129, ARCH["dilations"])):
        t.add("receptive_field", f"rf{rf}", f"Phase 10 / H2.2: final recipe, circular stack seeing {rf} sites "
              f"(kernel 3, dilations {dil})", three, model={"dilations": dil})
    t.add("soft_labels", "soft_on", "Phase 10 / H3.2: final recipe (soft labels, sigma 1)", three)
    t.add("soft_labels", "soft_off", "Phase 10 / H3.2: final recipe with hard labels (other loss terms kept)", three,
          loss={"soft_sigma": 0.0})
    add_deploy(t)
    return timing.configs + t.configs


THRESHOLD_POOLS = {s: f"thrpool_full_{s}" for s in ("train", "val", "test")}


def threshold():
    """Which localization threshold? v1 set 10% of the total energy "somewhat arbitrarily", and its label jitters
    because by then energy is already passing to the neighbours. The final recipe on first-crossing labels at 3-15%
    of the total, at absolute multiples of the separatrix energy (E_sep = 2, the on-site energy of a pendulum at the
    top; sep2 and sep3 bracket the 4-5% suggested by the label checks) and at the first pendulum past the top, all
    from scripts/threshold_labels.py (data/thrpool_*; the shared pools are untouched). Runs that never cross carry
    no site loss. The v1_settings group repeats v1's apparent published settings (4%, offsets +-0.0125, labels cut
    at t_f = 100) for the v2 cell beside the rebuilt v1's 92.5%."""
    t = Table(merge(final_recipe(), {"pools": THRESHOLD_POOLS, "ignore_unlocalized": True}), "threshold")
    three = (0, 1, 2)
    for s in (3, 4, 5, 6, 7, 8, 9, 10, 12, 15):
        t.add("share", f"thr{s:02d}", f"threshold study: first site past {s}% of the total energy (spring to the right)",
              three, label=f"thr{s:02d}")
    t.add("physical", "ott", "threshold study: the first pendulum past the top (|theta| > pi); no threshold", three, label="ott")
    for k in (2, 3):
        t.add("physical", f"sep{k}", f"threshold study: first site energy above {k} E_sep (absolute, not a share)",
              three, label=f"sep{k}")
    half = {s: f"thrpool_half_{s}" for s in ("train", "val", "test")}
    t.add("v1_settings", "thr04_half_tf100", "v1's apparent published settings (4%, offsets +-0.0125, labels cut at "
          "t_f = 100): v2 beside the rebuilt v1's 92.5%", three, label="thr04_tf100", pools=half)
    return t.configs


def setpools(tag):
    return {s: f"setpool_{tag}_{s}" for s in ("train", "val", "test")}


def settings_hump():
    """Where the settings come from, part 1: the hump label's own knobs. The hump label (Phase 9) sums the
    swing-averaged energy of 3 neighbouring pendulums and fires when they hold 15% of the total; the 15% was chosen so
    that it fires at v1's median time, so the recommended target inherited v1's arbitrary 10%. The threshold study
    showed that the labels which are learnable all fire when the first pendulum swings over the top, a physical event
    with no threshold. This phase asks whether the hump label's knobs matter, and whether the event can time the label
    instead of a share. All labels from scripts/setting_labels.py on the main pools' runs (data/setpool_main_*).
      hump_k{K}_s{S}: the Phase 9 definition with K = 3, 5, 7 pendulums and S = 10, 15, 20% (hump_k3_s15 is Phase 9's)
      eh_k{K}: timed by the first pendulum over the top; the site is the centre of the K pendulums with the most energy,
               averaged over the swing centred on that moment (8 time units, one swing); K = 1 is the top pendulum
    Final recipe, full budget, 3 seeds per label."""
    t = Table(merge(final_recipe(), {"pools": setpools("main"), "ignore_unlocalized": True}), "settings_hump")
    three = (0, 1, 2)
    # Predictions and the decision rule, written before the runs started:
    # - The grid should be flat: K and S move when the label fires (S) and how wide a window locates the hump (K), but
    #   every variant reads the same hump. Expect within +-1 to stay within 0.5 points of 99.4% across the grid.
    #   Larger S fires later, when rival humps have had longer to compete, so if anything accuracy falls with S.
    # - eh_k3 should match hump_k3_s15 (99.4%): it reads the same hump about when it forms (the hump label's window
    #   ends a median 5 time units after the event, so its centre lies about 1 time unit after it).
    # - eh_k1 (the top pendulum of the averaged energy) should be worse in exact hits, as hump_top was in Phase 10.
    # - Decision (on the validation pool, seed means): if eh_k3 is within 0.3 points within +-1 and 0.03 in mean
    #   distance of hump_k3_s15, the event-timed label becomes the recommended target, because its only remaining knob
    #   is the window K, which the grid covers; otherwise the Phase 9 label stays, justified or qualified by the grid.
    for k in (3, 5, 7):
        for s in (10, 15, 20):
            t.add("threshold_timed", f"hump_k{k}_s{s}", f"settings: hump label with {k} pendulums at {s}% of the total "
                  "energy (Phase 9 uses 3 and 15%)", three, label=f"hump_k{k}_s{s}")
    for k in (1, 3, 5, 7):
        t.add("event_timed", f"eh_k{k}", f"settings: hump label timed by the first pendulum over the top; site = centre of "
              f"the {k} pendulums with the most swing-averaged energy around that moment", three, label=f"eh_k{k}")
    return t.configs


# The target of the settings studies below, fixed by the decision rule of settings_hump before they ran. Result
# (validation, 3 seeds): eh_k3 99.4% within +-1 and 0.28 against hump_k3_s15 99.4% and 0.28 (+0.03 points, +0.001),
# so the event-timed label is the target. The threshold-timed grid was flat (99.4-99.6%) except 3 pendulums at 20%
# (98.0%), which fires 7 time units after the event; wider windows (5, 7) score better on their own labels but locate
# the 3-pendulum centre and the first pendulum over the top worse (cross-scoring), so they are easier, not better.
SETTINGS_LABEL = "eh_k3"
# Every pendulum must see the whole ring (Phase 2). For rings up to 400 pendulums the dilations go up to 128
# (receptive field 513 pendulums), at every N, including 100, so that only the ring changes.
WIDE_ARCH = merge(ARCH, {"dilations": [1, 2, 4, 8, 16, 32, 64, 128, 1]})
AMPLITUDES = (1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95)
OFFSET_SCALES = (0.125, 0.25, 0.5, 2, 4)


def _settings_base(tag):
    assert SETTINGS_LABEL, "set SETTINGS_LABEL from the settings_hump result first"
    return merge(final_recipe(), {"pools": setpools(tag), "ignore_unlocalized": True, "label": SETTINGS_LABEL})


def tag_of(x):
    return f"{x:g}".replace(".", "p")


def settings_amplitude():
    """Where the settings come from, part 2: the amplitude. v1 drew A from 1.65-1.85 rad, which its authors call
    arbitrary; they suggest a narrower range could improve the results. Physically the amplitude matters through the
    Floquet spectrum: across 1.60-1.95 the fastest growth rate rises from 0.042 to 0.17 and the number of growing ring
    modes from 2 to 7. Pools at fixed amplitudes (zero width, the narrowest possible range) and one wide pool
    (1.60-1.95), each 60,000 / 5,000 / 10,000 runs at offsets +-0.025, labels searched to tau = 300."""
    t = Table(None, "settings_amplitude")
    three = (0, 1, 2)
    # Predictions, written before the runs started:
    # - The median localization time times the fastest growth rate stays near 5.7 (5.6-5.9 measured on 1.65-1.85).
    # - Accuracy stays near the main model's at every amplitude where at most about 6 modes grow; with 7 modes (1.95)
    #   rival humps compete more often and it may fall. Near threshold (1.60, 2 modes) it should be at least as high.
    # - A model trained at one amplitude gains little over the wide model at that amplitude (under 0.5 points within
    #   +-1), because v2 is already flat across 1.65-1.85; so narrowing the range is not what makes the problem easy.
    for a in AMPLITUDES:
        t.base = _settings_base(f"amp{tag_of(a)}")
        t.add("fixed", f"amp{tag_of(a)}", f"settings: trained and tested at the fixed amplitude A = {a}", three)
    t.base = _settings_base("ampwide")
    t.add("wide", "ampwide", "settings: trained on A ~ U(1.60, 1.95) (scored per amplitude on the fixed pools later)", three)
    return t.configs


def settings_ring():
    """Where the settings come from, part 3: the number of pendulums. N = 100 comes from v1. The growing modes have
    wavelengths of at least about 20 pendulums at A = 1.75 and the hump is about 8 wide, so a longer ring holds more
    competing modes (about N/20) at the same growth rate, whereas changing the amplitude changes both. Pools of
    50, 200 and 400 pendulums with v1's amplitudes and offsets (60,000 / 5,000 / 10,000 runs), and the main pools for
    100. The same whole-ring architecture (receptive field 513) at every N."""
    t = Table(None, "settings_ring")
    three = (0, 1, 2)
    # Predictions, written before the runs started:
    # - If more competing modes make the site harder to call, within +-1 falls with N (rival humps more frequent);
    #   if the outcome is set locally once the hump forms, it stays flat. Amplitude slices (3-6 modes) were flat, so
    #   expect a mild fall at most up to 200 and a clearer one at 400 (about 20 growing modes).
    # - The model trained at N = 100 transfers to other N when it sees the whole ring (the earlier drop to 6.3 at
    #   N = 150 used a receptive field of 129, smaller than the ring).
    for n in (50, 100, 200, 400):
        t.base = merge(_settings_base("main" if n == 100 else f"n{n}"), {"model": WIDE_ARCH})
        t.add("ring", f"ring_n{n}", f"settings: ring of {n} pendulums, whole-ring architecture (receptive field 513)", three)
    return t.configs


def settings_offsets():
    """Where the settings come from, part 4: the size of the initial offsets. They are the release scatter: the
    uniform swing is an exact solution, so without them nothing localizes, and their pattern decides where. v1 states
    +-0.025 rad; its data were probably made at +-0.0125. The same runs as the main pools with the offsets about A
    scaled by 1/8, 1/4, 1/2, 2 and 4 (+-0.003 to +-0.1 rad), so only the size changes; labels searched to tau = 300."""
    t = Table(None, "settings_offsets")
    three = (0, 1, 2)
    # Predictions, written before the runs started:
    # - Each halving delays localization by about ln2 / sigma_1 (linear growth from a smaller start).
    # - Smaller offsets leave the non-growing modes relatively smaller when the hump forms, so the site changes at the
    #   one-pendulum level between sizes (37% of runs moved, 95% within +-1, at 1/2 on the first pendulum over the top).
    # - Accuracy stays flat for small offsets (v2 99.4% at 1/2 on the 4% label). At 2-4x the offsets are no longer
    #   small (0.1 rad) and the growth leaves the linear regime sooner, so accuracy may fall there.
    for s in OFFSET_SCALES:
        t.base = _settings_base(f"off{tag_of(s)}")
        t.add("offsets", f"off{tag_of(s)}", f"settings: offsets scaled by {s} (+-{0.025 * s:g} rad)", three)
    return t.configs


PHASES = {"phase2": phase2, "phase3": phase3, "phase4": phase4, "phase5": phase5, "phase6": phase6,
          "phase6_deploy": phase6_deploy, "phase6_label": phase6_label, "phase6_damping": phase6_damping,
          "phase7": phase7, "phase8": phase8, "phase3_avg": phase3_avg, "phase9": phase9, "deploy_start": deploy_start,
          "phase10": phase10, "threshold": threshold,
          "settings_hump": settings_hump, "settings_amplitude": settings_amplitude, "settings_ring": settings_ring,
          "settings_offsets": settings_offsets}
