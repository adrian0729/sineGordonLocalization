"""Report figures from the saved results, written to figures/ as PNG and PDF.

    python scripts/make_figures.py            # every figure whose inputs exist
    python scripts/make_figures.py --paper    # only the preprint's five figures, into papers/v2_preprint/figures/

Style follows the data-viz rules used for the report: one hue per job, the reference categorical
order (blue, orange, aqua), a single-hue sequential ramp, thin marks, hairline grids, text in ink
colors (never series colors), and a legend whenever two or more series share a plot.

The paper figures reuse the report's plotting functions: given axes, a function draws into them in the
paper's style (no titles, shorter labels, v1 orange and v2 blue) and leaves saving to the caller. They are
drawn at the printed width, 6.5 in, so the font sizes are the printed ones.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical order: blue, orange, aqua, yellow
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "lines.markersize": 6,
})


PAPER = ROOT / "papers" / "v2_preprint" / "figures"
PAPER_WIDTH = 6.5  # inches: the preprint's text width
PAPER_RC = {  # printed sizes: axis labels and ticks at 8 pt or more; math upright, like the plain-text symbols
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.3, "lines.markersize": 4.5, "axes.linewidth": 0.6, "grid.linewidth": 0.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "legend.handlelength": 1.6, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue",
    "mathtext.bf": "Helvetica Neue:bold", "mathtext.sf": "Helvetica Neue", "mathtext.cal": "Helvetica Neue",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",  # the printed page
}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"figures/{name}.png")


def subplots(axes, *args, **kwargs):
    """plt.subplots for a report figure; for a paper figure, the axes given and their figure."""
    if axes is None:
        return plt.subplots(*args, **kwargs)
    return (axes if isinstance(axes, plt.Axes) else axes[0]).figure, axes


def hbars(ax, labels, values, color=SLOT[0], fmt="{:.2f}", xmax=None, highlight=None, pad=0.012):
    """Thin horizontal bars from one baseline, value at the tip (pad: gap as a share of the axis)."""
    y = np.arange(len(labels))[::-1]
    xmax = xmax or max(values) * 1.18
    colors = [SLOT[1] if highlight and lab in highlight else color for lab in labels]
    ax.barh(y, values, height=0.56, color=colors, linewidth=0)
    for yi, v in zip(y, values):
        ax.text(v + pad * xmax, yi, fmt.format(v), va="center", ha="left", color=INK, fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def fig_growth(axes=None):
    p1 = json.loads((ROOT / "results" / "phase1.json").read_text())
    g = p1["growth"]
    A, modes, sigma = np.array(g["A"]), np.array(g["modes"]), np.array(g["sigma"])
    paper = axes is not None
    fig, (a, b) = subplots(axes, 1, 2, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1.5, 1]},
                           layout="constrained")
    if not paper:
        fig.get_layout_engine().set(wspace=0.08)
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["white" if paper else RAMP[0]] + RAMP[1:])  # from the background
    dA = A[1] - A[0]
    mesh = a.pcolormesh(np.r_[A - dA / 2, A[-1] + dA / 2], np.r_[modes - 0.5, modes[-1] + 0.5],
                        np.clip(sigma, 0, None).T, cmap=cmap, vmin=0, vmax=sigma.max(), shading="flat")
    cb = fig.colorbar(mesh, ax=a, pad=0.02)
    cb.set_label("Growth rate σ (per time unit)", color=INK2)
    cb.outline.set_visible(False)
    a.axvline(g["threshold"], color=INK2, linewidth=0.8)
    a.text(g["threshold"] + 0.004, 10.9 if paper else 11.6,  # the paper's narrower panel: below the data-range note
           f"threshold\n{'$A_c$' if paper else 'A_c'} = {g['threshold']:.3f}", va="top", fontsize=8, color=INK)
    a.plot([1.65, 1.85], [12.2, 12.2], color=INK2, linewidth=1.2, solid_capstyle="butt")
    a.text(1.75, 11.9, "amplitudes in the data", ha="center", va="top", fontsize=8, color=INK2)
    a.set_xlabel("Swing amplitude A (rad)")
    a.set_ylabel("Ring mode m (wavelength 100/m)" if paper else "Ring mode m (wavelength 100/m sites)")
    a.set_yticks(modes)
    a.grid(False)
    if not paper:
        a.set_title("(a) Which ring modes grow", loc="left")

    j = int(np.argmin(np.abs(A - 1.75)))
    s175 = sigma[j]
    b.bar(modes, np.clip(s175, 0, None), width=0.55, color=[SLOT[0] if v > 1e-6 else GRID for v in s175], linewidth=0)
    for m, v in zip(modes, s175):
        if v > 1e-6:
            b.text(m, v + 0.003, f"{1 / v:.1f}", ha="center", va="bottom", fontsize=7.5 if paper else 7, color=INK2)
    b.text(5.6, 0.131 if paper else 0.118, ("A = 1.75\n" if paper else "") + "labels: e-folding time\n(time units)",
           fontsize=7.5, color=INK2, va="top")  # the paper names the amplitude here and the unit on the axis
    b.set_ylim(0, 0.137)
    b.text(9.2, 0.012, "stable (σ = 0)", ha="center", fontsize=8, color=INK2)
    b.set_xticks(modes)
    b.set_xlabel("Ring mode m")
    b.set_ylabel("Growth rate σ (per time unit)" if paper else "Growth rate σ")
    b.grid(axis="x", visible=False)
    if not paper:
        b.set_title("(b) A = 1.75: only modes 1-5 grow", loc="left")
        save(fig, "fig1_growth_spectrum")


def fig_best_width(ax=None):
    p1 = json.loads((ROOT / "results" / "phase1.json").read_text())
    rows = p1["best_width"]
    A = np.array([r["A"] for r in rows])
    pred = np.array([r["predicted_K"] for r in rows])
    meas = np.array([r["measured_K"] for r in rows])
    lo = np.array([r["measured_K_16_84"][0] for r in rows])
    hi = np.array([r["measured_K_16_84"][1] for r in rows])
    paper = ax is not None
    fig, ax = subplots(ax, figsize=(5.4, 3.4))
    ax.plot(A, pred, color=SLOT[0], marker="o", markersize=4 if paper else 5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label="$K^* = N/m_c$" if paper else "Predicted: K* = N / (first stable mode)")
    ax.errorbar(A, meas, yerr=[meas - lo, hi - meas], fmt="none", ecolor=SLOT[1], elinewidth=1.2, capsize=0)
    ax.plot(A, meas, linestyle="none", marker="o", markersize=5.5 if paper else 7, color=SLOT[1], markeredgecolor=SURFACE,
            markeredgewidth=1.2 if paper else 1.5, label="measured" if paper else "Measured best width (16-84% bootstrap range)")
    note, at = ("empirical optimum,\nK = 17", (1.80, 25)) if paper else ("paper's empirical optimum, K = 17", (1.775, 27))
    ax.annotate(note, xy=(1.75, 17), xytext=at, fontsize=8, color=INK, ha="center" if paper else "left",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlabel("Swing amplitude A (rad)")
    ax.set_ylabel("Box-car width K (sites)")
    ax.set_ylim(8, 44)
    ax.legend(loc="upper right")
    if not paper:
        ax.set_title("Best smoothing width follows the Floquet band edge", loc="left")
        save(fig, "fig2_best_width")


def fig_sensitivity(axes=None):
    p1 = json.loads((ROOT / "results" / "phase1.json").read_text())
    pert = p1["sensitivity"]["perturbations"]
    knock = p1["sensitivity"]["knockouts"]
    rms = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    m1 = [100 * pert[f"white {r:g}"]["moved_over_1"] for r in rms]
    m3 = [100 * pert[f"white {r:g}"]["moved_over_3"] for r in rms]
    paper = axes is not None
    fig, (a, b) = subplots(axes, 1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.35, 1]},
                           layout="constrained")
    a.plot(rms, m1, color=SLOT[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2,
           label="moved > 1 site" if paper else "moved by more than 1 site")
    a.plot(rms, m3, color=SLOT[1], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2,
           label="moved > 3 sites" if paper else "moved by more than 3 sites")
    band_lo = 100 * pert["band m<=5 0.001"]["moved_over_1"]
    band_hi = 100 * pert["band m>=6 0.001"]["moved_over_1"]
    a.plot([1e-3], [band_lo], marker="D", linestyle="none", markersize=5 if paper else 6, color=SLOT[2],
           markeredgecolor=SURFACE, label="error in m ≤ 5 only" if paper else
           f"1e-3 error only in growing modes m ≤ 5 (>1 site): {band_lo:.0f}%")
    a.plot([1e-3], [band_hi], marker="D", linestyle="none", markersize=5 if paper else 6, color=INK2,
           markeredgecolor=SURFACE, label="error in m ≥ 6 only" if paper else
           f"1e-3 error only in stable modes m ≥ 6 (>1 site): {band_hi:.1f}%")
    a.set_xscale("log")
    a.set_xlabel("Initial error (rad, rms per pendulum)" if paper else "Error in the initial angles (rad, rms per pendulum)")
    a.set_ylabel("Labels that change (%)")
    a.set_ylim(0, 100 if paper else 80)  # the paper's narrow panel needs the room above the data for its legend
    a.legend(loc="upper left")
    if paper:
        a.set_xticks([1e-6, 1e-4, 1e-2])
    else:
        a.set_title("(a) How precisely the start must be known", loc="left")

    names = ["keep m ≤ 2", "keep m ≤ 3", "keep m ≤ 5", "keep m ≤ 6", "keep m ≤ 8", "keep m ≤ 12", "keep m ≤ 25", "remove m ≤ 5"]
    keys = ["keep m<=2", "keep m<=3", "keep m<=5", "keep m<=6", "keep m<=8", "keep m<=12", "keep m<=25", "remove m<=5"]
    hbars(b, names, [100 * knock[k]["moved_over_3"] for k in keys], color=SLOT[1] if paper else SLOT[0],  # paper: the > 3 sites color
          fmt="{:.0f}%", xmax=112, pad=0.035 if paper else 0.012)
    b.set_xlabel("Labels moved > 3 sites (%)" if paper else "Labels moved by more than 3 sites (%)")
    if not paper:
        b.set_title("(b) Removing parts of the initial noise", loc="left")
        save(fig, "fig3_label_sensitivity")


def load_runs():
    recs = []
    with open(ROOT / "results" / "runs.jsonl") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def fig_baselines():
    recs = load_runs()
    by_model = defaultdict(list)
    for r in recs:
        if r.get("phase") in (0, 1) and r.get("label", "right") == "right":
            by_model[r["model"]].append(r["test"])
    order = [("boxcar", "Box-car K = 17 on E (no training)"), ("floquet_filter", "Floquet filter (no training)"),
             ("linear_dtheta0", "Linear ring filter on δθ(0)"), ("v1_cnn_bilstm", "v1 CNN-BiLSTM, rebuilt (mean of 2 seeds)"),
             ("linear_dtheta0_amp", "Linear filter, amplitude-dependent"), ("linear_v1_inputs", "Linear filters on v1's inputs")]
    order = [(k, lab) for k, lab in order if k in by_model]
    labels = [lab for _, lab in order]
    mae = [np.mean([t["mae"] for t in by_model[k]]) for k, _ in order]
    acc = [100 * np.mean([t["acc_1"] for t in by_model[k]]) for k, _ in order]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 2.9), sharey=True)
    hbars(a, labels, mae, fmt="{:.2f}", xmax=7.0, highlight={"v1 CNN-BiLSTM, rebuilt (mean of 2 seeds)"})
    a.set_xlabel("Mean distance to the true site (lower is better)")
    hbars(b, labels, acc, fmt="{:.1f}%", xmax=100, highlight={"v1 CNN-BiLSTM, rebuilt (mean of 2 seeds)"})
    b.set_xlabel("Within ±1 site (%, higher is better)")
    a.set_title("Baselines on the test pool (20,000 runs)", loc="left")
    save(fig, "fig4_baselines")


def phase_records(phase):
    """{(group, name): [records]} for one ablation table, matched by config id."""
    from sgloc import ablations, experiment
    by_id = {}
    for r in load_runs():
        if r.get("config_id"):
            by_id[r["config_id"]] = r
    out = defaultdict(list)
    for cfg in ablations.PHASES[phase]():
        rec = by_id.get(experiment.config_id(cfg))
        if rec:
            out[(cfg["group"], cfg["name"])].append(rec)
    return out


def fig_receptive_field(ax=None):
    recs = phase_records("phase2")
    rfs = [9, 17, 33, 65, 129]
    pts = [(rf, [r["test"]["mae"] for r in recs.get(("receptive_field", f"local_rf{rf}"), [])]) for rf in rfs]
    pts = [(rf, v) for rf, v in pts if v]
    if not pts:
        raise FileNotFoundError("no receptive-field runs yet")
    paper = ax is not None
    fig, ax = subplots(ax, figsize=(5.4, 3.3))
    x = [rf for rf, _ in pts]
    y = [np.mean(v) for _, v in pts]
    for rf, v in pts:
        ax.plot([rf] * len(v), v, linestyle="none", marker="o", markersize=3 if paper else 4, color=SLOT[0], alpha=0.45)
    ax.plot(x, y, color=SLOT[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2, label="local-only dilated CNN")
    lin = [r["test"]["mae"] for r in load_runs() if r.get("model") == "linear_dtheta0_amp"]
    if lin:
        ax.axhline(lin[0], color=INK2, linewidth=0.9)
        if paper:  # right of where the CNN crosses the line
            ax.text(94, lin[0] + 0.06, "linear ring filter\n(200 parameters)", fontsize=7.5, color=INK2, ha="right")
        else:
            ax.text(9, lin[0] + 0.08, "linear ring filter, amplitude-dependent (200 parameters)", fontsize=7.5, color=INK2)
    ax.axvline(100, color=MUTED if paper else GRID, linewidth=0.9,  # in the paper, dashed apart from the grid
               linestyle=(0, (3, 2)) if paper else "-")
    ax.text(97 if paper else 102, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] < 6 else 5.2, "whole ring", fontsize=7.5,
            color=INK2, ha="right" if paper else "left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rfs, [str(r) for r in rfs])
    ax.set_xlabel("Receptive field (sites)")
    ax.set_ylabel("Mean distance on test (sites)")
    if not paper:
        ax.set_title("A site must see most of the ring", loc="left")
        save(fig, "fig5_receptive_field")


def fig_data_efficiency(ax=None):
    recs = phase_records("phase2")
    ns = [1000, 3000, 10_000, 30_000, 60_000]
    paper = ax is not None
    fig, ax = subplots(ax, figsize=(5.6, 3.4))
    plotted = False
    series = (("circular", "circular, one score per site" if paper else "circular padding, one score per site"),
              ("zero_pad", "zero padding"),
              ("dense_head", "v1-style flatten + dense head" if paper else "flatten + dense head (like v1)"))
    colors = [SLOT[0], SLOT[2], SLOT[3]] if paper else SLOT  # in the paper, orange is kept for v1
    for color, (name, label) in zip(colors, series):
        pts = [(n, [r["test"]["mae"] for r in recs.get(("symmetry", f"{name}_n{n}"), [])]) for n in ns]
        pts = [(n, v) for n, v in pts if v]
        if not pts:
            continue
        plotted = True
        ax.plot([n for n, _ in pts], [np.mean(v) for _, v in pts], color=color, marker="o",
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    if not plotted:
        raise FileNotFoundError("no symmetry runs yet")
    v1 = [r["test"]["mae"] for r in load_runs() if r.get("model") == "v1_cnn_bilstm" and r.get("label", "right") == "right"]
    if v1:
        ax.plot([60_000], [np.mean(v1)], marker="D", linestyle="none", color=SLOT[1] if paper else INK2,
                markeredgecolor=SURFACE, label="rebuilt v1 (26.5M parameters)" if paper else "v1 rebuilt (26.5M parameters)")
    ax.set_xscale("log")
    ax.set_xlabel("Training runs")
    ax.set_ylabel("Mean distance on test (sites)")
    ax.legend(loc="upper right")
    if paper:
        ax.set_xticks(ns, [f"{n // 1000}k" for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_title("Respecting the ring's symmetry saves data", loc="left")
        save(fig, "fig6_data_efficiency")


def fig_architectures():
    recs = phase_records("phase2")
    picks = [("time", "single_frame", "single frame, Fourier + local (early fusion)"),
             ("time", "early_fusion", "10 frames stacked (early fusion)"),
             ("time", "conv2d", "2D (time x site) convolution front"),
             ("time", "pointwise_mlp", "per-site MLP front"),
             ("time", "frame_gru", "per-site GRU front"),
             ("scales", "global_only", "Fourier branch only"),
             ("scales", "local_only", "local branch only (sees 11 sites)"),
             ("receptive_field", "local_rf33", "dilated, sees 33 sites"),
             ("receptive_field", "local_rf65", "dilated, sees 65 sites"),
             ("receptive_field", "local_rf129", "dilated, sees the whole ring"),
             ("combine", "dilated129_plus_fourier", "whole-ring dilated + Fourier"),
             ("combine", "dilated129_mlp_front", "whole-ring dilated + MLP front"),
             ("combine", "dilated129_gru_front", "whole-ring dilated + GRU front"),
             ("combine", "dilated129_w64", "whole-ring dilated, width 64"),
             ("symmetry", "dense_head_n60000", "flatten + dense head (like v1)")]
    rows = [(lab, [r["test"]["mae"] for r in recs.get((g, n), [])]) for g, n, lab in picks]
    rows = [(lab, v) for lab, v in rows if v]
    v1 = [r["test"]["mae"] for r in load_runs() if r.get("model") == "v1_cnn_bilstm" and r.get("label", "right") == "right"]
    if v1:
        rows.append(("v1 rebuilt (26.5M parameters)", v1))
    rows.sort(key=lambda r: -np.mean(r[1]))
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(rows) + 1.0))
    y = np.arange(len(rows))
    for yi, (lab, v) in zip(y, rows):
        c = SLOT[1] if lab.startswith("v1") else SLOT[0]
        ax.plot(v, [yi] * len(v), linestyle="none", marker="o", markersize=4, color=c, alpha=0.4)
        ax.plot([np.mean(v)], [yi], linestyle="none", marker="o", markersize=7, color=c, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.text(max(v) + 0.06, yi, f"{np.mean(v):.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean distance on test (sites); small dots are seeds, large dots their mean")
    ax.set_title("Phase 2: architectures (same data and training budget)", loc="left")
    save(fig, "fig7_architectures")


ORDINAL4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # validated ordinal ramp (scripts/validate_palette.js --ordinal)


def fig_noise_time():
    recs = phase_records("phase3")
    noises = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    starts = [0, 10, 20, 30]
    setups = [("S1", "(a) Camera: angles, 8 frames (4 time units)"), ("S2", "(b) Camera + gyros: angles and velocities, 1 frame"),
              ("S3", "(c) Energies computed from (b), 1 frame")]
    if not any(recs.get(("noise_time", f"S1_t0_noise{n:g}")) for n in noises):
        raise FileNotFoundError("no noise/time runs yet")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True, sharey="row", layout="constrained")
    x_plot = [3e-5] + noises[1:]  # noise-free plotted at the left edge of the log axis
    for col, (kind, title) in enumerate(setups):
        for color, start in zip(ORDINAL4, starts):
            runs = [(xp, recs.get(("noise_time", f"{kind}_t{start}_noise{n:g}"))) for xp, n in zip(x_plot, noises)]
            runs = [(xp, r[0]["test"]) for xp, r in runs if r]
            if not runs:
                continue
            for row, (key, scale) in enumerate((("acc_3", 100), ("mae", 1))):
                axes[row, col].plot([xp for xp, _ in runs], [scale * t[key] for _, t in runs], color=color, marker="o",
                                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=f"watch from τ = {start}")
        axes[0, col].set_title(title, loc="left", fontsize=9)
        axes[1, col].set_xscale("log")
        axes[1, col].set_xticks(x_plot, ["none", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"])
        axes[1, col].set_xlabel("Sensor noise (rad)")
    axes[0, 0].set_ylabel("Within ±3 sites (%)")
    axes[1, 0].set_ylabel("Mean distance (sites)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Sensor precision against how long you wait (test pool; localization happens at τ ≈ 57)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    save(fig, "fig8_noise_vs_time")


def fig_noise_collapse(axes=None):
    """H1.4 test: if waiting only helps by amplifying the growing modes, the error depends on sensor noise
    and start time only through the noise relative to the grown signal, sigma_s * exp(-sigma_1 * tau_obs).
    The paper shows the camera alone: pass one axes."""
    recs = phase_records("phase3")
    sigma1 = 0.118  # fastest Floquet growth rate at A = 1.75 (Phase 1)
    noises = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    setups = [("S1", "(a) Camera: angles, 8 frames"), ("S2", "(b) Angles and velocities, 1 frame"),
              ("S3", "(c) Energies, 1 frame")]
    if not any(recs.get(("noise_time", f"S1_t0_noise{n:g}")) for n in noises):
        raise FileNotFoundError("no noise/time runs yet")
    paper = axes is not None
    fig, axes = subplots(axes, 1, 3, figsize=(11.5, 3.5), sharey=True, layout="constrained")
    for ax, (kind, title) in zip(axes, setups):
        clean = recs.get(("noise_time", f"{kind}_t0_noise0"))
        if clean:
            ax.axhline(clean[0]["test"]["mae"], color=INK2, linewidth=0.8, label="noise-free, from τ = 0" if paper else None)
        for color, start in zip(ORDINAL4, (0, 10, 20, 30)):
            pts = [(n * np.exp(-sigma1 * start), recs.get(("noise_time", f"{kind}_t{start}_noise{n:g}"))) for n in noises]
            pts = [(x, r[0]["test"]["mae"]) for x, r in pts if r]
            if pts:
                ax.plot(*zip(*pts), color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=f"watch from τ = {start}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if paper:
            ax.set_xlabel(r"Sensor noise ÷ growth before watching, $\sigma_s\,e^{-\sigma_1\tau}$ (rad)")
        else:
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_xlabel("Sensor noise ÷ growth since release, σ_s·exp(−σ₁τ) (rad)")
    ticks = [1.5, 2, 3, 5, 8] if paper else [1.5, 2, 3, 5, 8, 12]  # the camera alone stays below 8; ticks widen the axis
    axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("Mean distance on test (sites)")
    axes[0].legend(loc="upper left")
    if not paper:
        fig.suptitle("Waiting divides the sensor noise by the growth of the unstable modes (line: noise-free, from release)",
                     x=0.01, ha="left", fontweight="bold", fontsize=10)
        save(fig, "fig13_noise_collapse")


def fig_noise_avg_label(ax=None):
    """Phase 3 addendum: Fig. 13's collapse on the swing-averaged label (top), and the waiting cost per 10x noise
    fitted by scripts/noise_tradeoff.py against the Floquet growth rates, on both labels (bottom).
    With ax given, draws only panel (d), for the paper."""
    recs = phase_records("phase3_avg")
    path = ROOT / "results" / "noise_tradeoff.json"
    if not recs.get(("noise_time", "S1_t0_noise0.001_avg")) or not path.exists():
        raise FileNotFoundError("no phase3_avg runs or trade-off fit yet (scripts/noise_tradeoff.py)")
    fits = json.loads(path.read_text())
    paper = ax is not None
    labels = [("right", "first-crossing label" if paper else "v1's label (Phase 3)", SLOT[0]),
              ("avg", "swing-averaged label", SLOT[2])]  # as in Fig. 19
    if paper:
        return noise_cost_by_amplitude(ax, fits, labels, paper=True)
    sigma1, noises = fits["sigma1"], [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    setups = [("S1", "camera, 8 angle frames"), ("S2", "angles and velocities, 1 frame"), ("S3", "energies, 1 frame")]
    fig, ax = plt.subplot_mosaic([["a", "b", "c"], ["d", "d", "e"]], figsize=(11.5, 7.2), layout="constrained")
    for key, (kind, title) in zip("abc", setups):
        a = ax[key]
        if key != "a":
            a.sharey(ax["a"])
            a.tick_params(labelleft=False)
        clean = recs.get(("noise_time", f"{kind}_t0_noise0_avg"))
        if clean:
            a.axhline(clean[0]["test"]["mae"], color=INK2, linewidth=0.8)
        for color, start in zip(ORDINAL4, (0, 10, 20, 30)):
            pts = [(n * np.exp(-sigma1 * start), recs.get(("noise_time", f"{kind}_t{start}_noise{n:g}_avg"))) for n in noises]
            pts = [(x, r[0]["test"]["mae"]) for x, r in pts if r]
            if pts:
                a.plot(*zip(*pts), color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.0,
                       label=f"watch from τ = {start}")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_title(f"({key}) Averaged label: {title}", loc="left", fontsize=9)
        a.set_xlabel("Sensor noise ÷ growth since release, σ_s·exp(−σ₁τ) (rad)")
    ax["a"].set_yticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
    ax["a"].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax["a"].set_ylabel("Mean distance on test (sites)")
    ax["a"].legend(loc="upper left")
    noise_cost_by_amplitude(ax["d"], fits, labels)

    # (e) the whole test pool, every setup
    e = ax["e"]
    for x, name in ((fits["predicted_cost_per_decade"], "σ₁"),
                    (np.log(10) / fits["amplitude_bins"]["all"]["sigma_bar"], "σ̄")):
        e.axvline(x, color=INK2 if name == "σ̄" else MUTED, linewidth=0.8)
        e.text(x, 2.62, f" {name}", color=INK2, fontsize=7.5, ha="left", va="bottom")
    for i, (label, name, color) in enumerate(labels):
        rows = [(j, fits["labels"].get(label, {}).get(kind)) for j, (kind, _) in enumerate(setups)]
        rows = [(2 - j + (0.5 - i) * 0.28, f) for j, f in rows if f]
        for y, f in rows:
            e.hlines(y, *f["cost_ci95"], color=color, linewidth=1.2)
        e.plot([f["cost_per_decade"] for _, f in rows], [y for y, _ in rows], linestyle="none", marker="o", markersize=6,
               color=color, markeredgecolor=SURFACE, markeredgewidth=1.2)
    e.set_yticks([2, 1, 0], [kind for kind, _ in setups])
    e.set_ylim(-0.6, 2.9)
    e.tick_params(axis="y", length=0)
    e.grid(axis="y", visible=False)
    e.set_xlabel("Waiting per 10× noise, whole test pool (time units)")
    e.set_title("(e) Whole test pool, every setup (colors as in d)", loc="left", fontsize=9)
    fig.suptitle("On either label, waiting divides the sensor noise at about the mean growth rate of the unstable modes, "
                 "not the fastest", x=0.01, ha="left", fontweight="bold", fontsize=10)
    save(fig, "fig13b_noise_avg_label")


def noise_cost_by_amplitude(d, fits, labels, paper=False):
    """Fig. 13b (d): the camera's fitted waiting cost per amplitude bin of the test runs, against each growth rate's
    prediction. The report names the predictions beside the last bin, the paper in the legend."""
    bins = {b: g for b, g in fits["amplitude_bins"].items() if b != "all"}
    predictions = ((("sigma1", "fastest mode", MUTED), ("sigma_bar", "mean growth rate of the growing modes", INK)) if paper
                   else (("sigma1", "fastest mode, ln 10 / σ₁", MUTED), ("sigma_bar", "mean of the growing\nmodes, ln 10 / σ̄", INK)))
    for key, name, color in predictions:
        for b, g in bins.items():  # each bin's prediction, from the rate averaged over its test runs
            d.hlines(np.log(10) / g[key], g["A"][0] + 0.004, g["A"][1] - 0.004, color=color, linewidth=1.2,
                     label=name if paper and b == min(bins) else None)
        if not paper:
            last = bins[max(bins)]
            d.text(last["A"][1] + 0.002, np.log(10) / last[key], name, color=INK2, fontsize=7.5, va="center", ha="left")
    for i, (label, name, color) in enumerate(labels):
        fit = fits["labels"].get(label, {}).get("S1")
        if not fit:
            continue
        x = np.array([bins[b]["A_mean"] for b in sorted(bins)]) + (i - 0.5) * 0.006
        cost = [fit["by_amplitude"][b]["cost_per_decade"] for b in sorted(bins)]
        lo = [fit["by_amplitude"][b]["cost_ci95"][0] for b in sorted(bins)]
        hi = [fit["by_amplitude"][b]["cost_ci95"][1] for b in sorted(bins)]
        d.vlines(x, lo, hi, color=color, linewidth=1.2)
        d.plot(x, cost, linestyle="none", marker="o", markersize=4.5 if paper else 6, color=color, markeredgecolor=SURFACE,
               markeredgewidth=1.2, label=name)
    d.set_xlim(1.645, 1.855 if paper else 1.92)
    d.set_xticks([1.65, 1.70, 1.75, 1.80, 1.85])
    d.set_ylabel("Waiting per 10× noise (time units)")
    if paper:  # the labels' legend above the right bins, the predictions' in room left below the data
        d.set_xlabel("Swing amplitude A of the test runs (rad)")
        d.set_ylim(5, 43)
        handles, names = d.get_legend_handles_labels()
        d.add_artist(d.legend(handles[2:], names[2:], loc="upper right"))
        d.legend(handles[1::-1], names[1::-1], loc="lower left")
    else:
        d.set_xlabel("Swing amplitude A of the test runs (four bins; dots: fit, bars: 95% intervals, steps: predictions)")
        d.set_title("(d) Camera: each amplitude waits at about the mean rate of its growing modes", loc="left", fontsize=9)
        d.legend(loc="upper right")


def fig_warning_tradeoff():
    """Q1 deliverable: accuracy against the warning time left after the last frame, per sensing setup."""
    recs = phase_records("phase3")
    from sgloc import data
    t_loc = float(np.median(data.load_pool(ROOT / "data" / "pool_test.npz")["time_right"]))
    unit = (0.2 / 9.81) ** 0.5  # seconds per time unit for a 20 cm pendulum
    setups = [("S1", 3.5, "camera: angles, 8 frames"), ("S2", 0.0, "camera + gyros: angles and velocities"),
              ("S3", 0.0, "energies from camera + gyros")]
    if not recs.get(("noise_time", "S1_t0_noise0.001")):
        raise FileNotFoundError("no noise/time runs yet")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True, layout="constrained")
    for ax, noise, title in zip(axes, (1e-3, 1e-2), ("(a) Precise sensors: 1e-3 rad", "(b) Coarse sensors: 1e-2 rad")):
        for slot, (kind, window, label) in enumerate(setups):
            pts = [(t_loc - start - window, recs.get(("noise_time", f"{kind}_t{start}_noise{noise:g}"))) for start in (0, 10, 20, 30)]
            pts = [(w, r[0]["test"]["mae"]) for w, r in pts if r]
            ax.plot(*zip(*pts), color=SLOT[slot], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.0, label=label)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("Median warning left after the last frame (time units)")
        ax.set_yscale("log")
        top = ax.secondary_xaxis("top", functions=(lambda t: t * unit, lambda s: s / unit))
        top.set_xlabel("seconds, for a 20 cm pendulum", color=INK2)
        top.tick_params(colors=MUTED, labelcolor=INK2)
    axes[0].set_yticks([1.5, 2, 3, 5, 8, 12], ["1.5", "2", "3", "5", "8", "12"])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("Mean distance on test (sites)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Warning time against accuracy: watch later with a worse sensor (start times 30, 20, 10, 0 from left)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    save(fig, "fig17_warning_tradeoff")


def fig_inputs():
    recs = phase_records("phase3")
    names = [("I0_v1_inputs", "E at τ = 0…4.5 + θ(0) (v1's inputs)"), ("I1_theta0", "θ(0) only"), ("I2_E0", "E(0) only"),
             ("I3_theta_frames", "θ at τ = 0…4.5"), ("I4_E_frames", "E at τ = 0…4.5"),
             ("I5_theta_omega_random_start", "θ and θ' at a random time in the first swing"),
             ("I6_dtheta0_amplitude", "θ(0) minus chain mean + amplitude")]
    rows = [(lab, [r["test"]["mae"] for r in recs.get(("information", n), [])]) for n, lab in names]
    rows = [(lab, v) for lab, v in rows if v]
    if not rows:
        raise FileNotFoundError("no information runs yet")
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        ax.plot(v, [yi] * len(v), linestyle="none", marker="o", markersize=4, color=SLOT[0], alpha=0.4)
        ax.plot([np.mean(v)], [yi], linestyle="none", marker="o", markersize=7, color=SLOT[0], markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.text(max(v) + 0.03, yi, f"{np.mean(v):.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean distance on test (sites); 15-epoch screening runs, small dots are seeds")
    ax.set_title("Which measurements carry the information (noise-free)", loc="left")
    save(fig, "fig9_inputs")


def seed_dots(ax, rows, ref=None):
    """One row per config: small dots for seeds, a large dot for their mean, the mean as a label."""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        if not len(v):  # label only; the caller draws this row
            continue
        ax.plot(v, [yi] * len(v), linestyle="none", marker="o", markersize=4, color=SLOT[0], alpha=0.4)
        ax.plot([np.mean(v)], [yi], linestyle="none", marker="o", markersize=7, color=SLOT[0],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(max(v) + 0.02, yi, f"{np.mean(v):.2f}", va="center", fontsize=7.5, color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


def fig_losses():
    recs = phase_records("phase4")
    names = [("L1_ce", "cross-entropy (v1's site loss)"), ("L2_angle", "circular regression (angle)"),
             ("L3_soft0.5", "soft labels, σ = 0.5"), ("L3_soft1", "soft labels, σ = 1"), ("L3_soft2", "soft labels, σ = 2"),
             ("L4_soft1_dist0.03", "σ = 1 + 0.03 × expected distance"), ("L4_soft1_dist0.1", "σ = 1 + 0.1 × expected distance"),
             ("L4_soft1_dist0.3", "σ = 1 + 0.3 × expected distance"), ("L5_soft1_dist0.1_time", "+ localization-time head"),
             ("L6_soft1_dist0.1_profile", "+ later-energy-profile head"), ("L7_soft1_dist0.1_localizes", "+ v1's binary head")]
    rows = [(lab, [r["test"]["mae"] for r in recs.get(("loss", n), [])]) for n, lab in names]
    rows = [(lab, v) for lab, v in rows if v]
    if not rows:
        raise FileNotFoundError("no loss runs yet")
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.0))
    seed_dots(ax, rows, ref=np.mean(rows[0][1]))
    ax.set_xlabel("Mean distance on test (sites); line = plain cross-entropy")
    ax.set_title("Phase 4: loss functions (15-epoch screening runs)", loc="left")
    save(fig, "fig10_losses")


def fig_optimizers():
    recs = phase_records("phase5")
    names = [("optimizer", "O1_adam_constant", "Adam, constant 1e-3 (v1)"), ("optimizer", "O2_adamw_cosine", "AdamW + cosine"),
             ("optimizer", "O3_adamw_onecycle", "AdamW + one-cycle"), ("optimizer", "O4_sgd_nesterov_cosine", "SGD-Nesterov + cosine"),
             ("optimizer", "O5_lion_cosine", "Lion + cosine"), ("batch", "adamw_batch64", "batch 64"),
             ("batch", "adamw_batch1024", "batch 1024"), ("weight_decay", "adamw_wd0", "no weight decay"),
             ("weight_decay", "adamw_wd0.01", "weight decay 0.01")]
    rows = [(lab, [r["test"]["mae"] for r in recs.get((g, n), [])]) for g, n, lab in names]
    rows = [(lab, v) for lab, v in rows if v]
    if not rows:
        raise FileNotFoundError("no optimizer runs yet")
    ref = [v for lab, v in rows if lab == "AdamW + cosine"]
    cut = 2.3  # runs that did not train are drawn at the right edge with their value
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.0))
    seed_dots(ax, [(lab, v) if np.mean(v) < cut else (lab, []) for lab, v in rows], ref=np.mean(ref[0]) if ref else None)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        if np.mean(v) >= cut:
            ax.plot([cut - 0.02], [yi], linestyle="none", marker=">", markersize=7, color=SLOT[0])
            ax.text(cut - 0.06, yi, f"{np.mean(v):.1f}: did not train (off scale)", va="center", ha="right", fontsize=7.5, color=INK)
    ax.set_xlim(1.4, cut)
    ax.set_xlabel("Mean distance on test (sites); line = AdamW + cosine; small dots are seeds")
    ax.set_title("Phase 5: optimizers and schedules (15-epoch screening runs)", loc="left")
    save(fig, "fig11_optimizers")


def fig_sparse():
    recs = phase_records("phase3")
    strides = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    plotted = False
    for color, start in ((SLOT[0], 0), (SLOT[1], 20)):
        for style, noise in (("-", 0.0), ("--", 1e-3)):
            pts = []
            for s in strides:
                r = recs.get(("sparse", f"S1_t{start}_noise{noise:g}_every{s}"))
                if r:
                    pts.append((s, r[0]["test"]["mae"]))
            if pts:
                plotted = True
                ax.plot(*zip(*pts), color=color, linestyle=style, marker="o", markeredgecolor=SURFACE,
                        markeredgewidth=1.0, label=f"watch from τ = {start}, noise {'none' if noise == 0 else '1e-3 rad'}")
    if not plotted:
        raise FileNotFoundError("no sparse-sensor runs yet")
    ax.set_xscale("log", base=2)
    ax.set_xticks(strides, ["all 100", "every 2nd (50)", "every 4th (25)", "every 8th (13)"])
    ax.set_xlabel("Pendulums tracked by the camera")
    ax.set_ylabel("Mean distance on test (sites)")
    ax.legend(loc="upper left", fontsize=7.5, handlelength=3.2)
    ax.set_title("Tracking every 2nd or 4th pendulum costs little; every 8th aliases", loc="left")
    save(fig, "fig12_sparse_sensors")


def fig_leave_one_out():
    recs = phase_records("phase6")
    final = [r["test"]["mae"] for r in recs.get(("final", "v2_final"), [])]
    if not final:
        raise FileNotFoundError("no final runs yet")
    labels = {"no_dilation": "no dilation (sees 17 sites)", "zero_padding": "zero padding instead of circular",
              "dense_head": "flatten + dense head (like v1)", "plain_cross_entropy": "plain cross-entropy (v1's loss)",
              "no_soft_labels": "hard labels (other loss terms kept)", "no_time_head": "no localization-time head",
              "v1_inputs": "v1's inputs (energy frames + θ(0))", "adam_constant": "v1's optimizer (Adam, constant 1e-3)"}
    rows = [("v2 final model", final)] + [(lab, [r["test"]["mae"] for r in recs.get(("leave_one_out", n), [])])
                                          for n, lab in labels.items()]
    rows = [(lab, v) for lab, v in rows if v]
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.0))
    seed_dots(ax, rows, ref=np.mean(final))
    ax.set_xlabel("Mean distance on test (sites); line = final model; small dots are seeds")
    ax.set_title("Phase 6: take one choice out of the final model (30 epochs)", loc="left")
    save(fig, "fig14_leave_one_out")


def fig_final_data():
    recs = phase_records("phase6")
    pts = []
    for n in (1000, 3000, 10_000, 30_000, 60_000, 100_000):
        v = [r["test"]["mae"] for r in recs.get(("final", "v2_final") if n == 60_000 else ("data", f"final_n{n}"), [])]
        if v:
            pts.append((n, v))
    if len(pts) < 2:
        raise FileNotFoundError("not enough data-efficiency runs yet")
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for n, v in pts:
        ax.plot([n] * len(v), v, linestyle="none", marker="o", markersize=4, color=SLOT[0], alpha=0.4)
    ax.plot([n for n, _ in pts], [np.mean(v) for _, v in pts], color=SLOT[0], marker="o", markeredgecolor=SURFACE,
            markeredgewidth=1.2, label="v2 final recipe (23k parameters, 7,050 steps)")
    phase8 = phase_records("phase8")
    mirror = [(n, phase8.get((f"n{n}", f"n{n}_mirror"), [])) for n in (1000, 3000, 10_000, 60_000)]
    # the best Phase 8 variant at each size, chosen on the validation pool (needs 3 seeds, as the combinations have)
    best = [(n, min((rs for (g, _), rs in phase8.items() if g == f"n{n}" and len(rs) >= 3),
                    key=lambda rs: np.mean([r["val"]["mae"] for r in rs]), default=[])) for n in (1000, 3000, 10_000)]
    for slot, rows, label in ((1, mirror, "+ mirror images (Phase 8)"),
                              (2, best, "best regularization per size (Phase 8)")):
        rows = [(n, [r["test"]["mae"] for r in rs]) for n, rs in rows if rs]
        if not rows:
            continue
        for n, v in rows:
            ax.plot([n] * len(v), v, linestyle="none", marker="o", markersize=4, color=SLOT[slot], alpha=0.4)
        sizes = [n for n, _ in pts]
        for (n0, v0), (n1, v1_) in zip(rows, rows[1:]):  # join only neighbouring sizes: no line across an untested size
            if sizes.index(n1) == sizes.index(n0) + 1:
                ax.plot([n0, n1], [np.mean(v0), np.mean(v1_)], color=SLOT[slot])
        ax.plot([n for n, _ in rows], [np.mean(v) for _, v in rows], color=SLOT[slot], linestyle="none", marker="o",
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.plot([], [], color=SLOT[slot], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    v1 = [r["test"]["mae"] for r in load_runs() if r.get("model") == "v1_cnn_bilstm" and r.get("label", "right") == "right"]
    title = "Data efficiency of the final recipe"
    if v1:
        ax.axhline(np.mean(v1), color=INK2, linewidth=0.8, linestyle="--", label="v1 rebuilt, 60,000 runs (26.5M parameters)")
        beat = [n for n, v in pts if np.mean(v) < np.mean(v1)]
        if beat:
            title = f"v2 beats v1 from {min(beat):,} training runs, {60_000 // min(beat)}× less data"
    ax.set_xscale("log")
    ax.set_xticks([n for n, _ in pts], [f"{n // 1000}k" for n, _ in pts])
    ax.set_xlabel("Training runs")
    ax.set_ylabel("Mean distance on test (sites)")
    ax.legend(loc="center right", handlelength=3.2)
    ax.set_title(title, loc="left")
    save(fig, "fig15_final_data_efficiency")


def fig_regularization():
    """Phase 8: validation curves with little data, the final recipe against each regularizer (seed means)."""
    path = ROOT / "results" / "phase8_curves.json"
    if not path.exists():
        raise FileNotFoundError("no Phase 8 curves yet (scripts/regularization.py)")
    curves = json.loads(path.read_text())
    series = [("fine_eval", "final recipe"), ("mirror", "+ mirror images"), ("wd1", "+ weight decay 1"),
              ("mirror_wd1", "+ both")]  # same colors as Fig. 15: final recipe blue, mirror images orange
    sizes = [n for n in (1000, 3000, 10_000) if f"n{n}_fine_eval" in curves]
    if not sizes:
        raise FileNotFoundError("no Phase 8 runs yet")
    def smooth(steps, val, width=1.2):  # mean over checkpoints within ±20% of each step: even smoothing on a log axis
        return np.array([val[(steps >= s / width) & (steps <= s * width)].mean() for s in steps])

    fig, axes = plt.subplots(1, len(sizes), figsize=(3.5 * len(sizes) + 0.6, 3.4), layout="constrained", squeeze=False)
    for ax, n in zip(axes[0], sizes):
        shown = [(slot, label, curves[f"n{n}_{key}"]) for slot, (key, label) in enumerate(series) if f"n{n}_{key}" in curves]
        lows, ends = [], []
        for slot, label, c in shown:
            steps = np.array(c["steps"], dtype=float)
            val = smooth(steps, np.array(c["val"]))
            ax.plot(steps, val, color=SLOT[slot], linewidth=1.4, label=label)
            i = val.argmin()
            ax.plot(steps[i], val[i], linestyle="none", marker="o", markersize=5, color=SLOT[slot],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            lows.append(val[i])
            ends.append(val[-1])
        ax.axvline(211, color=INK2, linewidth=0.7, linestyle=":")
        low, high = min(lows), max(ends)
        bottom = low - 0.12 * (high - low)
        ax.set_ylim(bottom, low + 1.12 * (high - low))
        ax.text(205, bottom + 0.02 * (high - bottom), "warmup ends ", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(80, 7600)
        ax.set_xticks([100, 300, 1000, 3000, 7050], ["100", "300", "1k", "3k", "7,050"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Training step (log scale)")
        ax.set_title(f"({'abc'[sizes.index(n)]}) {n:,} training runs", loc="left", fontsize=9)
    axes[0, 0].set_ylabel("Validation mean distance (sites)")
    axes[0, 0].legend(loc="lower right", handlelength=2.4)
    fig.suptitle("Mirror images delay overfitting, which gives weight decay time to act (smoothed seed means; dots: lowest point)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    save(fig, "fig21_regularization")


def fig_label_jitter(axes=None):
    """Why v1's label is hard: the first pendulum over 10% jitters around the swing-averaged hotspot."""
    from sgloc import ablations, data, experiment
    pool = data.load_pool(ROOT / "data" / "pool_test.npz")
    right, avg = pool["site_right"].astype(int), pool["site_avg"].astype(int)
    errors = np.arange(-4, 5)

    def signed(a, b):
        return (a - b + 50) % 100 - 50

    def histogram(name, truth):
        hs = []
        for c in ablations.phase6():
            path = ROOT / "runs" / f"{c['name']}_s{c['seed']}_{experiment.config_id(c)}" / "pred_test.npy"
            if c["name"] == name and path.exists():
                e = signed(np.load(path).astype(int), truth)
                hs.append([100 * np.mean(e == k) for k in errors])
        return np.mean(hs, axis=0) if hs else None

    paper = axes is not None
    # the paper colors by label (blue: first-crossing, aqua: swing-averaged, as in Fig. 13b) and keeps orange for v1;
    # the report colors the series that exist in the categorical order
    series = [("plain cross-entropy,\nfirst-crossing label" if paper else "v2, plain cross-entropy, on v1's label",
               ORDINAL4[0], histogram("plain_cross_entropy", right)),
              ("final (soft labels),\nfirst-crossing label" if paper else "v2 final (soft labels), on v1's label",
               SLOT[0], histogram("v2_final", right)),
              ("final recipe,\nswing-averaged label" if paper else "v2 recipe on the swing-averaged label",
               SLOT[2], histogram("final_avg_label", avg))]
    series = [(lab, color, h) for lab, color, h in series if h is not None]
    series = [(lab, color if paper else SLOT[i], h) for i, (lab, color, h) in enumerate(series)]
    if len(series) < 2:
        raise FileNotFoundError("label runs not finished yet")
    fig, (a, b) = subplots(axes, 1, 2, figsize=(10.5, 3.4), layout="constrained")
    offset = signed(avg, right)
    a.bar(errors, [100 * np.mean(offset == k) for k in errors], width=0.6, color=SLOT[0], linewidth=0)
    a.set_xticks(errors)
    a.set_xlabel("Swing-averaged minus first-crossing site" if paper else "Swing-averaged site minus v1's site (pendulums)")
    a.set_ylabel("Share of test runs (%)")
    width = 0.8 / len(series)
    for i, (lab, color, h) in enumerate(series):
        b.bar(errors + (i - (len(series) - 1) / 2) * width, h, width=width * 0.9, color=color, linewidth=0, label=lab)
    b.set_xticks(errors)
    b.set_xlabel("Predicted minus true site" if paper else "Predicted minus true site (pendulums)")
    if paper:  # the legend above the short bars left of 0
        b.set_ylabel("Share of test runs (%)")
        b.legend(loc="upper left", handlelength=1.2, handletextpad=0.5)
    else:
        a.set_title("(a) The two labels agree in only a third of runs", loc="left", fontsize=9)
        b.set_title("(b) Errors of v2 against each label", loc="left", fontsize=9)
        b.legend(loc="upper left", fontsize=7.5)
        fig.suptitle("The first pendulum over 10% jitters around where the energy settles", x=0.01, ha="left",
                     fontweight="bold", fontsize=10)
        save(fig, "fig18_label_jitter")


def fig_avg_label_ablations():
    """Phase 7: the main choices again on the swing-averaged label, each beside its control on v1's label."""
    recs = phase_records("phase7")
    rows = [("inputs", "angle_frames", "v2 final recipe"), ("inputs", "v1_inputs", "v1's inputs (energy frames + θ(0))"),
            ("inputs", "theta0_only", "θ(0) only"), ("receptive_field", "rf33", "sees 33 sites, not the whole ring"),
            ("receptive_field", "rf17", "sees 17 sites"), ("soft_labels", "soft_off", "hard labels (other loss terms kept)"),
            ("deploy", "deploy_camera_every2_noise1e-3", "camera on every 2nd pendulum, 10⁻³ rad"),
            ("deploy", "deploy_camera_noise1e-2_t20", "coarse camera (10⁻² rad) from τ = 20"),
            ("deploy", "deploy_energy_t30", "energies at τ = 30, one frame")]
    labels = [("v1label", "v1's label (first pendulum over 10%)", SLOT[0]), ("avg", "swing-averaged label", SLOT[2])]
    data = [(group, lab, {key: recs.get((group, f"{name}_{key}"), []) for key, _, _ in labels}) for group, name, lab in rows]
    data = [(group, lab, d) for group, lab, d in data if all(d.values())]
    if len(data) < 2 or data[0][1] != "v2 final recipe":
        raise FileNotFoundError("phase 7 runs not finished yet")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 0.36 * len(data) + 1.4), sharey=True, layout="constrained")
    y = np.arange(len(data))[::-1]
    for ax, metric, scale in ((axes[0], "mae", 1), (axes[1], "acc_1", 100)):
        means = {key: [scale * np.mean([r["test"][metric] for r in d[key]]) for _, _, d in data] for key, _, _ in labels}
        for yi, a, b in zip(y, *means.values()):
            ax.plot([a, b], [yi, yi], color=GRID, linewidth=2.0, zorder=1)
        for key, name, color in labels:
            ax.axvline(means[key][0], color=color, linewidth=0.8, alpha=0.5, zorder=0)  # the final recipe on this label
            for yi, (_, _, d) in zip(y, data):
                v = [scale * r["test"][metric] for r in d[key]]
                ax.plot(v, [yi] * len(v), linestyle="none", marker="o", markersize=3.5, color=color, alpha=0.4, zorder=2)
            ax.plot(means[key], y, linestyle="none", marker="o", markersize=7, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name, zorder=3)
        for yi, (prev, _, _), (group, _, _) in zip(y[1:], data[:-1], data[1:]):
            if group != prev or yi == y[1]:  # separate the final recipe and each group of choices
                ax.axhline(yi + 0.5, color=GRID, linewidth=0.6)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [lab for _, lab, _ in data])
    axes[0].set_xlim(0, None)
    axes[1].set_xlim(None, 100)
    axes[0].set_xlabel("Mean distance on test (sites); small dots are seeds, lines the final recipe")
    axes[1].set_xlabel("Within ±1 site (%)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=2, fontsize=8)
    axes[0].set_title("(a) Mean distance", loc="left", fontsize=9)
    axes[1].set_title("(b) Within ±1", loc="left", fontsize=9)
    fig.suptitle("The same choices win on the swing-averaged label; only soft labels stop mattering (3 seeds each)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    save(fig, "fig19_avg_label_ablations")


ROBUST_LABELS = [("pool_test", "nominal test pool"), ("robust_g-10", "neighbour coupling g −10%"),
                 ("robust_g+10", "neighbour coupling g +10%"), ("robust_h-10", "global coupling h −10%"),
                 ("robust_h+10", "global coupling h +10%"), ("robust_noise2x", "twice the initial spread"),
                 ("robust_N80", "80 pendulums"), ("robust_N150", "150 pendulums"),
                 ("robust_impurity1", "one pendulum 1% weaker (2 mm longer)"),
                 ("test_noise0.0003", "sensor noise 3e-4 rad"), ("test_noise0.001", "sensor noise 1e-3 rad"),
                 ("test_noise0.003", "sensor noise 3e-3 rad")]


def fig_robustness():
    from sgloc import ablations, experiment
    path = ROOT / "results" / "robustness.jsonl"
    if not path.exists():
        raise FileNotFoundError("no robustness results yet")
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "mae" in r["test"]:  # strongly damped pools can have no localizing runs, hence no metrics
                latest[(r["run_dir"], r["condition"])] = r["test"]["mae"]
    groups = {"v2 final": [f"runs/v2_final_s{c['seed']}_{experiment.config_id(c)}" for c in ablations.phase6()
                           if c["name"] == "v2_final"],
              "v1 rebuilt": [f"runs/{r['run_id']}" for r in load_runs() if r.get("model") == "v1_cnn_bilstm" and r.get("label", "right") == "right"]}
    rows = []
    for cond, label in ROBUST_LABELS:
        means = {g: np.mean([latest[(d, cond)] for d in dirs if (d, cond) in latest]) if any((d, cond) in latest for d in dirs)
                 else None for g, dirs in groups.items()}
        if means["v2 final"] is not None:
            rows.append((label, means))
    if not rows:
        raise FileNotFoundError("no v2 robustness results yet")
    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (label, means) in zip(y, rows):
        vals = [v for v in means.values() if v is not None]
        if len(vals) == 2:
            ax.plot(vals, [yi, yi], color=GRID, linewidth=2.0, zorder=1)
    for slot, group in enumerate(groups):
        pts = [(means[group], yi) for yi, (_, means) in zip(y, rows) if means[group] is not None]
        if pts:
            ax.plot(*zip(*pts), linestyle="none", marker="o", markersize=7, color=SLOT[slot], markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=group, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, [label for label, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean distance on test (sites, log scale); no retraining. v1 cannot take other chain lengths")
    ax.legend(loc="upper right")
    ax.set_title("Trained on the nominal chain, tested on shifted ones", loc="left")
    save(fig, "fig16_robustness")


def paper_figure(name, height, widths, draw):
    """One full-width preprint figure: a row of panels drawn by draw(axes), each marked only by its letter."""
    fig, axes = plt.subplots(1, len(widths), figsize=(PAPER_WIDTH, height), gridspec_kw={"width_ratios": widths},
                             layout="constrained")
    draw(axes)
    letters = [ax.set_title(f"({s})", loc="left", fontsize=9, fontweight="bold") for ax, s in zip(axes, "abc")]
    fig.canvas.draw()  # then line each letter up with the left edge of its panel's axis label
    for ax, letter in zip(axes, letters):
        letter.set_x(ax.transAxes.inverted().transform(ax.yaxis.get_tightbbox().p0)[0])
    PAPER.mkdir(exist_ok=True)
    # no date: a rerun leaves the file unchanged; dpi only for the colorbar, which matplotlib rasterizes
    fig.savefig(PAPER / f"{name}.pdf", dpi=600, metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", f"papers/v2_preprint/figures/{name}.pdf")


def paper_figures():
    """The preprint's five data figures, numbered as in the preprint (its other figures are made elsewhere)."""
    plt.rcParams.update(PAPER_RC)
    paper_figure("paper_fig1_spectrum", 2.5, [1.25, 1], fig_growth)
    paper_figure("paper_fig2_physics", 2.4, [1, 1.1, 0.9], lambda ax: (fig_best_width(ax[0]), fig_sensitivity(ax[1:])))
    paper_figure("paper_fig4_ring", 2.4, [1, 1], lambda ax: (fig_receptive_field(ax[0]), fig_data_efficiency(ax[1])))
    paper_figure("paper_fig5_jitter", 2.4, [1, 1.4], fig_label_jitter)
    paper_figure("paper_fig6_noise", 2.6, [1, 1], lambda ax: (fig_noise_collapse(ax[:1]), fig_noise_avg_label(ax[1])))


if __name__ == "__main__":
    if "--paper" in sys.argv[1:]:
        paper_figures()
        sys.exit()
    for fn in (fig_growth, fig_best_width, fig_sensitivity, fig_baselines, fig_receptive_field, fig_data_efficiency,
               fig_architectures, fig_noise_time, fig_noise_collapse, fig_noise_avg_label, fig_inputs, fig_losses,
               fig_optimizers, fig_sparse, fig_leave_one_out, fig_final_data, fig_robustness, fig_warning_tradeoff,
               fig_label_jitter, fig_avg_label_ablations, fig_regularization):
        try:
            fn()
        except FileNotFoundError as err:
            print(f"skipped {fn.__name__}: {err}")
