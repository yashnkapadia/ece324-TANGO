"""Generate report-quality figures for ASCE final report.

White background, print-friendly colors, PDF output for LaTeX.

Usage:
    pixi run python scripts/generate_report_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Report theme (print-friendly) ────────────────────────────────────────────
C_MAPPO = "#2166ac"  # Blue
C_MP = "#b2182b"  # Red
C_FT = "#999999"  # Grey
C_BEST = "#2166ac"  # Blue marker

SCENARIO_COLORS = {
    "am_peak": "#2166ac",
    "pm_peak": "#b2182b",
    "demand_surge": "#d6604d",
    "midday_multimodal": "#4393c3",
}

SCENARIO_LABELS = {
    "am_peak": "AM Peak",
    "pm_peak": "PM Peak",
    "demand_surge": "Demand Surge",
    "midday_multimodal": "Midday Multimodal",
}

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ_ROOT / "reports" / "results"
FIGURES_DIR = PROJ_ROOT / "reports" / "final"


def _apply_theme(ax: plt.Axes) -> None:
    """Apply clean report theme to axes."""
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=9)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#cccccc")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5)


def generate_bar_chart(out_path: Path) -> None:
    """Grouped bar chart: MAPPO vs MP vs Actuated Default vs FT, 4 scenarios."""
    import csv
    import statistics

    eval_dir = RESULTS_DIR / "eval_matrix"
    scenarios = ["am_peak", "pm_peak", "demand_surge", "midday_multimodal"]
    model = "asce_mappo_curriculum_best"

    data = {"mappo": ([], []), "max_pressure": ([], []), "fixed_time": ([], []), "nema": ([], [])}

    for scen in scenarios:
        # MAPPO, MP, FT from main eval
        fpath = eval_dir / f"{model}__{scen}.csv"
        with open(fpath) as f:
            rows = list(csv.DictReader(f))
        for ctrl in ["mappo", "max_pressure", "fixed_time"]:
            vals = [float(r["person_time_loss_s"]) for r in rows if r["controller"] == ctrl]
            data[ctrl][0].append(statistics.mean(vals))
            data[ctrl][1].append(statistics.stdev(vals) if len(vals) > 1 else 0)

        # Actuated Default from nema eval
        nema_path = eval_dir / f"nema__{scen}.csv"
        with open(nema_path) as f:
            nema_rows = list(csv.DictReader(f))
        nema_vals = [float(r["person_time_loss_s"]) for r in nema_rows]
        data["nema"][0].append(statistics.mean(nema_vals))
        data["nema"][1].append(statistics.stdev(nema_vals) if len(nema_vals) > 1 else 0)

    x = np.arange(len(scenarios))
    width = 0.19
    C_ACT = "#e6850e"  # Orange for Actuated Default

    fig, ax = plt.subplots(figsize=(7, 3.8))
    _apply_theme(ax)

    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    controllers = [
        ("mappo", "MAPPO (best)", C_MAPPO, 0.85),
        ("max_pressure", "Max-Pressure", C_MP, 0.7),
        ("nema", "Actuated Default", C_ACT, 0.7),
        ("fixed_time", "Fixed-Time", C_FT, 0.5),
    ]

    for (ctrl, label, color, alpha), offset in zip(controllers, offsets):
        means, stds = data[ctrl]
        ax.bar(
            x + offset,
            [v / 1000 for v in means],
            width,
            yerr=[s / 1000 for s in stds],
            capsize=2,
            label=label,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.5,
        )

    # Ratio annotations on MAPPO bars
    mappo_means, mappo_stds = data["mappo"]
    mp_means = data["max_pressure"][0]
    for i, (m, mp) in enumerate(zip(mappo_means, mp_means)):
        pct = (1 - m / mp) * 100
        ax.text(
            x[i] + offsets[0],
            m / 1000 + max(mappo_stds[i], 1) / 1000 + 5,
            f"$-${pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=C_MAPPO,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8.5)
    ax.set_ylabel("Person-Time-Loss (×1000 s)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    all_means = [v for ctrl in data.values() for v in ctrl[0]]
    ax.set_ylim(0, max(all_means) / 1000 * 1.18)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved: {out_path}")


def generate_training_curve(out_path: Path) -> None:
    """Training curve: warm-start + curriculum, white background."""
    warmstart_path = RESULTS_DIR / "asce_train_episode_metrics_person_obj.csv"
    curriculum_path = RESULTS_DIR / "curriculum_v1_reconstructed_metrics.csv"

    if not curriculum_path.exists():
        print(f"  SKIP training curve — {curriculum_path} not found")
        return

    df_cur = pd.read_csv(curriculum_path)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_theme(ax)

    window = 8

    # Phase 1: warm-start
    if warmstart_path.exists():
        df_ws = pd.read_csv(warmstart_path).sort_values("episode")
        ax.scatter(
            df_ws["episode"],
            df_ws["mean_global_reward"],
            color="#aaaaaa",
            alpha=0.15,
            s=4,
            zorder=0,
        )
        smoothed_ws = df_ws["mean_global_reward"].rolling(window, min_periods=1).mean()
        ax.plot(
            df_ws["episode"].values,
            smoothed_ws.values,
            color="#888888",
            linewidth=1.5,
            alpha=0.7,
            label="Warm-start (AM only)",
            linestyle="--",
            zorder=1,
        )

    # Boundary
    ax.axvline(x=200, color="#cccccc", linewidth=1, linestyle=":", zorder=3)
    ax.text(
        205,
        ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1.5,
        "Curriculum\nstarts",
        color="#666666",
        fontsize=7,
        va="bottom",
    )

    # Phase 2: curriculum
    for scenario_id, color in SCENARIO_COLORS.items():
        subset = df_cur[df_cur["scenario_id"] == scenario_id].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("episode")
        x_vals = subset["episode"].values + 200

        ax.scatter(x_vals, subset["mean_global_reward"], color=color, alpha=0.12, s=5, zorder=1)
        smoothed = subset["mean_global_reward"].rolling(window, min_periods=1).mean()
        ax.plot(
            x_vals,
            smoothed.values,
            color=color,
            linewidth=1.8,
            alpha=0.9,
            label=SCENARIO_LABELS[scenario_id],
            zorder=2,
        )

    # Best checkpoint
    best_total = 432 + 200
    ax.axvline(x=best_total, color="#d6604d", linewidth=1, linestyle="--", alpha=0.7, zorder=3)
    ax.text(
        best_total + 5,
        ax.get_ylim()[1] * 0.95,
        "Best (ep 632)",
        color="#d6604d",
        fontsize=8,
        fontweight="bold",
        va="top",
    )

    ax.set_xlim(0, 800)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Mean Global Reward", fontsize=10)
    ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curve saved: {out_path}")


def generate_gate_fraction(out_path: Path) -> None:
    """Gate fraction: warm-start + curriculum."""
    warmstart_path = RESULTS_DIR / "asce_train_episode_metrics_person_obj.csv"
    curriculum_path = RESULTS_DIR / "curriculum_v1_reconstructed_metrics.csv"

    if not curriculum_path.exists():
        print(f"  SKIP gate fraction — {curriculum_path} not found")
        return

    df_cur = pd.read_csv(curriculum_path)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    _apply_theme(ax)

    window = 16

    # Phase 1: warm-start
    if warmstart_path.exists():
        df_ws = pd.read_csv(warmstart_path).sort_values("episode")
        if "gate_fraction" in df_ws.columns:
            ax.scatter(
                df_ws["episode"],
                df_ws["gate_fraction"] * 100,
                color="#aaaaaa",
                alpha=0.15,
                s=4,
                zorder=0,
            )
            sm_ws = df_ws["gate_fraction"].rolling(window, min_periods=1).mean() * 100
            ax.plot(
                df_ws["episode"].values,
                sm_ws.values,
                color="#888888",
                linewidth=1.5,
                alpha=0.7,
                linestyle="--",
                zorder=1,
            )

    ax.axvline(x=200, color="#cccccc", linewidth=1, linestyle=":", alpha=0.5)

    # Phase 2: curriculum
    df_sorted = df_cur.sort_values("episode")
    x_vals = df_sorted["episode"].values + 200
    ax.scatter(x_vals, df_sorted["gate_fraction"] * 100, color=C_MAPPO, alpha=0.15, s=4, zorder=1)
    smoothed = df_sorted["gate_fraction"].rolling(window, min_periods=1).mean() * 100
    ax.plot(x_vals, smoothed.values, color=C_MAPPO, linewidth=2, alpha=0.9, zorder=2)

    # Cold init reference
    ax.axhline(y=2, color="#cccccc", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(5, 3.5, "Init ~2%", color="#888888", fontsize=7, va="bottom")

    # Best checkpoint
    ax.axvline(x=632, color="#d6604d", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlim(0, 800)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Gate Override (%)", fontsize=10)
    ax.set_ylim(0, 45)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gate fraction saved: {out_path}")


def generate_architecture_diagram(out_path: Path) -> None:
    """Generate ASCE architecture diagram — clean, professional, print-ready."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Colors
    C_INPUT = "#dce8f5"  # Light blue fill
    C_INPUT_E = "#4393c3"  # Blue edge
    C_PROC = "#f5f5f5"  # Light grey fill
    C_PROC_E = "#aaaaaa"  # Grey edge
    C_MLP = "#fff3e0"  # Warm yellow fill
    C_MLP_E = "#c75b39"  # Warm edge
    C_GATE = "#fbe9e7"  # Light red fill
    C_GATE_E = "#c75b39"  # Red edge
    C_PHASE = "#e3f2e8"  # Light green fill
    C_PHASE_E = "#388e5e"  # Green edge
    C_DECISION = "#f5f5f5"  # Light fill
    C_DECISION_E = "#333333"
    C_CRITIC = "#dce8f5"
    C_CRITIC_E = "#4393c3"
    C_ARROW = "#555555"
    C_LABEL = "#333333"

    def _box(x, y, w, h, fc, ec, lw=1.2):
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.3",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=ax.transData,
            zorder=2,
        )
        ax.add_patch(p)

    def _arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.0):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
            zorder=3,
        )

    # ── Track labels ──
    ax.text(
        50,
        54,
        "ACTOR  (per-agent, decentralized)",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="#555555",
        family="monospace",
    )

    # ── Input boxes ──
    _box(10, 42, 16, 9, C_INPUT, C_INPUT_E)
    ax.text(
        10, 44, "Local Obs", ha="center", va="center", fontsize=8, fontweight="bold", color=C_LABEL
    )
    ax.text(
        10,
        40.5,
        "queue, density,\nphase, min-green",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#555555",
    )

    _box(10, 30, 16, 6.5, C_PROC, C_PROC_E)
    ax.text(
        10,
        31.5,
        "MP Action",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(10, 28.5, "one-hot  |  6 dim", ha="center", va="center", fontsize=6.5, color="#555555")

    # ── Concat ──
    _arrow(18, 42, 25, 38)
    _arrow(18, 30, 25, 35)
    _box(28, 36.5, 8, 5, C_PROC, C_PROC_E)
    ax.text(
        28, 36.5, "Concat", ha="center", va="center", fontsize=7, fontweight="bold", color=C_LABEL
    )

    # ── Welford ──
    _arrow(32, 36.5, 37, 36.5)
    _box(42, 36.5, 10, 5.5, C_PROC, C_PROC_E)
    ax.text(
        42, 38, "Welford", ha="center", va="center", fontsize=7, fontweight="bold", color=C_LABEL
    )
    ax.text(42, 35.5, "$\\mu / \\sigma$", ha="center", va="center", fontsize=7, color="#555555")

    # ── Shared MLP ──
    _arrow(47, 36.5, 52, 36.5)
    _box(60, 36.5, 14, 8, C_MLP, C_MLP_E, lw=1.8)
    ax.text(
        60,
        39,
        "Shared MLP",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(60, 36.5, "128→ReLU→128", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(
        60,
        34,
        "×12 agents",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#888888",
        fontstyle="italic",
    )

    # ── Gate Head ──
    _arrow(67, 39.5, 74, 46)
    _box(80, 46, 13, 7, C_GATE, C_GATE_E, lw=1.5)
    ax.text(
        80,
        48,
        "Gate Head",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(80, 45.5, "follow / override", ha="center", va="center", fontsize=6.5, color="#555555")
    ax.text(
        80,
        43.5,
        "init: 98% follow",
        ha="center",
        va="center",
        fontsize=6,
        color="#888888",
        fontstyle="italic",
    )

    # ── Phase Head ──
    _arrow(67, 34, 74, 28)
    _box(80, 28, 13, 7, C_PHASE, C_PHASE_E, lw=1.5)
    ax.text(
        80,
        30,
        "Phase Head",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(80, 27.5, "6 phases (masked)", ha="center", va="center", fontsize=6.5, color="#555555")
    ax.text(
        80,
        25.5,
        "softmax over valid",
        ha="center",
        va="center",
        fontsize=6,
        color="#888888",
        fontstyle="italic",
    )

    # ── Decision ──
    _arrow(87, 46, 93, 40)
    _arrow(87, 28, 93, 34)
    _box(96, 37, 7, 10, C_DECISION, C_DECISION_E, lw=1.8)
    ax.text(
        96,
        39.5,
        "Action",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(
        96, 37, "g=0→MP", ha="center", va="center", fontsize=6, color=C_GATE_E, fontweight="bold"
    )
    ax.text(
        96,
        35,
        "g=1→$\\pi$",
        ha="center",
        va="center",
        fontsize=6,
        color=C_PHASE_E,
        fontweight="bold",
    )

    # ── CTDE separator ──
    ax.plot([0, 100], [20, 20], color="#cccccc", linewidth=1, linestyle="--", zorder=1)
    ax.text(
        50,
        21,
        "CTDE boundary — critic used during training only",
        ha="center",
        fontsize=6.5,
        color="#999999",
        fontstyle="italic",
    )

    # ── Critic track (bottom) ──
    ax.text(
        50,
        17.5,
        "CRITIC  (centralized, training only)",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="#555555",
        family="monospace",
    )

    _box(15, 10, 18, 6, C_CRITIC, C_CRITIC_E)
    ax.text(
        15,
        11.5,
        "Global Obs",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(
        15, 9, "all 12 agents  |  ~460d", ha="center", va="center", fontsize=6.5, color="#555555"
    )

    _arrow(24, 10, 35, 10)

    _box(48, 10, 22, 6, C_CRITIC, C_CRITIC_E, lw=1.5)
    ax.text(
        48,
        11.5,
        "Centralized Critic",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=C_LABEL,
    )
    ax.text(48, 9, "128→ReLU→128→ReLU→1", ha="center", va="center", fontsize=6.5, color="#555555")

    _arrow(59, 10, 68, 10)

    _box(74, 10, 10, 5.5, C_PROC, C_CRITIC_E)
    ax.text(74, 11, "V(s)", ha="center", va="center", fontsize=8, fontweight="bold", color=C_LABEL)
    ax.text(74, 8.5, "scalar value", ha="center", va="center", fontsize=6, color="#555555")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Architecture diagram saved: {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating ASCE report figures...")
    generate_bar_chart(FIGURES_DIR / "asce_bar_chart.pdf")
    generate_training_curve(FIGURES_DIR / "asce_training_curve.pdf")
    generate_gate_fraction(FIGURES_DIR / "asce_gate_fraction.pdf")
    generate_architecture_diagram(FIGURES_DIR / "asce_architecture.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
