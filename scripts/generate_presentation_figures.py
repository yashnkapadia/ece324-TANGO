"""Generate presentation figures for ASCE slides.

Produces two figures matching the TANGO presentation dark theme:
1. Grouped bar chart: MAPPO vs Max-Pressure person-time-loss across 4 scenarios
2. Training curve: mean reward per episode colored by scenario (curriculum run)

Usage:
    pixi run python scripts/generate_presentation_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Theme colors (from TANGO-presentation.html CSS variables) ─────────────────
BG_0 = "#0b1118"
BG_1 = "#121b26"
BG_2 = "#1a2633"
INK_HIGH = "#f9fbff"
INK_MID = "#d6dfea"
INK_SOFT = "#bcc8d6"
LANE = "#d9bb6d"
SIGNAL_RED = "#df7a72"
SIGNAL_AMBER = "#d8b36a"
SIGNAL_GREEN = "#6eaf8b"
ACCENT_BLUE = "#88a7c9"
LINE_COLOR = "rgba(230,237,248,0.2)"

# Matplotlib-compatible line color
LINE_GREY = "#3a4555"

SCENARIO_COLORS = {
    "am_peak": SIGNAL_GREEN,
    "pm_peak": SIGNAL_RED,
    "demand_surge": SIGNAL_AMBER,
    "midday_multimodal": ACCENT_BLUE,
}

SCENARIO_LABELS = {
    "am_peak": "AM Peak",
    "pm_peak": "PM Peak",
    "demand_surge": "Demand Surge",
    "midday_multimodal": "Midday Multimodal",
}

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ_ROOT / "reports" / "results"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"


def _apply_theme(ax: plt.Axes) -> None:
    """Apply dark presentation theme to axes."""
    ax.set_facecolor(BG_1)
    ax.figure.set_facecolor(BG_0)
    ax.tick_params(colors=INK_MID, labelsize=11)
    ax.xaxis.label.set_color(INK_MID)
    ax.yaxis.label.set_color(INK_MID)
    ax.title.set_color(INK_HIGH)
    for spine in ax.spines.values():
        spine.set_color(LINE_GREY)
    ax.grid(axis="y", color=LINE_GREY, alpha=0.4, linewidth=0.5)


def generate_bar_chart(out_path: Path) -> None:
    """Grouped bar chart: MAPPO vs MP person-time-loss, 4 scenarios (ep 432 eval)."""
    # Verified eval data from reconstructed_curriculum_v1_run.log, ep 432
    scenarios = ["am_peak", "pm_peak", "demand_surge", "midday_multimodal"]
    mappo_ptl = [384_562, 469_994, 437_357, 464_394]
    mp_ptl = [431_590, 537_140, 502_485, 541_112]
    ratios = [m / b for m, b in zip(mappo_ptl, mp_ptl)]

    x = np.arange(len(scenarios))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _apply_theme(ax)

    # MP bars (muted)
    ax.bar(
        x + width / 2,
        [v / 1000 for v in mp_ptl],
        width,
        label="Max-Pressure",
        color=INK_SOFT,
        alpha=0.45,
        edgecolor=LINE_GREY,
        linewidth=0.8,
    )

    # MAPPO bars (colored per scenario)
    bars_mappo = []
    for i, (sc, val) in enumerate(zip(scenarios, mappo_ptl)):
        bar = ax.bar(
            x[i] - width / 2,
            val / 1000,
            width,
            color=SCENARIO_COLORS[sc],
            edgecolor=LINE_GREY,
            linewidth=0.8,
            alpha=0.88,
        )
        bars_mappo.append(bar)

    # Ratio annotations above MAPPO bars
    for i, (ratio, val) in enumerate(zip(ratios, mappo_ptl)):
        pct = (1 - ratio) * 100
        ax.text(
            x[i] - width / 2,
            val / 1000 + 8,
            f"-{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=SIGNAL_GREEN,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=12, fontweight="600")
    ax.set_ylabel("Person-Time-Loss (×1000 s)", fontsize=12, fontweight="600")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=SIGNAL_GREEN, alpha=0.88, edgecolor=LINE_GREY, label="MAPPO"),
        Patch(facecolor=INK_SOFT, alpha=0.45, edgecolor=LINE_GREY, label="Max-Pressure"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=11,
        facecolor=BG_2,
        edgecolor=LINE_GREY,
        labelcolor=INK_HIGH,
    )

    ax.set_ylim(0, max(mp_ptl) / 1000 * 1.18)
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=200, facecolor=BG_0, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved: {out_path}")


def generate_training_curve(out_path: Path) -> None:
    """Training curve: 200 ep AM-only warm-start + 600 ep curriculum (total 800)."""
    warmstart_path = RESULTS_DIR / "asce_train_episode_metrics_person_obj.csv"
    curriculum_path = RESULTS_DIR / "curriculum_v1_reconstructed_metrics.csv"

    if not curriculum_path.exists():
        print(f"  SKIP training curve — {curriculum_path} not found")
        return

    df_cur = pd.read_csv(curriculum_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_theme(ax)

    window = 8

    # Phase 1: warm-start (person_obj, AM peak only, ep 0-199)
    if warmstart_path.exists():
        df_ws = pd.read_csv(warmstart_path).sort_values("episode")
        ax.scatter(
            df_ws["episode"],
            df_ws["mean_global_reward"],
            color=INK_SOFT,
            alpha=0.12,
            s=6,
            zorder=0,
        )
        smoothed_ws = df_ws["mean_global_reward"].rolling(window, min_periods=1).mean()
        ax.plot(
            df_ws["episode"].values,
            smoothed_ws.values,
            color=INK_SOFT,
            linewidth=1.8,
            alpha=0.6,
            label="Warm-start (AM only)",
            linestyle="--",
            zorder=1,
        )

    # Boundary marker
    ax.axvline(x=200, color=INK_SOFT, linewidth=1, linestyle=":", alpha=0.5, zorder=3)
    ax.text(
        203,
        -1.25,
        "Curriculum\nstarts",
        color=INK_SOFT,
        fontsize=8,
        fontweight="500",
        va="bottom",
    )

    # Phase 2: curriculum (ep 0-599, plotted as ep 200-799)
    for scenario_id, color in SCENARIO_COLORS.items():
        subset = df_cur[df_cur["scenario_id"] == scenario_id].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("episode")
        # Offset curriculum episodes by 200 to show true timeline
        x_vals = subset["episode"].values + 200

        ax.scatter(x_vals, subset["mean_global_reward"], color=color, alpha=0.15, s=8, zorder=1)

        smoothed = subset["mean_global_reward"].rolling(window, min_periods=1).mean()
        ax.plot(
            x_vals,
            smoothed.values,
            color=color,
            linewidth=2.2,
            alpha=0.9,
            label=SCENARIO_LABELS[scenario_id],
            zorder=2,
        )

    # Mark best checkpoint (curriculum ep 432 = total ep 632)
    best_total = 432 + 200
    ax.axvline(x=best_total, color=LANE, linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)
    ax.text(
        best_total + 5,
        ax.get_ylim()[1] * 0.97,
        "Best (ep 632)",
        color=LANE,
        fontsize=10,
        fontweight="600",
        va="top",
    )

    ax.set_xlim(0, 800)
    ax.set_xlabel("Episode", fontsize=12, fontweight="600")
    ax.set_ylabel("Mean Global Reward", fontsize=12, fontweight="600")
    ax.legend(
        loc="lower right",
        fontsize=9,
        facecolor=BG_2,
        edgecolor=LINE_GREY,
        labelcolor=INK_HIGH,
        ncol=2,
    )

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=200, facecolor=BG_0, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curve saved: {out_path}")


def generate_gate_fraction_curve(out_path: Path) -> None:
    """Gate fraction: warm-start + curriculum, showing learned override."""
    warmstart_path = RESULTS_DIR / "asce_train_episode_metrics_person_obj.csv"
    curriculum_path = RESULTS_DIR / "curriculum_v1_reconstructed_metrics.csv"

    if not curriculum_path.exists():
        print(f"  SKIP gate fraction — {curriculum_path} not found")
        return

    df_cur = pd.read_csv(curriculum_path)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    _apply_theme(ax)

    window = 16

    # Phase 1: warm-start gate fraction
    if warmstart_path.exists():
        df_ws = pd.read_csv(warmstart_path).sort_values("episode")
        if "gate_fraction" in df_ws.columns:
            ax.scatter(
                df_ws["episode"],
                df_ws["gate_fraction"] * 100,
                color=INK_SOFT,
                alpha=0.15,
                s=5,
                zorder=0,
            )
            sm_ws = df_ws["gate_fraction"].rolling(window, min_periods=1).mean() * 100
            ax.plot(
                df_ws["episode"].values,
                sm_ws.values,
                color=INK_SOFT,
                linewidth=1.8,
                alpha=0.6,
                linestyle="--",
                zorder=1,
            )

    # Boundary
    ax.axvline(x=200, color=INK_SOFT, linewidth=1, linestyle=":", alpha=0.4)

    # Phase 2: curriculum gate fraction (offset by 200)
    df_sorted = df_cur.sort_values("episode")
    x_vals = df_sorted["episode"].values + 200

    ax.scatter(x_vals, df_sorted["gate_fraction"] * 100, color=LANE, alpha=0.2, s=6, zorder=1)

    smoothed = df_sorted["gate_fraction"].rolling(window, min_periods=1).mean() * 100
    ax.plot(x_vals, smoothed.values, color=LANE, linewidth=2.5, alpha=0.9, zorder=2)

    # Reference line for cold-start init (~2%)
    ax.axhline(y=2, color=INK_SOFT, linewidth=1, linestyle=":", alpha=0.5)
    ax.text(5, 3.5, "Cold init ~2%", color=INK_SOFT, fontsize=9, va="bottom")

    # Best checkpoint (curriculum ep 432 = total ep 632)
    ax.axvline(x=632, color=SIGNAL_GREEN, linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xlim(0, 800)
    ax.set_xlabel("Episode", fontsize=11, fontweight="600")
    ax.set_ylabel("Gate Override %", fontsize=11, fontweight="600")
    ax.set_ylim(0, 45)

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=200, facecolor=BG_0, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gate fraction saved: {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating ASCE presentation figures...")
    generate_bar_chart(FIGURES_DIR / "asce_bar_chart.png")
    generate_training_curve(FIGURES_DIR / "asce_training_curve.png")
    generate_gate_fraction_curve(FIGURES_DIR / "asce_gate_fraction.png")
    print("Done.")


if __name__ == "__main__":
    main()
