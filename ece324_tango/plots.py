"""Generate interim report figures from ASCE objective-run artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
RESULTS_DIR = REPO_ROOT / "reports" / "results"

TRAIN_CSV = RESULTS_DIR / "asce_train_episode_metrics_toronto_demand.csv"
EVAL_CSV = (
    RESULTS_DIR / "asce_eval_metrics_toronto_demand_objective_retest_e10.csv"
)

OUTPUT_STEM = "interim_objective_results"


# ── Data loading ───────────────────────────────────────────────────────────
_TRAIN_REQUIRED = {"episode", "mean_global_reward"}
_EVAL_REQUIRED = {
    "controller",
    "person_time_loss_s",
    "avg_trip_time_s",
    "arrived_vehicles",
    "vehicle_delay_jain",
}


def _load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df


# ── Aggregate helpers ──────────────────────────────────────────────────────
_CONTROLLER_ORDER = ["mappo", "fixed_time", "max_pressure"]
_CONTROLLER_LABELS = {"mappo": "MAPPO", "fixed_time": "Fixed-Time", "max_pressure": "Max-Pressure"}
_CONTROLLER_COLORS = {"mappo": "#4C72B0", "fixed_time": "#DD8452", "max_pressure": "#55A868"}


def _controller_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-controller mean and std for key eval metrics."""
    metrics = ["person_time_loss_s", "avg_trip_time_s", "arrived_vehicles", "vehicle_delay_jain"]
    grouped = eval_df.groupby("controller")[metrics]
    summary = grouped.agg(["mean", "std"]).reindex(_CONTROLLER_ORDER)
    return summary


# ── Figure rendering ───────────────────────────────────────────────────────
def render_figure(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Figure:
    """Create the 2x2 composite interim-report figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        "TANGO Interim Results — Toronto (10 seeds)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    _panel_a_training_curve(axes[0, 0], train_df)
    _panel_b_time_loss(axes[0, 1], eval_df)
    _panel_c_eval_metrics(axes[1, 0], eval_df)
    _panel_d_context(axes[1, 1], eval_df)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _panel_a_training_curve(ax: plt.Axes, train_df: pd.DataFrame) -> None:
    """Panel A: MAPPO objective training progress."""
    ax.plot(
        train_df["episode"],
        train_df["mean_global_reward"],
        color=_CONTROLLER_COLORS["mappo"],
        linewidth=1.5,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Global Reward")
    ax.set_title("A) MAPPO Training Progress", fontsize=10, fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)


def _panel_b_time_loss(ax: plt.Axes, eval_df: pd.DataFrame) -> None:
    """Panel B: per-controller person_time_loss_s distribution."""
    data = [
        eval_df.loc[eval_df["controller"] == c, "person_time_loss_s"].values
        for c in _CONTROLLER_ORDER
    ]
    labels = [_CONTROLLER_LABELS[c] for c in _CONTROLLER_ORDER]
    colors = [_CONTROLLER_COLORS[c] for c in _CONTROLLER_ORDER]

    bplot = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Person Time Loss (s)")
    ax.set_title(
        "B) Person Time Loss by Controller", fontsize=10, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)


def _panel_c_eval_metrics(ax: plt.Axes, eval_df: pd.DataFrame) -> None:
    """Panel C: metrics normalized to Max-Pressure baseline (ratio = 1.0)."""
    import numpy as np

    summary = _controller_summary(eval_df)
    metrics = [
        ("avg_trip_time_s", "Avg Trip\nTime"),
        ("arrived_vehicles", "Arrived\nVehicles"),
        ("vehicle_delay_jain", "Vehicle Delay\nFairness (Jain)"),
    ]

    # Normalize each metric to the max_pressure mean
    mp_means = {m: summary.loc["max_pressure", (m, "mean")] for m, _ in metrics}

    x = np.arange(len(metrics))
    width = 0.22
    offsets = np.arange(len(_CONTROLLER_ORDER)) - 1

    for i, ctrl in enumerate(_CONTROLLER_ORDER):
        ratios = [
            summary.loc[ctrl, (m, "mean")] / mp_means[m] for m, _ in metrics
        ]
        ax.bar(
            x + offsets[i] * width,
            ratios,
            width,
            label=_CONTROLLER_LABELS[ctrl],
            color=_CONTROLLER_COLORS[ctrl],
            alpha=0.8,
        )

    ax.axhline(1.0, color="grey", linewidth=1, linestyle="--", label="_nolegend_")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=8)
    ax.set_ylabel("Ratio vs Max-Pressure")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(
        "C) Eval Metrics (normalized to Max-Pressure)",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)


def _panel_d_context(ax: plt.Axes, eval_df: pd.DataFrame) -> None:
    """Panel D: environment facts and takeaway context card."""
    ax.set_axis_off()

    # Compute summary stats for the context card
    means = eval_df.groupby("controller")["person_time_loss_s"].mean()
    mp_mean = means["max_pressure"]
    mappo_mean = means["mappo"]
    pct_worse = (mappo_mean - mp_mean) / mp_mean * 100

    lines = [
        ("Environment", [
            "Toronto OSM corridor, 12 signalized intersections",
            "70 car flows, all active 0–3600 s",
            "Single-mode (car only), nominal demand",
        ]),
        ("Evaluation", [
            "10 seeds per controller",
            "Controllers: MAPPO (30-ep), Fixed-Time, Max-Pressure",
            f"Simulation: 300 s episodes, Δt = 5 s",
        ]),
        ("Key Takeaway", [
            f"MAPPO person-time-loss is {pct_worse:.0f}% higher than Max-Pressure",
            "Max-Pressure dominates in this nominal single-mode regime",
            "Expected - MAPPO needs harder scenarios to show value",
        ]),
    ]

    y = 0.95
    for heading, bullets in lines:
        ax.text(0.05, y, heading, fontsize=9, fontweight="bold",
                transform=ax.transAxes, verticalalignment="top")
        y -= 0.07
        for bullet in bullets:
            ax.text(0.08, y, f"• {bullet}", fontsize=7.5,
                    transform=ax.transAxes, verticalalignment="top")
            y -= 0.07
        y -= 0.04

    ax.set_title("D) Context", fontsize=10, fontweight="bold")


# ── Save helpers ───────────────────────────────────────────────────────────
def save_figure(fig: Figure) -> list[Path]:
    """Write PNG and PDF to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = FIGURES_DIR / f"{OUTPUT_STEM}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    return paths


# ── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    train_df = _load_csv(TRAIN_CSV, _TRAIN_REQUIRED)
    eval_df = _load_csv(EVAL_CSV, _EVAL_REQUIRED)

    fig = render_figure(train_df, eval_df)
    paths = save_figure(fig)
    plt.close(fig)

    for p in paths:
        print(f"Wrote: {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
