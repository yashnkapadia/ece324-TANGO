"""Compute paired t-tests on the 5-seed eval matrix.

For each scenario, runs paired t-tests comparing MAPPO (best curriculum
checkpoint) against each baseline (Max-Pressure, Actuated Default / NEMA,
Fixed-Time) on person-time-loss across the 5 matched seeds, and prints
mean, std, MAPPO/MP ratio, t-statistic, and two-sided p-value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "results" / "eval_matrix"

SCENARIOS = ["am_peak", "pm_peak", "demand_surge", "midday_multimodal"]


def load_ptl(path: Path, controller: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if controller is not None:
        df = df[df["controller"] == controller]
    return df.sort_values("seed")[["seed", "person_time_loss_s"]].reset_index(drop=True)


def main() -> None:
    rows = []
    for scenario in SCENARIOS:
        m_csv = EVAL / f"asce_mappo_curriculum_best__{scenario}.csv"
        n_csv = EVAL / f"nema__{scenario}.csv"
        mappo = load_ptl(m_csv, "mappo")
        mp = load_ptl(m_csv, "max_pressure")
        ft = load_ptl(m_csv, "fixed_time")
        nema = load_ptl(n_csv, "nema")

        # NEMA seeds are 1000-1004; matrix seeds also 1000-1004 → matched.
        assert (mappo["seed"].values == mp["seed"].values).all()
        assert (mappo["seed"].values == nema["seed"].values).all()

        m_ptl = mappo["person_time_loss_s"].values
        mp_ptl = mp["person_time_loss_s"].values
        ft_ptl = ft["person_time_loss_s"].values
        n_ptl = nema["person_time_loss_s"].values

        for name, ptl in [("max_pressure", mp_ptl), ("nema", n_ptl), ("fixed_time", ft_ptl)]:
            t, p = stats.ttest_rel(m_ptl, ptl)
            diff = m_ptl - ptl
            ci_half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            rows.append(
                {
                    "scenario": scenario,
                    "baseline": name,
                    "mappo_mean": m_ptl.mean(),
                    "mappo_std": m_ptl.std(ddof=1),
                    "base_mean": ptl.mean(),
                    "base_std": ptl.std(ddof=1),
                    "ratio": m_ptl.mean() / ptl.mean(),
                    "delta_pct": 100 * (m_ptl.mean() / ptl.mean() - 1),
                    "paired_diff_mean": diff.mean(),
                    "paired_ci95_half": ci_half,
                    "t_stat": t,
                    "p_value": p,
                }
            )

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print(out.to_string(index=False))
    out_path = EVAL / "paired_significance.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
