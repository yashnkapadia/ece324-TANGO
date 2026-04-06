"""Minimal live status panel for curriculum training using rich.Live.

Shows a persistent header with progress, batch stats, gate fraction,
and eval results above the normal scrolling log output.

Falls back to a no-op when stdout is not a TTY (e.g. piped through tee).
Use --log-file to write logs to a file directly and skip tee so the TUI works.
"""

from __future__ import annotations

import math
import sys
import time as _time


class TrainingStatus:
    """Lightweight status tracker that renders a rich panel on each update.

    Automatically disables the live panel when not connected to a real terminal.
    All update methods are safe to call regardless — they become no-ops.
    """

    def __init__(
        self,
        total_episodes: int,
        num_workers: int,
        scenario_names: list[str],
        device: str = "cuda",
        scenario_id: str = "curriculum",
        start_episode: int = 0,
        initial_best_ratio: float = float("inf"),
        initial_best_scenario: str = "",
        initial_eval_ep: int | None = None,
        initial_eval_ratios: dict[str, float] | None = None,
    ):
        self.total_episodes = total_episodes
        self.num_workers = num_workers
        self.scenario_names = scenario_names
        self.device = device
        self.scenario_id = scenario_id

        # Mutable state
        self.current_ep = start_episode
        self.batch_wall_s = 0.0
        self.avg_gate_frac = 0.0
        self.avg_reward = 0.0
        self.eval_ep: int | None = None
        self.eval_ratios: dict[str, float] = initial_eval_ratios or {}
        self.best_ratio = initial_best_ratio
        self.best_scenario = initial_best_scenario
        self.eval_running = False
        self._ema_batch_wall_s: float | None = None
        self.start_time = _time.time()
        self.eval_ep = initial_eval_ep

        self._is_tty = sys.stderr.isatty()
        self._live = None
        self._console = None
        self._loguru_id = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def start(self) -> TrainingStatus:
        if not self._is_tty:
            return self

        import ece324_tango.config as _cfg

        _cfg._tui_active = True

        from rich.console import Console
        from rich.live import Live

        self._console = Console(stderr=True)
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=1,
            transient=True,
        )
        self._live.start()

        # Redirect loguru to print through rich.Live so log lines
        # don't collide with the panel. Live.console.print() clears
        # the panel, prints the line, then redraws the panel.
        from loguru import logger

        def _rich_sink(message):
            if self._live is not None:
                self._live.console.print(message, end="", highlight=False)
            else:
                sys.stderr.write(str(message))

        self._loguru_id = logger.add(
            _rich_sink,
            format="{time:HH:mm:ss.SSS} | {level: <8} | {message}",
            level="INFO",
        )
        return self

    def stop(self):
        if self._live is not None:
            # Remove the loguru sink before stopping Live
            from loguru import logger

            if hasattr(self, "_loguru_id"):
                try:
                    logger.remove(self._loguru_id)
                except ValueError:
                    pass
            self._live.stop()
            self._live = None

        import ece324_tango.config as _cfg

        _cfg._tui_active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------
    # Update methods (called from training loop)
    # ------------------------------------------------------------------

    def update_batch(
        self,
        last_ep: int,
        batch_wall_s: float,
        avg_gate_frac: float,
        avg_reward: float,
    ):
        self.current_ep = last_ep + 1
        self.batch_wall_s = batch_wall_s
        self.avg_gate_frac = avg_gate_frac
        self.avg_reward = avg_reward
        if self._ema_batch_wall_s is None:
            self._ema_batch_wall_s = batch_wall_s
        else:
            # Smooth the batch duration so ETA stays stable across updates.
            self._ema_batch_wall_s = 0.7 * self._ema_batch_wall_s + 0.3 * batch_wall_s
        self._refresh()

    def update_eval_started(self):
        self.eval_running = True
        self._refresh()

    def update_eval_results(
        self,
        eval_ep: int,
        ratios: dict[str, float],
    ):
        self.eval_ep = eval_ep
        self.eval_ratios = ratios
        self.eval_running = False
        if ratios:
            worst = max(ratios.values())
            best_name = max(ratios, key=ratios.get)
            if worst < self.best_ratio:
                self.best_ratio = worst
                self.best_scenario = best_name
        self._refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self):
        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        from rich.table import Table
        from rich.text import Text

        grid = Table.grid(padding=(0, 1))
        grid.add_column(min_width=66)

        # Title
        grid.add_row(
            Text(f"─ {self.scenario_id.upper()} ", style="bold cyan") + Text("─" * 50, style="dim")
        )

        # Progress bar
        pct = self.current_ep / max(self.total_episodes, 1)
        filled = int(pct * 30)
        bar = "\u2588" * filled + "\u2591" * (30 - filled)
        remaining_episodes = max(self.total_episodes - self.current_ep, 0)
        if remaining_episodes == 0:
            eta_str = "0s"
        elif self._ema_batch_wall_s is not None:
            remaining_batches = math.ceil(remaining_episodes / max(self.num_workers, 1))
            eta_str = _fmt_duration(self._ema_batch_wall_s * remaining_batches)
        else:
            eta_str = "\u2014"
        grid.add_row(
            Text(
                f" Progress: {bar} {self.current_ep}/{self.total_episodes} "
                f"({pct:.0%})  ETA: {eta_str}"
            )
        )

        # Batch stats
        grid.add_row(
            Text(
                f" Batch: {self.num_workers}w \u00d7 {self.batch_wall_s:.0f}s  "
                f"Device: {self.device}  "
                f"Reward: {self.avg_reward:.4f}  "
                f"Gate: {self.avg_gate_frac:.3f}"
            )
        )

        # Eval results
        if self.eval_running:
            eval_line = " Eval: running in background ..."
        elif self.eval_ratios:
            parts = "  ".join(f"{_short_name(n)}: {r:.3f}" for n, r in self.eval_ratios.items())
            eval_line = f" Eval ep {self.eval_ep}: {parts}"
        else:
            eval_line = " Eval: pending"
        grid.add_row(Text(eval_line))

        # Best
        if self.best_ratio < float("inf"):
            grid.add_row(
                Text(
                    f" Best: {self.best_ratio:.3f} ({self.best_scenario})",
                    style="bold green",
                )
            )

        grid.add_row(Text("\u2500" * 66, style="dim"))
        return grid


def _short_name(scenario: str) -> str:
    """Shorten scenario names for compact display."""
    _MAP = {
        "am_peak": "am",
        "pm_peak": "pm",
        "demand_surge": "surge",
        "midday_multimodal": "midday",
    }
    return _MAP.get(scenario, scenario[:8])


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"
