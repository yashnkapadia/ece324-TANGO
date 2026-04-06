"""Online running observation normalizer using Welford's algorithm."""

from __future__ import annotations

import numpy as np


class ObsRunningNorm:
    """Per-feature Welford running mean/variance normalizer.

    Tracks statistics online across all observations seen during training.
    Padded-zero dimensions converge to mean≈0, std≈ε and stay 0 after
    normalization — no explicit masking needed.

    Save/load via state_dict() / load_state_dict() for checkpoint persistence.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = dim
        self.eps = eps
        self._count: int = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running stats with one observation vector (shape: [dim])."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError(f"ObsRunningNorm.update: non-finite values in observation: {x}")
        if x.size != self.dim:
            raise ValueError(f"ObsRunningNorm: expected dim={self.dim}, got {x.size}")
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return normalized observation. If fewer than 2 samples seen, returns x unchanged."""
        x = np.asarray(x, dtype=np.float32).ravel()
        if self._count < 2:
            return x
        var = self._M2 / (self._count - 1)
        std = np.sqrt(var + self.eps).astype(np.float32)
        mean = self._mean.astype(np.float32)
        return (x - mean) / std

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "eps": self.eps,
            "count": self._count,
            "mean": self._mean.tolist(),
            "M2": self._M2.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dim = int(state["dim"])
        self.eps = float(state["eps"])
        self._count = int(state["count"])
        self._mean = np.asarray(state["mean"], dtype=np.float64)
        self._M2 = np.asarray(state["M2"], dtype=np.float64)
