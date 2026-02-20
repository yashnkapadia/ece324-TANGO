from __future__ import annotations

from ece324_tango.asce.trainers.base import AsceTrainerBackend, EvalConfig, TrainConfig


class LibsignalBackend(AsceTrainerBackend):
    name = "libsignal"

    @staticmethod
    def _unsupported() -> RuntimeError:
        return RuntimeError(
            "LibSignal backend is registered for planning, but not yet wired into the "
            "ASCE train/eval pipeline. Current blocker: LibSignal uses its own "
            "config/registry/trainer stack and bundled simulator configs rather than "
            "our direct net/route contract."
        )

    def train(self, cfg: TrainConfig) -> None:
        raise self._unsupported()

    def evaluate(self, cfg: EvalConfig) -> None:
        raise self._unsupported()
