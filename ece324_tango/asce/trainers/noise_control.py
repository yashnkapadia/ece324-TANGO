from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os


@contextmanager
def quiet_output(enabled: bool):
    """Suppress noisy third-party stdout/stderr when enabled."""
    if not enabled:
        with nullcontext():
            yield
        return

    old_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    with open(os.devnull, "w") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            if old_tqdm is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = old_tqdm
