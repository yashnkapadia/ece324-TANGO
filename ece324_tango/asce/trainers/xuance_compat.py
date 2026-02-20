from __future__ import annotations

import numpy as np

_PATCHED = False


def apply_xuance_value_norm_patch() -> bool:
    """Patch Xuance on-policy buffer to handle value_normalizer output shape robustly.

    Xuance's value normalizer can return shape (1, T) on some custom env paths,
    while finish_path expects a 1D vector of length T. This patch flattens
    denormalized arrays before scalar indexing.

    Returns:
        True if patch applied in this call, False if it was already applied.
    """

    global _PATCHED
    if _PATCHED:
        return False

    from xuance.common import memory_tools_marl

    def _patched_finish_path(self, i_env=None, value_next=None, value_normalizer=None):
        if self.size == 0:
            return
        if self.full:
            path_slice = np.arange(self.start_ids[i_env], self.n_size).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i_env], self.ptr).astype(np.int32)

        use_value_norm = value_normalizer is not None
        use_parameter_sharing = False
        if use_value_norm and value_normalizer.keys() != set(self.agent_keys):
            use_parameter_sharing = True

        for key in self.agent_keys:
            rewards = self.data["rewards"][key][i_env, path_slice]
            vs = np.append(self.data["values"][key][i_env, path_slice], [value_next[key]], axis=0)
            dones = self.data["terminals"][key][i_env, path_slice]
            returns = np.zeros_like(rewards)
            last_gae_lam = 0.0
            step_nums = len(path_slice)
            key_vn = self.agent_keys[0] if use_parameter_sharing else key

            if use_value_norm:
                # Xuance may return shape (1, T); flatten to shape (T,) before scalar indexing.
                vs_denorm = np.asarray(value_normalizer[key_vn].denormalize(vs), dtype=np.float32).reshape(-1)
                if vs_denorm.size != vs.size:
                    # conservative guard: align by truncation if backend returns unexpected shape.
                    vs_denorm = vs_denorm[: vs.size]
            else:
                vs_denorm = None

            if self.use_gae:
                for t in reversed(range(step_nums)):
                    if use_value_norm:
                        vs_t = float(vs_denorm[t])
                        vs_next = float(vs_denorm[t + 1])
                    else:
                        vs_t = float(vs[t])
                        vs_next = float(vs[t + 1])
                    delta = float(rewards[t]) + (1.0 - float(dones[t])) * self.gamma * vs_next - vs_t
                    last_gae_lam = (
                        delta + (1.0 - float(dones[t])) * self.gamma * self.gae_lambda * last_gae_lam
                    )
                    returns[t] = last_gae_lam + vs_t
                advantages = returns - (vs_denorm[:-1] if use_value_norm else vs[:-1])
            else:
                returns_ = np.append(returns, [value_next[key]], axis=0)
                for t in reversed(range(step_nums)):
                    returns_[t] = float(rewards[t]) + (1.0 - float(dones[t])) * self.gamma * returns_[t + 1]
                advantages = returns_ - (vs_denorm if use_value_norm else vs)
                advantages = advantages[:-1]
                returns = returns_[:-1]

            self.data["returns"][key][i_env, path_slice] = returns
            self.data["advantages"][key][i_env, path_slice] = advantages
        self.start_ids[i_env] = self.ptr

    memory_tools_marl.MARL_OnPolicyBuffer.finish_path = _patched_finish_path
    _PATCHED = True
    return True
