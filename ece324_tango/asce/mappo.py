from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ece324_tango.asce.obs_norm import ObsRunningNorm


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def augment_obs_with_mp(
    obs_arr: np.ndarray, mp_actions_list: list[int], n_actions: int
) -> np.ndarray:
    """Append MP one-hot to each agent's padded obs. Shape: [N, obs_dim + n_actions]."""
    N = obs_arr.shape[0]
    one_hots = np.zeros((N, n_actions), dtype=np.float32)
    for i, mp_a in enumerate(mp_actions_list):
        one_hots[i, int(mp_a)] = 1.0
    return np.concatenate([obs_arr, one_hots], axis=1)


class GatedActor(nn.Module):
    """Two-head actor: gate_head (binary follow/override) + phase_head (phase selection).

    gate_head: Linear(hidden_dim, 2) — index 0 = gate=1 (override), index 1 = gate=0 (follow MP)
    phase_head: Linear(hidden_dim, n_actions) — used only when gate==1
    MLP body (self.body) is shared between both heads.

    gate_init_bias: negative values (e.g. -2.0) bias gate toward follow-MP at init.
        Sign convention: bias[1] (follow-MP) is set to -gate_init_bias (positive when
        gate_init_bias is negative), bias[0] (override) is set to gate_init_bias (negative).
        This makes softmax assign higher probability to index 1 (follow MP).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, n_actions)
        self.gate_head = nn.Linear(hidden_dim, 2)
        # Warm-start: bias gate toward index 1 (follow MP) by setting
        # gate_head.bias[1] high and bias[0] low.
        with torch.no_grad():
            self.gate_head.bias[0] = gate_init_bias  # override logit low
            self.gate_head.bias[1] = -gate_init_bias  # follow-MP logit high

    def forward_gate(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns gate logits [N, 2]."""
        return self.gate_head(self.body(obs))

    def forward_phase(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns phase logits [N, n_actions]."""
        return self.phase_head(self.body(obs))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gate_logits [N,2], phase_logits [N, n_actions])."""
        h = self.body(obs)
        return self.gate_head(h), self.phase_head(h)


class Critic(nn.Module):
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    global_obs: np.ndarray
    action: int
    logp: float
    reward: float
    done: bool
    value: float
    n_valid_actions: int = 0
    mp_action: int = 0  # MP recommendation at this step (for action_gate mode)
    gate: int = 0  # Gate decision: 0=follow MP, 1=override (for action_gate mode)


class MAPPOTrainer:
    """Minimal MAPPO with parameter-sharing actor and centralized critic."""

    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        n_actions: int,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        use_obs_norm: bool = False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor = Actor(obs_dim, n_actions).to(self.device)
        self.critic = Critic(global_obs_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.use_obs_norm = use_obs_norm
        if use_obs_norm:
            self.obs_norm: ObsRunningNorm | None = ObsRunningNorm(obs_dim)
            self.gobs_norm: ObsRunningNorm | None = ObsRunningNorm(global_obs_dim)
        else:
            self.obs_norm = None
            self.gobs_norm = None

    def norm_update(self, obs: np.ndarray, global_obs: np.ndarray) -> None:
        """Update running normalizer stats. Call once per transition during training."""
        if self.obs_norm is not None:
            self.obs_norm.update(obs)
        if self.gobs_norm is not None:
            self.gobs_norm.update(global_obs)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        n_valid_actions: int | None = None,
    ) -> Dict[str, float]:
        obs_n = (
            self.obs_norm.normalize(obs)
            if self.obs_norm is not None
            else np.asarray(obs, dtype=np.float32)
        )
        gobs_n = (
            self.gobs_norm.normalize(global_obs)
            if self.gobs_norm is not None
            else np.asarray(global_obs, dtype=np.float32)
        )
        obs_t = torch.tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        gobs_t = torch.tensor(
            gobs_n, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits = self.actor(obs_t)
        if n_valid_actions is not None and n_valid_actions < logits.shape[-1]:
            logits[0, n_valid_actions:] = float("-inf")
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(gobs_t)
        return {
            "action": int(action.item()),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def act_batch(
        self,
        obs_list: list,
        global_obs: np.ndarray,
        n_valid_actions_list: list,
    ) -> list:
        """Single forward pass for all N agents — one GPU call per environment step."""
        N = len(obs_list)
        obs_arr = np.stack(
            [
                self.obs_norm.normalize(o)
                if self.obs_norm is not None
                else np.asarray(o, dtype=np.float32)
                for o in obs_list
            ]
        )
        gobs_n = (
            self.gobs_norm.normalize(global_obs)
            if self.gobs_norm is not None
            else np.asarray(global_obs, dtype=np.float32)
        )
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
        gobs_t = (
            torch.tensor(gobs_n, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(N, -1)
        )

        logits = self.actor(obs_t)
        for i, n_valid in enumerate(n_valid_actions_list):
            if n_valid < logits.shape[-1]:
                logits[i, n_valid:] = float("-inf")
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logps = dist.log_prob(actions)
        values = self.critic(gobs_t)

        return [
            {
                "action": int(actions[i].item()),
                "logp": float(logps[i].item()),
                "value": float(values[i].item()),
            }
            for i in range(N)
        ]

    def _compute_gae(self, traj: List[Transition], last_value: float = 0.0):
        rewards = np.array([t.reward for t in traj], dtype=np.float32)
        values = np.array([t.value for t in traj] + [last_value], dtype=np.float32)
        dones = np.array([t.done for t in traj], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return returns, advantages

    def update(
        self,
        batch: Dict[str, np.ndarray],
        ppo_epochs: int = 5,
        minibatch_size: int = 512,
    ):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        gobs = torch.tensor(
            batch["global_obs"], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(
            batch["returns"], dtype=torch.float32, device=self.device
        )
        adv = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        n_actions_total = int(self.actor.net[-1].out_features)
        n_valid_actions = torch.tensor(
            batch.get(
                "n_valid_actions",
                np.full((actions.shape[0],), n_actions_total, dtype=np.int64),
            ),
            dtype=torch.long,
            device=self.device,
        )

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.obs_norm is not None:
            obs_np = batch["obs"]
            gobs_np = batch["global_obs"]
            obs = torch.tensor(
                np.stack([self.obs_norm.normalize(o) for o in obs_np]),
                dtype=torch.float32,
                device=self.device,
            )
            gobs = torch.tensor(
                np.stack([self.gobs_norm.normalize(g) for g in gobs_np]),
                dtype=torch.float32,
                device=self.device,
            )

        n = obs.shape[0]
        idx = np.arange(n)

        last_losses = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        for _ in range(ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, minibatch_size):
                mb = idx[start : start + minibatch_size]

                logits = self.actor(obs[mb])
                mb_valid = torch.clamp(n_valid_actions[mb], min=1, max=logits.shape[-1])
                if torch.any(mb_valid < logits.shape[-1]):
                    action_ids = torch.arange(
                        logits.shape[-1], device=self.device
                    ).unsqueeze(0)
                    invalid_mask = action_ids >= mb_valid.unsqueeze(1)
                    logits = logits.masked_fill(invalid_mask, float("-inf"))
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * adv[mb]
                )
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                values = self.critic(gobs[mb])
                critic_loss = nn.functional.mse_loss(values, returns[mb])

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                last_losses = {
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy.item()),
                }

        return last_losses

    def build_batch(
        self,
        trajectories: Dict[str, List[Transition]],
        last_values: Dict[str, float] | None = None,
    ):
        obs_list: List[np.ndarray] = []
        gobs_list: List[np.ndarray] = []
        act_list: List[int] = []
        logp_list: List[float] = []
        ret_list: List[float] = []
        adv_list: List[float] = []
        n_valid_list: List[int] = []

        for agent_id, traj in trajectories.items():
            bootstrap = 0.0
            if last_values is not None and agent_id in last_values:
                bootstrap = last_values[agent_id]
            returns, advantages = self._compute_gae(traj, last_value=bootstrap)
            for t, transition in enumerate(traj):
                obs_list.append(transition.obs)
                gobs_list.append(transition.global_obs)
                act_list.append(transition.action)
                logp_list.append(transition.logp)
                ret_list.append(float(returns[t]))
                adv_list.append(float(advantages[t]))
                n_valid_list.append(int(transition.n_valid_actions))

        return {
            "obs": np.stack(obs_list).astype(np.float32),
            "global_obs": np.stack(gobs_list).astype(np.float32),
            "actions": np.asarray(act_list, dtype=np.int64),
            "logp": np.asarray(logp_list, dtype=np.float32),
            "returns": np.asarray(ret_list, dtype=np.float32),
            "advantages": np.asarray(adv_list, dtype=np.float32),
            "n_valid_actions": np.asarray(n_valid_list, dtype=np.int64),
        }

    def save(self, out_path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_norm": self.obs_norm.state_dict()
            if self.obs_norm is not None
            else None,
            "gobs_norm": self.gobs_norm.state_dict()
            if self.gobs_norm is not None
            else None,
            "use_obs_norm": bool(self.use_obs_norm),
        }
        torch.save(payload, out_path)

    @staticmethod
    def checkpoint_use_obs_norm(in_path: str) -> bool:
        payload = torch.load(in_path, map_location="cpu", weights_only=False)
        if "use_obs_norm" in payload:
            return bool(payload["use_obs_norm"])
        # Backward compatibility for checkpoints predating explicit metadata.
        return bool(
            payload.get("obs_norm") is not None or payload.get("gobs_norm") is not None
        )

    def load(self, in_path: str):
        # weights_only=False is required because the payload includes numpy/list arrays
        # from ObsRunningNorm.state_dict(). These are written by our own save() method
        # and loaded from the local models/ directory only — not from untrusted sources.
        payload = torch.load(in_path, map_location=self.device, weights_only=False)
        ckpt_use_obs_norm = bool(
            payload.get(
                "use_obs_norm",
                payload.get("obs_norm") is not None
                or payload.get("gobs_norm") is not None,
            )
        )
        if ckpt_use_obs_norm != bool(self.use_obs_norm):
            raise RuntimeError(
                f"Checkpoint use_obs_norm={ckpt_use_obs_norm} but trainer configured with "
                f"use_obs_norm={self.use_obs_norm}. Align flags and retry."
            )
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if self.obs_norm is not None and payload.get("obs_norm") is not None:
            self.obs_norm.load_state_dict(payload["obs_norm"])
        if self.gobs_norm is not None and payload.get("gobs_norm") is not None:
            self.gobs_norm.load_state_dict(payload["gobs_norm"])
