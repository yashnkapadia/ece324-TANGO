from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


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

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, global_obs: np.ndarray, n_valid_actions: int | None = None
    ) -> Dict[str, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        gobs_t = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
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

    def _compute_gae(self, traj: List[Transition], last_value: float = 0.0):
        rewards = np.array([t.reward for t in traj], dtype=np.float32)
        values = np.array([t.value for t in traj] + [last_value], dtype=np.float32)
        dones = np.array([t.done for t in traj], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return returns, advantages

    def update(self, batch: Dict[str, np.ndarray], ppo_epochs: int = 5, minibatch_size: int = 512):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        gobs = torch.tensor(batch["global_obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
        adv = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)

        last_losses = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        for _ in range(ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, minibatch_size):
                mb = idx[start : start + minibatch_size]

                logits = self.actor(obs[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

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

        return {
            "obs": np.stack(obs_list).astype(np.float32),
            "global_obs": np.stack(gobs_list).astype(np.float32),
            "actions": np.asarray(act_list, dtype=np.int64),
            "logp": np.asarray(logp_list, dtype=np.float32),
            "returns": np.asarray(ret_list, dtype=np.float32),
            "advantages": np.asarray(adv_list, dtype=np.float32),
        }

    def save(self, out_path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        torch.save(payload, out_path)

    def load(self, in_path: str):
        payload = torch.load(in_path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
