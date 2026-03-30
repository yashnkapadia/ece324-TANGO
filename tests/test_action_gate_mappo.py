from __future__ import annotations

import numpy as np
import pytest
import torch

from ece324_tango.asce.mappo import GatedActor, ResidualMAPPOTrainer, augment_obs_with_mp


def test_joint_logp_gate0_equals_gate_logp() -> None:
    """When gate=0 (follow MP), joint logp should equal log P(gate=0) only."""
    trainer = ResidualMAPPOTrainer(
        obs_dim=4,
        global_obs_dim=4,
        n_actions=3,
        residual_mode="action_gate",
    )
    # Force gate=0 deterministically: index 1 is gate=0 (follow MP)
    with torch.no_grad():
        trainer.actor.gate_head.bias.copy_(torch.tensor([-100.0, 100.0]))

    outs = trainer.act_batch_residual(
        obs_list=[np.zeros(4, dtype=np.float32)],
        global_obs=np.zeros(4, dtype=np.float32),
        n_valid_actions_list=[3],
        mp_actions_list=[1],
    )

    # gate=0 chosen, so logp should be log P(gate=0) only
    # Must use augmented obs through full actor forward to get correct gate logits
    aug_obs = augment_obs_with_mp(np.zeros((1, 4), dtype=np.float32), [1], 3)
    aug_t = torch.tensor(aug_obs, dtype=torch.float32)
    gate_logits, _ = trainer.actor(aug_t)
    expected_logp = torch.log_softmax(gate_logits, dim=-1)[0, 1].item()
    assert abs(outs[0]["logp"] - expected_logp) < 1e-5


def test_joint_logp_gate1_equals_sum() -> None:
    """When gate=1 (use phase head), joint logp = log P(gate=1) + log P(phase)."""
    trainer = ResidualMAPPOTrainer(
        obs_dim=4,
        global_obs_dim=4,
        n_actions=3,
        residual_mode="action_gate",
    )
    # Force gate=1 deterministically: index 0 is gate=1 (use phase head)
    with torch.no_grad():
        trainer.actor.gate_head.bias.copy_(torch.tensor([100.0, -100.0]))

    outs = trainer.act_batch_residual(
        obs_list=[np.zeros(4, dtype=np.float32)],
        global_obs=np.zeros(4, dtype=np.float32),
        n_valid_actions_list=[3],
        mp_actions_list=[1],
    )

    # Compute expected: log P(gate=1) + log P(phase action)
    # Must use augmented obs through full actor forward
    aug_obs = augment_obs_with_mp(np.zeros((1, 4), dtype=np.float32), [1], 3)
    aug_t = torch.tensor(aug_obs, dtype=torch.float32)
    gate_logits, phase_logits = trainer.actor(aug_t)
    gate_logp = torch.log_softmax(gate_logits, dim=-1)[0, 0].item()

    phase_logits_masked = phase_logits.clone()
    phase_logits_masked[0, 3:] = float("-inf")  # mask invalid actions
    phase_logp = torch.log_softmax(phase_logits_masked, dim=-1)[
        0, outs[0]["action"]
    ].item()

    expected = gate_logp + phase_logp
    assert abs(outs[0]["logp"] - expected) < 1e-5


def test_gate0_phase_head_zero_gradient() -> None:
    """PPO update on a gate=0-only batch must produce zero gradient on phase head."""
    trainer = ResidualMAPPOTrainer(
        obs_dim=4,
        global_obs_dim=4,
        n_actions=3,
        residual_mode="action_gate",
    )

    # Build a batch where all transitions have gate=0
    # Obs must be augmented (obs_dim + n_actions = 7) as act_batch_residual would produce
    batch = {
        "obs": np.zeros((4, 7), dtype=np.float32),
        "global_obs": np.zeros((4, 4), dtype=np.float32),
        "actions": np.array([1, 2, 0, 1], dtype=np.int64),
        "logp": np.full(4, -0.5, dtype=np.float32),
        "returns": np.zeros(4, dtype=np.float32),
        "advantages": np.ones(4, dtype=np.float32),
        "n_valid_actions": np.full(4, 3, dtype=np.int64),
        "gate_decisions": np.zeros(4, dtype=np.int64),  # all gate=0
    }

    trainer.update(batch=batch, ppo_epochs=1, minibatch_size=4)

    # Phase head weights should have zero (or None) gradient
    for p in trainer.actor.phase_head.parameters():
        assert p.grad is None or p.grad.abs().max().item() < 1e-9


def test_gate0_dispatches_mp_action() -> None:
    """When gate=0, the returned action must be the MP action, not the phase head."""
    trainer = ResidualMAPPOTrainer(
        obs_dim=4,
        global_obs_dim=4,
        n_actions=3,
        residual_mode="action_gate",
    )
    with torch.no_grad():
        trainer.actor.gate_head.bias.copy_(torch.tensor([-100.0, 100.0]))

    mp_action = 2
    outs = trainer.act_batch_residual(
        obs_list=[np.zeros(4, dtype=np.float32)],
        global_obs=np.zeros(4, dtype=np.float32),
        n_valid_actions_list=[3],
        mp_actions_list=[mp_action],
    )
    assert outs[0]["action"] == mp_action


def test_gate1_dispatches_phase_action() -> None:
    """When gate=1, the returned action comes from the phase head, not MP."""
    trainer = ResidualMAPPOTrainer(
        obs_dim=4,
        global_obs_dim=4,
        n_actions=3,
        residual_mode="action_gate",
    )
    # Force gate=1
    with torch.no_grad():
        trainer.actor.gate_head.bias.copy_(torch.tensor([100.0, -100.0]))
        # Force phase head to always pick action 0
        for p in trainer.actor.phase_head.parameters():
            p.zero_()
        trainer.actor.phase_head.bias.copy_(
            torch.tensor([100.0, -100.0, -100.0])
        )

    outs = trainer.act_batch_residual(
        obs_list=[np.zeros(4, dtype=np.float32)],
        global_obs=np.zeros(4, dtype=np.float32),
        n_valid_actions_list=[3],
        mp_actions_list=[2],  # MP says 2, but gate=1 should ignore it
    )
    assert outs[0]["action"] == 0


def test_gate_warm_start_biases_toward_zero() -> None:
    """Negative gate_init_bias should make gate=0 (follow MP) dominant."""
    torch.manual_seed(42)
    actor = GatedActor(obs_dim=4, n_actions=3, hidden_dim=64, gate_init_bias=-3.0)

    obs = torch.randn(200, 4)
    gate_logits = actor.forward_gate(obs)
    gate_probs = torch.softmax(gate_logits, dim=-1)
    # Index 1 = gate=0 (follow MP); should be > 0.85 on average
    gate0_frac = (gate_probs[:, 1] > gate_probs[:, 0]).float().mean().item()
    assert gate0_frac > 0.85, (
        f"Expected >85% gate=0 samples with bias=-3.0, got {gate0_frac:.2%}"
    )


def test_residual_mode_none_unchanged() -> None:
    """residual_mode='none' should behave like vanilla MAPPOTrainer."""
    trainer = ResidualMAPPOTrainer(
        obs_dim=4,
        global_obs_dim=4,
        n_actions=3,
        residual_mode="none",
    )

    outs = trainer.act_batch(
        obs_list=[np.zeros(4, dtype=np.float32)],
        global_obs=np.zeros(4, dtype=np.float32),
        n_valid_actions_list=[3],
    )

    # Should return same structure as MAPPOTrainer.act_batch
    assert len(outs) == 1
    assert "action" in outs[0]
    assert "logp" in outs[0]
    assert "value" in outs[0]

    # No gate_head attribute (or it is None) in none mode
    gate_head = getattr(trainer.actor, "gate_head", None)
    assert gate_head is None
