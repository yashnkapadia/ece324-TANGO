import math

import numpy as np
import torch

from ece324_tango.asce.mappo import MAPPOTrainer, Transition


def _zero_module_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.zero_()


def test_update_respects_action_mask_from_batch() -> None:
    trainer = MAPPOTrainer(
        obs_dim=1,
        global_obs_dim=1,
        n_actions=4,
        actor_lr=0.0,
        critic_lr=0.0,
        entropy_coef=0.0,
    )
    _zero_module_parameters(trainer.actor)
    _zero_module_parameters(trainer.critic)

    batch = {
        "obs": np.asarray([[0.0], [0.0]], dtype=np.float32),
        "global_obs": np.asarray([[0.0], [0.0]], dtype=np.float32),
        "actions": np.asarray([0, 0], dtype=np.int64),
        "logp": np.asarray([math.log(0.5), math.log(0.25)], dtype=np.float32),
        "returns": np.asarray([0.0, 0.0], dtype=np.float32),
        "advantages": np.asarray([1.0, 3.0], dtype=np.float32),
        "n_valid_actions": np.asarray([2, 4], dtype=np.int64),
    }

    losses = trainer.update(batch=batch, ppo_epochs=1, minibatch_size=2)

    # If update uses the same per-sample masks as action collection, ratio=1 for both
    # samples under zero logits and actor loss is ~0 (after advantage normalization).
    assert abs(losses["actor_loss"]) < 1e-6


def test_build_batch_carries_per_transition_valid_action_count() -> None:
    trainer = MAPPOTrainer(obs_dim=1, global_obs_dim=1, n_actions=4)

    trajectories = {
        "a0": [
            Transition(
                obs=np.asarray([0.0], dtype=np.float32),
                global_obs=np.asarray([0.0], dtype=np.float32),
                action=0,
                logp=-0.1,
                reward=1.0,
                done=False,
                value=0.0,
                n_valid_actions=2,
            )
        ]
    }

    batch = trainer.build_batch(trajectories, last_values={"a0": 0.0})
    assert "n_valid_actions" in batch
    assert batch["n_valid_actions"].tolist() == [2]


def test_act_never_samples_masked_actions() -> None:
    torch.manual_seed(0)
    trainer = MAPPOTrainer(obs_dim=1, global_obs_dim=1, n_actions=4)

    # Make invalid action 3 overwhelmingly likely if mask were absent.
    _zero_module_parameters(trainer.actor)
    with torch.no_grad():
        trainer.actor.net[-1].bias[:] = torch.tensor([0.0, 0.0, 0.0, 50.0])

    for _ in range(64):
        out = trainer.act(
            obs=np.asarray([0.0], dtype=np.float32),
            global_obs=np.asarray([0.0], dtype=np.float32),
            n_valid_actions=2,
        )
        assert out["action"] in {0, 1}


def test_act_batch_never_samples_masked_actions() -> None:
    torch.manual_seed(0)
    trainer = MAPPOTrainer(obs_dim=1, global_obs_dim=1, n_actions=4)

    _zero_module_parameters(trainer.actor)
    with torch.no_grad():
        trainer.actor.net[-1].bias[:] = torch.tensor([0.0, 0.0, 0.0, 50.0])

    outs = trainer.act_batch(
        obs_list=[np.asarray([0.0], dtype=np.float32) for _ in range(16)],
        global_obs=np.asarray([0.0], dtype=np.float32),
        n_valid_actions_list=[2 for _ in range(16)],
    )
    assert all(o["action"] in {0, 1} for o in outs)
