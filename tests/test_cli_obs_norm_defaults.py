import inspect

import typer

from ece324_tango.modeling import predict, train


def test_train_cli_defaults_obs_norm_enabled():
    default = inspect.signature(train.main).parameters["use_obs_norm"].default
    assert isinstance(default, typer.models.OptionInfo)
    assert default.default is True


def test_predict_cli_defaults_obs_norm_enabled():
    default = inspect.signature(predict.main).parameters["use_obs_norm"].default
    assert isinstance(default, typer.models.OptionInfo)
    assert default.default is True


def test_train_cli_defaults_eval_workers_and_baselines():
    eval_workers = inspect.signature(train.main).parameters["eval_workers"].default
    eval_baselines = inspect.signature(train.main).parameters["eval_baselines"].default
    assert isinstance(eval_workers, typer.models.OptionInfo)
    assert isinstance(eval_baselines, typer.models.OptionInfo)
    assert eval_workers.default == 2
    assert eval_baselines.default == "max_pressure"


def test_cli_reward_modes_include_time_loss_and_residual_mp():
    train_default = inspect.signature(train.main).parameters["reward_mode"].default
    predict_default = inspect.signature(predict.main).parameters["reward_mode"].default
    assert isinstance(train_default, typer.models.OptionInfo)
    assert isinstance(predict_default, typer.models.OptionInfo)
    assert train_default.default == "objective"
    assert predict_default.default == "objective"
    assert "time_loss" in train._VALID_REWARD_MODES
    assert "time_loss" in predict._VALID_REWARD_MODES
    assert "residual_mp" in train._VALID_REWARD_MODES
    assert "residual_mp" in predict._VALID_REWARD_MODES


def test_cli_eval_baselines_include_supported_baselines():
    assert "max_pressure" in train._VALID_EVAL_BASELINES
    assert "fixed_time" in train._VALID_EVAL_BASELINES
    assert "nema" in train._VALID_EVAL_BASELINES
