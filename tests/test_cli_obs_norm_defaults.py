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


def test_cli_reward_modes_include_time_loss():
    train_default = inspect.signature(train.main).parameters["reward_mode"].default
    predict_default = inspect.signature(predict.main).parameters["reward_mode"].default
    assert isinstance(train_default, typer.models.OptionInfo)
    assert isinstance(predict_default, typer.models.OptionInfo)
    assert train_default.default == "objective"
    assert predict_default.default == "objective"
    assert "time_loss" in train._VALID_REWARD_MODES
    assert "time_loss" in predict._VALID_REWARD_MODES
