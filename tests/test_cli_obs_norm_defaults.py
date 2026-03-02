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
