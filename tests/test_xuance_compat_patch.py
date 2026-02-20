from ece324_tango.asce.trainers.xuance_compat import apply_xuance_value_norm_patch


def test_xuance_patch_idempotent():
    first = apply_xuance_value_norm_patch()
    second = apply_xuance_value_norm_patch()
    assert first is True
    assert second is False
