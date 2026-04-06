from ece324_tango.asce.runtime import extract_step_details


def test_extract_step_details_preserves_termination_flag():
    out = (
        {"a0": [0.0]},
        {"a0": 0.0},
        {"a0": True},
        {"a0": False},
        {"a0": {}},
    )

    obs, rewards, done, infos, terminated, truncated = extract_step_details(out)
    assert done is True
    assert terminated is True
    assert truncated is False
    assert "a0" in obs and "a0" in rewards and "a0" in infos


def test_extract_step_details_preserves_truncation_flag():
    out = (
        {"a0": [0.0]},
        {"a0": 0.0},
        {"a0": False},
        {"a0": True},
        {"a0": {}},
    )

    _, _, done, _, terminated, truncated = extract_step_details(out)
    assert done is True
    assert terminated is False
    assert truncated is True
