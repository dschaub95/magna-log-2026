"""Tests for ``parse_dot_overrides`` CLI parsing."""

from magna.config import parse_dot_overrides


def test_equals_form():
    flat, rest = parse_dot_overrides(["--cfg_path=foo.yaml", "--train.lr=1e-4"])
    assert flat == {"cfg_path": "foo.yaml", "train.lr": 1e-4}
    assert rest == []


def test_space_separated_path():
    flat, rest = parse_dot_overrides(["--cfg_path", "configs/real_data.yaml"])
    assert flat == {"cfg_path": "configs/real_data.yaml"}
    assert rest == []


def test_space_separated_negative_number():
    flat, rest = parse_dot_overrides(["--env.time_horizon", "-1"])
    assert flat == {"env.time_horizon": -1}
    assert rest == []


def test_bare_flag_without_value_goes_to_rest():
    flat, rest = parse_dot_overrides(["--cfg_path"])
    assert flat == {}
    assert rest == ["--cfg_path"]


def test_next_long_flag_leaves_key_in_rest():
    flat, rest = parse_dot_overrides(["--cfg_path", "--train.lr=1"])
    assert flat == {"train.lr": 1}
    assert rest == ["--cfg_path"]
