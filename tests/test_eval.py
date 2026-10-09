import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from magna.config import Config, apply_cli_overrides
from magna.eval import (
    build_eval_env_and_policy,
    build_eval_env_kwargs,
    formation_metrics,
    load_eval_data,
    run_eval,
    run_policy_inference,
)


def _stub_run_eval_dependencies(
    monkeypatch,
    *,
    targets,
    initial_positions,
):
    calls = {}

    def fake_load_eval_data(resolved_cfg, **kwargs):
        calls["load_eval_data_cfg"] = resolved_cfg
        return (
            targets,
            initial_positions,
            "spatial_target",
            "spatial_init",
            None,
            None,
            None,
        )

    class DummyEnv:
        def close(self):
            calls["env_closed"] = True

    def fake_build_eval_env_and_policy(resolved_cfg, **kwargs):
        calls["build_eval_env_and_policy"] = kwargs
        return DummyEnv(), "policy", 7, {"env": "kwargs"}

    def fake_load_policy_weights(policy, loaded_checkpoint_path, device):
        calls["load_policy_weights"] = {
            "policy": policy,
            "checkpoint_path": loaded_checkpoint_path,
            "device": device,
        }

    def fake_run_policy_inference(**kwargs):
        calls["run_policy_inference"] = kwargs
        return {
            "video_path": "video.mp4",
            "num_steps": 1,
            "error_mean": 0.1,
            "rel_err": 0.2,
            "mae": 0.3,
            "mse": 0.35,
            "rmse": 0.4,
            "gw": 0.5,
            "positions": ["trajectory"],
        }

    monkeypatch.setattr("magna.eval.load_eval_data", fake_load_eval_data)
    monkeypatch.setattr(
        "magna.eval.build_eval_env_and_policy",
        fake_build_eval_env_and_policy,
    )
    monkeypatch.setattr("magna.eval.load_policy_weights", fake_load_policy_weights)
    monkeypatch.setattr("magna.eval.run_policy_inference", fake_run_policy_inference)
    return calls


def test_formation_metrics_includes_gw_and_allows_unequal_counts():
    target = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    pred = target + 0.1
    same = formation_metrics(target, pred)
    assert set(same) == {"rel_err", "mae", "mse", "rmse", "gw"}
    assert same["gw"] >= 0.0
    scaled = formation_metrics(target * 10.0, pred * 10.0)
    assert np.isclose(same["gw"], scaled["gw"])
    assert np.isclose(same["rmse"], scaled["rmse"])
    unequal = formation_metrics(target, pred[:2])
    assert set(unequal) == {"gw"}
    assert unequal["gw"] >= 0.0


def test_load_eval_data_processes_configured_target_and_init_keys(tmp_path, adata):
    adata.obsm["spatial_target"] = adata.obsm["spatial"] + 10.0
    adata.obsm["spatial_init"] = adata.obsm["spatial"] + 20.0
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)

    cfg = Config()
    cfg.data.path = str(path)
    cfg.data.target_obsm_key = "spatial_target"
    cfg.data.init_obsm_key = "spatial_init"
    cfg.data.coordinate_scale = 1.0
    cfg.env.width = 100
    cfg.env.height = 100

    (
        targets,
        initial_positions,
        target_key,
        initial_key,
        color_values,
        loaded_id_features,
        initial_headings,
    ) = load_eval_data(cfg)

    assert target_key == "spatial_target"
    assert initial_key == "spatial_init"
    assert targets.shape == (3, 2)
    assert initial_positions is not None
    assert initial_positions.shape == (3, 2)
    assert color_values is None
    assert loaded_id_features is None
    assert initial_headings is None


def test_load_eval_data_processes_configured_init_heading_key(tmp_path, adata):
    adata.obsm["spatial_init"] = adata.obsm["spatial"] + 20.0
    adata.obsm["heading_init"] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)

    cfg = Config()
    cfg.data.path = str(path)
    cfg.data.init_heading_obsm_key = "heading_init"
    cfg.data.coordinate_scale = 1.0
    cfg.env.width = 100
    cfg.env.height = 100

    *_, initial_headings = load_eval_data(cfg)

    np.testing.assert_allclose(initial_headings, adata.obsm["heading_init"].reshape(-1))


def test_load_eval_data_resolves_auto_circular_size(tmp_path, adata):
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, np.sqrt(3.0)],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)

    cfg = Config()
    cfg.data.path = str(path)
    cfg.data.coordinate_scale = 1.0
    cfg.env.size = "auto"
    cfg.env.circular = True
    cfg.env.width = None
    cfg.env.height = None
    cfg.env.agent_radius = 1

    targets, _, _, _, _, _, _ = load_eval_data(cfg)

    assert cfg.env.width == 20
    assert cfg.env.height == 20
    half = float(cfg.env.width) * 0.5
    env_center = np.array([half, half], dtype=np.float32)
    distances = np.linalg.norm(targets - env_center, axis=1)
    assert np.all(distances <= half - float(cfg.env.agent_radius) + 1e-5)


def test_load_eval_data_errors_when_spatial_init_shape_mismatches(tmp_path, adata):
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)

    cfg = Config()
    cfg.data.path = str(path)
    cfg.data.init_obsm_key = "spatial_init"

    class DummyAnnData:
        pass

    def fake_load_adata_cell_positions_and_colors(*args, **kwargs):
        return adata.obsm["spatial"], adata.obsm["spatial"][:2], None, None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "magna.eval.load_spatial_data",
        lambda path, **_: DummyAnnData(),
    )
    monkeypatch.setattr(
        "magna.eval.load_adata_cell_positions_and_colors",
        fake_load_adata_cell_positions_and_colors,
    )

    with pytest.raises(ValueError, match="spatial_init"):
        load_eval_data(cfg)
    monkeypatch.undo()


def test_run_eval_uses_checkpoint_and_auto_reuse_initial_positions(
    monkeypatch, tmp_path, adata, capsys
):
    cfg = Config()
    cfg.data.path = str(tmp_path / "cells.h5ad")
    cfg.train.device = "cpu"
    cfg.env.reuse_initial_positions = False
    override_path = tmp_path / "override.h5ad"
    adata.write_h5ad(override_path)

    calls = _stub_run_eval_dependencies(
        monkeypatch,
        targets=np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        initial_positions="initials",
    )

    result = run_eval(
        cfg,
        tmp_path / "abc.pt",
        flat_overrides={"data.path": str(override_path)},
        video_path="out.mp4",
        n_rollouts=2,
        greedy=True,
    )

    assert calls["load_eval_data_cfg"].data.path == str(override_path)
    assert calls["build_eval_env_and_policy"]["reuse_initial_positions"] is True
    assert calls["run_policy_inference"]["n_rollouts"] == 2
    assert calls["run_policy_inference"]["targets"] is not None
    assert calls["run_policy_inference"]["n_steps"] is None
    assert result["checkpoint_path"] == str(tmp_path / "abc.pt")
    assert calls["env_closed"] is True
    assert f"Checkpoint: {tmp_path / 'abc.pt'}" in capsys.readouterr().out


def test_run_eval_autoreset_false_keeps_training_step_count(
    monkeypatch, tmp_path, adata
):
    cfg = Config.model_validate({"env": {"time_horizon": 2048}})
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)
    cfg.data.path = str(path)
    calls = _stub_run_eval_dependencies(
        monkeypatch,
        targets=np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        initial_positions=None,
    )
    run_eval(cfg, "abc.pt", video_path=None, autoreset=False)
    assert calls["run_policy_inference"]["n_steps"] == 2048
    assert calls["build_eval_env_and_policy"]["time_horizon"] == -1


def test_run_eval_defaults_cell_colors_from_targets(monkeypatch, tmp_path):
    cfg = Config()
    cfg.train.device = "cpu"

    calls = _stub_run_eval_dependencies(
        monkeypatch,
        targets=np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        initial_positions=None,
    )

    run_eval(cfg, "abc.pt", video_path="out.mp4")

    assert calls["build_eval_env_and_policy"]["cell_colors"].tolist() == [
        [0, 0, 0],
        [255, 255, 0],
    ]
    assert calls["env_closed"] is True


def test_run_eval_preserves_explicit_cell_colors(monkeypatch, tmp_path):
    cfg = Config()
    cfg.train.device = "cpu"

    explicit_colors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    calls = _stub_run_eval_dependencies(
        monkeypatch,
        targets=np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        initial_positions=None,
    )

    run_eval(
        cfg,
        "abc.pt",
        video_path="out.mp4",
        cell_colors=explicit_colors,
    )

    assert calls["build_eval_env_and_policy"]["cell_colors"] is explicit_colors
    assert calls["env_closed"] is True


def test_run_eval_uses_data_color_key_when_present(monkeypatch, tmp_path):
    cfg = Config()
    cfg.train.device = "cpu"
    cfg.data.color_key = "cell_type"

    calls = _stub_run_eval_dependencies(
        monkeypatch,
        targets=np.array([[0.0, 0.0], [2.0, 4.0], [1.0, 3.0]], dtype=np.float32),
        initial_positions=None,
    )

    monkeypatch.setattr(
        "magna.eval.load_eval_data",
        lambda resolved_cfg, **kwargs: (
            np.array([[0.0, 0.0], [2.0, 4.0], [1.0, 3.0]], dtype=np.float32),
            None,
            "spatial_target",
            None,
            np.asarray(["B", "A", "B"], dtype=object),
            None,
            None,
        ),
    )

    run_eval(cfg, "abc.pt", video_path="out.mp4")

    colors = calls["build_eval_env_and_policy"]["cell_colors"]
    assert colors.dtype == np.uint8
    assert colors.shape == (3, 3)
    np.testing.assert_array_equal(colors[0], colors[2])
    assert not np.array_equal(colors[0], colors[1])


def test_apply_cli_overrides_recomputes_when_derived_field_is_set_to_none():
    cfg = Config().resolve_dependent_defaults()

    updated = apply_cli_overrides(
        cfg,
        {
            "env.agent_radius": 32,
            "env.max_speed": None,
            "env.collision_grid_size": None,
        },
    )

    assert updated.env.agent_radius == 32
    assert updated.env.max_speed == 32.0
    assert updated.env.collision_grid_size == 64


def test_apply_cli_overrides_clears_inherited_dimensions_when_size_is_overridden():
    cfg = Config().resolve_dependent_defaults()

    updated = apply_cli_overrides(
        cfg,
        {
            "env.size": "auto",
            "env.circular": True,
        },
    )

    assert updated.env.size == "auto"
    assert updated.env.width is None
    assert updated.env.height is None
    assert updated.env.depth is None
    assert updated.env.circular is True


def test_apply_cli_overrides_recomputes_depth_with_fixed_size():
    updated = apply_cli_overrides(
        Config().resolve_dependent_defaults(), {"env.size": 900}
    )
    assert (updated.env.width, updated.env.height, updated.env.depth) == (900, 900, 900)


def test_build_eval_env_kwargs_excludes_config_only_size():
    cfg = Config.model_validate({"env": {"size": 900}})
    targets = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)

    env_kwargs = build_eval_env_kwargs(cfg, targets)

    assert env_kwargs["cfg"].size == 900
    assert env_kwargs["cfg"].width == 900
    assert env_kwargs["cfg"].height == 900
    np.testing.assert_array_equal(env_kwargs["target_positions"], targets)


def test_build_eval_env_kwargs_routes_env_overrides_into_cfg():
    cfg = Config()
    targets = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)

    env_kwargs = build_eval_env_kwargs(
        cfg,
        targets,
        time_horizon=2048,
        reward_mode="interval_v1",
        capture_size=320,
        cell_colors=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
    )

    assert env_kwargs["cfg"].time_horizon == 2048
    assert env_kwargs["cfg"].reward_mode == "interval_v1"
    assert env_kwargs["capture_size"] == 320
    np.testing.assert_array_equal(
        env_kwargs["cell_colors"],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
    )
    assert "time_horizon" not in env_kwargs
    assert "reward_mode" not in env_kwargs


def test_build_eval_env_kwargs_revalidates_env_overrides():
    cfg = Config()
    targets = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)

    with pytest.raises(ValidationError):
        build_eval_env_kwargs(cfg, targets, reward_interval_multiplier=0.0)


def test_build_eval_env_and_policy_uses_configured_id_features(monkeypatch):
    cfg = Config.model_validate(
        {
            "train": {"device": "cpu"},
            "model": {
                "id_feature_encoder": True,
                "learned_id_dim": 3,
                "comm_enabled": True,
                "comm_msg_dim": 17,
                "cell_encoder": "raw",
                "comm_select_action": True,
            },
        }
    )
    cfg.data.id_features = "embedding"
    targets = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    calls = {}

    class DummyAnnData:
        pass

    def load_spatial_data(path, **kwargs):
        calls["data_loads"] = calls.get("data_loads", 0) + 1
        return DummyAnnData()

    monkeypatch.setattr("magna.eval.load_spatial_data", load_spatial_data)
    monkeypatch.setattr(
        "magna.eval.load_adata_id_features",
        lambda adata, id_features: np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "magna.eval.resolve_id_features",
        lambda loaded_id_features, targets, **_: (
            targets if loaded_id_features is None else loaded_id_features
        ),
    )

    class DummyEnv:
        single_action_space = type("A", (), {"nvec": np.array([2, 2])})()
        mutual_obs_bit = False
        perceived_ratio_obs = True
        self_pos_delta_obs = True

        def __init__(self, **kwargs):
            calls["env_kwargs"] = kwargs

    class DummyPolicy:
        def __init__(self, env, **kwargs):
            calls["policy_kwargs"] = kwargs

        def to(self, device):
            calls["device"] = device
            return self

    monkeypatch.setattr("magna.eval.MultiCellEnv", DummyEnv)
    monkeypatch.setattr("magna.eval.Policy", DummyPolicy)

    _, _, _, _ = build_eval_env_and_policy(cfg, targets)

    assert calls["env_kwargs"]["comm_select_action"] is True
    assert calls["policy_kwargs"]["id_dim"] == 2
    assert calls["policy_kwargs"]["id_feature_encoder"] is True
    assert calls["policy_kwargs"]["learned_id_dim"] == 3
    assert calls["policy_kwargs"]["comm_enabled"] is True
    assert calls["policy_kwargs"]["comm_msg_dim"] == 17
    assert calls["policy_kwargs"]["cell_encoder"] == "raw"
    np.testing.assert_allclose(
        calls["policy_kwargs"]["id_vectors"],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    cfg.data.id_features = None
    build_eval_env_and_policy(cfg, targets)
    assert calls["data_loads"] == 1

    cfg.data.id_features = "embedding"
    cfg.env.reward_mode = "interval_split_matched"
    build_eval_env_and_policy(cfg, targets)
    assert calls["policy_kwargs"]["id_dim"] == 1
    assert calls["policy_kwargs"]["learned_id_dim"] == 0
    np.testing.assert_allclose(
        calls["policy_kwargs"]["id_vectors"], np.ones((3, 1), dtype=np.float32)
    )


def _patch_inference(monkeypatch, env_cls):
    monkeypatch.setattr("magna.eval.MultiCellEnv", env_cls)
    monkeypatch.setattr(
        "magna.eval.pufferlib.pytorch.sample_logits",
        lambda logits: (
            torch.zeros((logits[0].shape[0], len(logits)), dtype=torch.int64),
            None,
            None,
        ),
    )


class _DummyPolicy:
    def forward_eval(self, observations, state):
        logits = [torch.zeros((observations.shape[0], 3)) for _ in range(2)]
        return logits, None


def test_run_policy_inference_keeps_cfg_time_horizon(monkeypatch):
    seen_horizons: list[int] = []

    class DummyCfg:
        def __init__(self, time_horizon=4):
            self.time_horizon = time_horizon

    class DummyEnv:
        def __init__(self, **kwargs):
            self.horizon = int(kwargs["cfg"].time_horizon)
            seen_horizons.append(self.horizon)
            self.step_idx = 0

        def reset(self, seed=0):
            self.step_idx = 0
            return np.zeros((2, 3), dtype=np.float32), []

        def positions(self):
            return np.zeros((1, 2, 2), dtype=np.float32)

        def step(self, actions):
            self.step_idx += 1
            terminals = np.zeros(2, dtype=bool)
            info = []
            if self.horizon > 0 and self.step_idx >= self.horizon:
                terminals[:] = True
                info = [{"error": float(self.step_idx)}]
            return (
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                terminals,
                np.zeros(2, dtype=bool),
                info,
            )

        def close(self):
            pass

    _patch_inference(monkeypatch, DummyEnv)
    out = run_policy_inference(
        env_kwargs={"cfg": DummyCfg(4), "num_agents": 2},
        policy=_DummyPolicy(),
        device="cpu",
        seed=0,
        show_progress=False,
    )
    assert seen_horizons == [4]
    assert out["num_steps"] == 4
    assert out["error_mean"] == 4.0


def test_run_policy_inference_uses_greedy_argmax(monkeypatch):
    action_traces: list[list[list[int]]] = []

    class DummyEnv:
        def __init__(self, **kwargs):
            self.actions: list[list[int]] = []

        def reset(self, seed=0):
            return np.zeros((2, 3), dtype=np.float32), []

        def positions(self):
            return np.zeros((1, 2, 2), dtype=np.float32)

        def step(self, actions):
            self.actions.append(np.asarray(actions, dtype=np.int64).tolist())
            terminals = np.zeros(2, dtype=bool)
            truncations = np.zeros(2, dtype=bool)
            truncations[:] = True
            return (
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                terminals,
                truncations,
                [],
            )

        def close(self):
            action_traces.append(self.actions)

    class DummyPolicy:
        def forward_eval(self, observations, state):
            logits = [
                torch.tensor([[0.1, 2.0, 0.3], [4.0, 1.0, 0.0]]),
                torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
            ]
            return logits, None

    def fail_sample_logits(logits):
        raise AssertionError("sample_logits should not be called in greedy mode")

    monkeypatch.setattr("magna.eval.MultiCellEnv", DummyEnv)
    monkeypatch.setattr(
        "magna.eval.pufferlib.pytorch.sample_logits",
        fail_sample_logits,
    )

    metrics = run_policy_inference(
        env_kwargs={"num_agents": 2},
        policy=DummyPolicy(),
        device="cpu",
        seed=17,
        n_steps=3,
        n_rollouts=1,
        video_path=None,
        collect_errors=False,
        show_progress=False,
        greedy=True,
    )

    assert metrics["num_steps"] == 1
    assert action_traces == [[[[1, 0], [0, 1]]]]


def test_eval_cli_defaults_checkpoint_from_cfg_and_passes_dot_overrides(
    monkeypatch, capsys
):
    calls = {}

    def fake_run_eval(cfg, checkpoint, **kwargs):
        calls.update(cfg=cfg, checkpoint=checkpoint, **kwargs)
        return {
            "target_obsm_key": "spatial_target",
            "initial_obsm_key": "spatial_init",
            "video_path": kwargs["video_path"],
            "metrics": {"error_mean": 0.25},
        }

    monkeypatch.setattr("magna.cli.eval.run_eval", fake_run_eval)
    monkeypatch.setattr("magna.cli.eval.load_yaml_config", lambda path, **_: path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval", "--cfg=configs/grid2d.yaml", "--data.path=data/custom.h5ad"],
    )

    from magna.cli.eval import main as eval_cli

    eval_cli()

    assert calls["cfg"] == "configs/grid2d.yaml"
    assert calls["checkpoint"] == Path("checkpoints/grid2d.pt")
    assert calls["greedy"] is False
    assert calls["flat_overrides"] == {"data.path": "data/custom.h5ad"}
    assert "Target obsm key: spatial_target" in capsys.readouterr().out
