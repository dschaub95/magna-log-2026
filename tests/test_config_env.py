import anndata as ad
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from magna.config import Config, finalize_with_wandb_overrides, load_yaml_config


def test_resolve_defaults_counts_h5ad_backed_and_pt_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ad,
        "read_h5ad",
        lambda path, backed: (
            calls.append((path, backed)) or type("A", (), {"n_obs": 3})()
        ),
    )
    monkeypatch.setattr(
        "magna.data.io.load_pt_coordinates",
        lambda path, mmap: calls.append((path, mmap)) or torch.zeros(3, 2),
    )

    h5ad = Config.model_validate(
        {"data": {"path": "data.h5ad"}}
    ).resolve_dependent_defaults()
    pt = Config.model_validate(
        {"data": {"path": "coordinates.pt"}}
    ).resolve_dependent_defaults()

    assert calls == [("data.h5ad", "r"), ("coordinates.pt", True)]
    assert h5ad.train.minibatch_size == pt.train.minibatch_size


def test_env_size_alone_resolves_width_and_height():
    cfg = Config.model_validate({"env": {"size": 1024}})

    assert cfg.env.size == 1024
    assert cfg.env.width == 1024
    assert cfg.env.height == 1024


def test_draw_obs_graph_defaults_false_in_env_config():
    cfg = Config()
    assert cfg.env.draw_obs_graph is False


def test_draw_obs_graph_can_be_set_via_env_config():
    cfg = Config.model_validate({"env": {"draw_obs_graph": True}})
    assert cfg.env.draw_obs_graph is True


def test_bounded_defaults_true():
    cfg = Config()
    assert cfg.env.bounded is True


def test_unbounded_rejects_boundary_dist_obs():
    with pytest.raises(
        ValidationError, match="boundary_dist_obs requires bounded=True"
    ):
        Config.model_validate({"env": {"bounded": False, "boundary_dist_obs": True}})


def test_unbounded_rejects_boundary_penalty():
    with pytest.raises(ValidationError, match="boundary_penalty requires bounded=True"):
        Config.model_validate({"env": {"bounded": False, "boundary_penalty": -0.1}})


def test_unbounded_rejects_collision_iters():
    with pytest.raises(
        ValidationError, match="collision_iters > 0 requires bounded=True"
    ):
        Config.model_validate({"env": {"bounded": False, "collision_iters": 1}})


def test_draw_perception_radius_defaults_false_in_env_config():
    cfg = Config()
    assert cfg.env.draw_perception_radius is False


def test_draw_perception_radius_can_be_set_via_env_config():
    cfg = Config.model_validate({"env": {"draw_perception_radius": True}})
    assert cfg.env.draw_perception_radius is True


def test_reward_scaler_must_be_positive():
    with pytest.raises(ValueError):
        Config.model_validate({"env": {"reward_scaler": 0}})


def test_env_size_auto_leaves_dimensions_unresolved_until_data_load():
    cfg = Config.model_validate({"env": {"size": "auto"}})

    assert cfg.env.size == "auto"
    assert cfg.env.width is None
    assert cfg.env.height is None


def test_explicit_equal_env_width_and_height_are_preserved():
    cfg = Config.model_validate({"env": {"size": 768, "width": 512, "height": 512}})

    assert cfg.env.size == 768
    assert cfg.env.width == 512
    assert cfg.env.height == 512


def test_explicit_non_square_env_dimensions_fail():
    with pytest.raises(ValueError, match="square arenas"):
        Config.model_validate({"env": {"width": 512, "height": 256}})


def test_old_wandb_style_env_width_height_still_load():
    class DummyRun:
        config = {"env.width": 1536, "env.height": 1536}

    cfg = finalize_with_wandb_overrides(Config(), DummyRun())

    assert cfg.env.size == 768
    assert cfg.env.width == 1536
    assert cfg.env.height == 1536


def test_wandb_sweep_size_override_ignores_backfilled_dimensions():
    class DummyRun:
        config = {
            "env.size": "auto",
            "env.width": 768,
            "env.height": 768,
            "env.depth": 768,
        }

    cfg = finalize_with_wandb_overrides(
        Config(), DummyRun(), override_keys={"env.size"}
    )
    assert (cfg.env.width, cfg.env.height, cfg.env.depth) == (None, None, None)


def test_missing_data_id_features_defaults_none_for_wandb_loads():
    class DummyRun:
        config = {"env.width": 1536, "env.height": 1536}

    cfg = finalize_with_wandb_overrides(Config(), DummyRun())

    assert cfg.data.id_features is None


def test_model_id_feature_encoder_defaults_false():
    cfg = Config()

    assert cfg.model.id_feature_encoder is False
    assert cfg.model.learned_id_dim == 0
    assert cfg.model.learned_id_init == "gaussian"
    assert cfg.model.mask_nontarget_id_features is False
    assert cfg.train.embedding_pretrain_steps == 10_000
    assert cfg.train.embedding_pretrain_progbar is False
    assert cfg.train.embedding_pretrain_learning_rate == pytest.approx(0.01)
    assert cfg.train.embedding_pretrain_min_learning_rate == pytest.approx(1e-4)
    assert cfg.train.embedding_pretrain_weight_decay == pytest.approx(1e-4)
    assert cfg.train.embedding_pretrain_margin == pytest.approx(0.5)
    assert cfg.train.embedding_pretrain_edge_coef == pytest.approx(10_000.0)
    assert cfg.train.embedding_pretrain_order_coef == pytest.approx(1.0)
    assert cfg.model.encoder_layers == 1
    Config.model_validate({"train": {"embedding_pretrain_steps": 10}})


def test_comm_select_action_requires_communication():
    with pytest.raises(ValueError, match="comm_select_action requires comm_enabled"):
        Config.model_validate(
            {"model": {"comm_enabled": False, "comm_select_action": True}}
        )


def test_comm_enabled_requires_use_rnn():
    with pytest.raises(ValueError, match="comm_enabled requires use_rnn"):
        Config.model_validate(
            {"model": {"comm_enabled": True}, "train": {"use_rnn": False}}
        )


def test_mask_nontarget_id_features_requires_target_neighbor_bit():
    with pytest.raises(
        ValueError, match="mask_nontarget_id_features requires target_neighbor_bit"
    ):
        Config.model_validate({"model": {"mask_nontarget_id_features": True}})


def test_missing_env_circular_defaults_false_for_wandb_loads():
    class DummyRun:
        config = {"env.width": 1536, "env.height": 1536}

    cfg = finalize_with_wandb_overrides(Config(), DummyRun())

    assert cfg.env.circular is False


def test_explicit_env_circular_true_survives_wandb_loads():
    class DummyRun:
        config = {"env.width": 1536, "env.height": 1536, "env.circular": True}

    cfg = finalize_with_wandb_overrides(Config(), DummyRun())

    assert cfg.env.circular is True


def test_missing_env_circular_defaults_false_for_yaml_loads(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("env:\n  size: 512\n")

    cfg = load_yaml_config(str(cfg_path))

    assert cfg.env.circular is False


def test_explicit_env_circular_true_survives_yaml_loads(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("env:\n  size: 512\n  circular: true\n")

    cfg = load_yaml_config(str(cfg_path))

    assert cfg.env.circular is True


def test_yaml_env_size_auto_clears_inherited_dimensions(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("env:\n  size: auto\n")

    cfg = load_yaml_config(str(cfg_path))

    assert cfg.env.size == "auto"
    assert cfg.env.width is None
    assert cfg.env.height is None


def test_missing_data_id_features_defaults_none_for_yaml_loads(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("env:\n  size: 512\n")

    cfg = load_yaml_config(str(cfg_path))

    assert cfg.data.id_features is None


def test_train_builds_env_and_checkpoints_evaluated_policy(monkeypatch, tmp_path):
    cfg = Config.model_validate(
        {
            "wandb_mode": "disabled",
            "train": {"device": "cpu", "n_repeat": 1, "generate_rollout_video": False},
            "env": {
                "size": "auto",
                "circular": True,
                "agent_radius": 4,
                "reset_init": 0.5,
                "reset_rotated_target": 0.25,
                "reset_interpolated": 0.25,
            },
            "model": {
                "learned_id_dim": 3,
                "comm_enabled": True,
                "comm_msg_dim": 17,
                "cell_encoder": "raw",
                "comm_select_action": True,
            },
        }
    )

    targets = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]], dtype=np.float32)
    calls = {}

    class DummyAnnData:
        pass

    class DummyRun:
        name = "dummy-run"
        dir = "/tmp/dummy-run"
        id = "dummy-id"

        def log(self, metrics, step):
            calls.setdefault("logged_metrics", {}).update(metrics)

        class ConfigProxy:
            def update(self, values, allow_val_change=False):
                calls["wandb_update"] = {
                    "values": dict(values),
                    "allow_val_change": allow_val_change,
                }

        config = ConfigProxy()

    monkeypatch.setattr(
        "magna.train.init_config_with_wandb_init",
        lambda input_cfg: (input_cfg, DummyRun()),
    )
    monkeypatch.setattr(
        "magna.train.load_spatial_data",
        lambda path, **_: DummyAnnData(),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_cell_positions_and_colors",
        lambda adata, **kwargs: (targets, None, None, None),
    )
    monkeypatch.setattr(
        "magna.train.build_target_neighbors",
        lambda processed_targets, k, **kwargs: (
            np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
            np.ones((3, 2), dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_id_features",
        lambda adata, id_features: np.ones((3, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "magna.train.resolve_id_features",
        lambda loaded_id_features, targets, learned_id_dim: (
            calls.update(learned_id_dim=learned_id_dim) or loaded_id_features
        ),
    )
    monkeypatch.setattr(
        "magna.train.resolve_perception_radius",
        lambda configured_perception_radius, target_neighbor_distances, **kwargs: 42.0,
    )
    monkeypatch.setattr(
        "magna.train._resolve_vector_backend",
        lambda name: "backend",
    )

    class DummyVecEnv:
        driver_env = object()

    def fake_vector_make(env_cls, **kwargs):
        calls["env_kwargs"] = kwargs["env_kwargs"]
        return DummyVecEnv()

    monkeypatch.setattr("magna.train.pufferlib.vector.make", fake_vector_make)

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, device):
            calls["policy_device"] = device
            return self

    def fake_policy(*args, **kwargs):
        calls["policy_kwargs"] = kwargs
        return DummyPolicy()

    monkeypatch.setattr("magna.train.Policy", fake_policy)

    class DummyTrainer:
        total_epochs = 4
        global_step = 0

        def __init__(self, train_cfg, vecenv, policy, logger=None):
            self.uncompiled_policy = policy
            self.stats = {}
            self.config = {"data_dir": tmp_path}
            self.logger = logger
            calls["trainer_init"] = {
                "train_cfg": train_cfg,
                "vecenv": vecenv,
                "policy": policy,
                "logger": logger,
            }

        def print_dashboard(self):
            return None

        def evaluate(self):
            calls["evaluations"] = calls.get("evaluations", 0) + 1
            self.stats = {"error": [1.0]}

        def train(self):
            with torch.no_grad():
                self.uncompiled_policy.weight.add_(1.0)
            return {"environment/error": 1.0 / calls["evaluations"]}

        def close(self):
            return None

    monkeypatch.setattr("magna.train.PuffeRL", DummyTrainer)
    monkeypatch.setattr(
        "magna.train._evaluate_frozen_policy", lambda *args: (4.0, 3.0, 5.0)
    )

    from magna.train import train

    run_id = train(cfg, max_updates=2)["run_id"]

    train_kw = calls["env_kwargs"]
    assert isinstance(train_kw, list) and len(train_kw) == cfg.vector.num_envs
    assert train_kw[0]["cfg"].size == "auto"
    assert train_kw[0]["cfg"].width == 74
    assert train_kw[0]["cfg"].height == 74
    assert train_kw[0]["comm_select_action"] is True
    assert [kw["reset_mode"] for kw in train_kw] == [0] * 32 + [1] * 16 + [2] * 16
    assert all("target_positions" in kw for kw in train_kw)
    assert calls["policy_kwargs"]["id_feature_encoder"] is False
    assert calls["policy_kwargs"]["learned_id_dim"] == 3
    assert calls["policy_kwargs"]["comm_enabled"] is True
    assert calls["policy_kwargs"]["comm_msg_dim"] == 17
    assert calls["policy_kwargs"]["cell_encoder"] == "raw"
    assert calls["learned_id_dim"] == 3
    assert calls["evaluations"] == 2
    best = torch.load(tmp_path / run_id / "model_best.pt", weights_only=True)
    assert best["weight"] == 2
    assert calls["logged_metrics"]["environment/error_eval"] == 4
    assert calls["logged_metrics"]["environment/error_eval_min"] == 3
    assert calls["logged_metrics"]["environment/error_eval_max"] == 5


def _stub_train_deps(monkeypatch, tmp_path, targets):
    class Run:
        name, dir, id = "dummy-run", "/tmp/dummy-run", "dummy-id"

        def log(self, metrics, step):
            return None

        class config:
            @staticmethod
            def update(*a, **k):
                return None

    class Trainer:
        total_epochs, global_step = 1, 0

        def __init__(self, train_cfg, vecenv, policy, logger=None):
            self.uncompiled_policy = policy
            self.stats = {}
            self.config = {"data_dir": tmp_path}
            self.logger = type("L", (), {"run_id": "test_run"})()

        def print_dashboard(self):
            return None

        def evaluate(self):
            self.stats = {"error": [1.0]}

        def train(self):
            return {"environment/error": 1.0}

        def close(self):
            return None

    policy = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        "magna.train.init_config_with_wandb_init", lambda cfg: (cfg, Run())
    )
    monkeypatch.setattr("magna.train.load_spatial_data", lambda *a, **k: object())
    monkeypatch.setattr(
        "magna.train.load_adata_cell_positions_and_colors",
        lambda *a, **k: (targets, None, None, None),
    )
    monkeypatch.setattr(
        "magna.train.build_target_neighbors",
        lambda *a, **k: (
            np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
            np.ones((3, 2), dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_id_features",
        lambda *a, **k: np.ones((3, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "magna.train.resolve_id_features",
        lambda loaded_id_features, **k: loaded_id_features,
    )
    monkeypatch.setattr("magna.train.resolve_perception_radius", lambda *a, **k: 42.0)
    monkeypatch.setattr("magna.train._resolve_vector_backend", lambda name: "backend")
    monkeypatch.setattr(
        "magna.train.pufferlib.vector.make",
        lambda *a, **k: type("V", (), {"driver_env": object()})(),
    )
    monkeypatch.setattr("magna.train.Policy", lambda *a, **k: policy)
    monkeypatch.setattr("magna.train.PuffeRL", Trainer)
    monkeypatch.setattr(
        "magna.train._evaluate_frozen_policy", lambda *a, **k: (1.0, 1.0, 1.0)
    )


def test_train_load_weights_from_checkpoint(monkeypatch, tmp_path):
    cfg = Config.model_validate(
        {
            "wandb_mode": "disabled",
            "train": {
                "device": "cpu",
                "n_repeat": 1,
                "generate_rollout_video": False,
                "data_dir": str(tmp_path),
            },
        }
    )
    targets = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]], dtype=np.float32)
    checkpoint_path = tmp_path / "abc.pt"
    calls = {}
    _stub_train_deps(monkeypatch, tmp_path, targets)
    monkeypatch.setattr(
        "magna.eval.load_policy_weights",
        lambda policy, path, device: calls.update(
            load_policy_weights={"path": path, "device": device}
        ),
    )

    from magna.train import train

    train(cfg, max_updates=1, load_weights_from=checkpoint_path)

    assert calls["load_policy_weights"] == {"path": checkpoint_path, "device": "cpu"}


def test_train_rollout_video_uses_data_color_key_without_coloring_train_vecenv(
    monkeypatch,
):
    cfg = Config.model_validate(
        {
            "wandb_mode": "disabled",
            "data": {"color_key": "cell_type"},
            "train": {"device": "cpu", "n_repeat": 1, "generate_rollout_video": True},
        }
    )

    targets = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]], dtype=np.float32)
    color_values = np.asarray(["B", "A", "B"], dtype=object)
    calls = {}

    class DummyAnnData:
        pass

    class DummyRun:
        name = "dummy-run"
        dir = "/tmp/dummy-run"
        id = "dummy-id"

        class ConfigProxy:
            def update(self, values, allow_val_change=False):
                calls["wandb_update"] = {
                    "values": dict(values),
                    "allow_val_change": allow_val_change,
                }

        config = ConfigProxy()

    monkeypatch.setattr(
        "magna.train.init_config_with_wandb_init",
        lambda input_cfg: (input_cfg, DummyRun()),
    )
    monkeypatch.setattr(
        "magna.train.load_spatial_data",
        lambda path, **_: DummyAnnData(),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_cell_positions_and_colors",
        lambda adata, **kwargs: (targets, None, color_values, None),
    )
    monkeypatch.setattr(
        "magna.train.resolve_env_geometry",
        lambda raw_targets, **kwargs: (raw_targets, 768, 768),
    )
    monkeypatch.setattr(
        "magna.train.build_target_neighbors",
        lambda processed_targets, k, **kwargs: (
            np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
            np.ones((3, 2), dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_id_features",
        lambda adata, id_features: np.ones((3, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "magna.train.resolve_id_features",
        lambda loaded_id_features, targets, learned_id_dim: (
            calls.update(learned_id_dim=learned_id_dim) or loaded_id_features
        ),
    )
    monkeypatch.setattr(
        "magna.train.resolve_perception_radius",
        lambda configured_perception_radius, target_neighbor_distances, **kwargs: 42.0,
    )
    monkeypatch.setattr(
        "magna.train._resolve_vector_backend",
        lambda name: "backend",
    )

    class DummyVecEnv:
        driver_env = object()

    def fake_vector_make(env_cls, **kwargs):
        calls["train_env_kwargs"] = kwargs["env_kwargs"]
        return DummyVecEnv()

    monkeypatch.setattr("magna.train.pufferlib.vector.make", fake_vector_make)

    class DummyPolicy:
        def to(self, device):
            return self

    monkeypatch.setattr("magna.train.Policy", lambda *args, **kwargs: DummyPolicy())

    class DummyTrainer:
        total_epochs = 0
        global_step = 0

        def __init__(self, train_cfg, vecenv, policy, logger=None):
            pass

        def print_dashboard(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("magna.train.PuffeRL", DummyTrainer)
    monkeypatch.setattr(
        "magna.eval.run_policy_inference",
        lambda **kwargs: calls.setdefault("render_env_kwargs", kwargs["env_kwargs"]),
    )

    from magna.train import train

    train(cfg)

    assert "cell_colors" not in calls["train_env_kwargs"]
    render_colors = calls["render_env_kwargs"]["cell_colors"]
    assert render_colors.dtype == np.uint8
    assert render_colors.shape == (3, 3)
    np.testing.assert_array_equal(render_colors[0], render_colors[2])
    assert not np.array_equal(render_colors[0], render_colors[1])
    assert calls["learned_id_dim"] == 0


def test_train_syncs_resolved_auto_geometry_into_wandb_config(monkeypatch):
    cfg = Config.model_validate(
        {
            "wandb_mode": "disabled",
            "train": {"device": "cpu", "n_repeat": 1, "generate_rollout_video": False},
            "env": {"size": "auto", "circular": True, "agent_radius": 1},
        }
    )

    targets = np.array(
        [[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]],
        dtype=np.float32,
    )
    calls = {}

    class DummyAnnData:
        pass

    class DummyRun:
        name = "dummy-run"
        dir = "/tmp/dummy-run"

        class ConfigProxy:
            def update(self, values, allow_val_change=False):
                calls["wandb_update"] = {
                    "values": dict(values),
                    "allow_val_change": allow_val_change,
                }

        config = ConfigProxy()
        id = "dummy-id"

    monkeypatch.setattr(
        "magna.train.init_config_with_wandb_init",
        lambda input_cfg: (input_cfg, DummyRun()),
    )
    monkeypatch.setattr(
        "magna.train.load_spatial_data",
        lambda path, **_: DummyAnnData(),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_cell_positions_and_colors",
        lambda adata, **kwargs: (targets, None, None, None),
    )
    monkeypatch.setattr(
        "magna.train.build_target_neighbors",
        lambda processed_targets, k, **kwargs: (
            np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
            np.ones((3, 2), dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        "magna.train.load_adata_id_features",
        lambda adata, id_features: np.ones((3, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "magna.train.resolve_id_features",
        lambda loaded_id_features, targets, learned_id_dim: (
            calls.update(learned_id_dim=learned_id_dim) or loaded_id_features
        ),
    )
    monkeypatch.setattr(
        "magna.train.resolve_perception_radius",
        lambda configured_perception_radius, target_neighbor_distances, **kwargs: 42.0,
    )
    monkeypatch.setattr(
        "magna.train._resolve_vector_backend",
        lambda name: "backend",
    )

    class DummyVecEnv:
        driver_env = object()

    monkeypatch.setattr(
        "magna.train.pufferlib.vector.make",
        lambda env_cls, **kwargs: DummyVecEnv(),
    )

    class DummyPolicy:
        def to(self, device):
            return self

    monkeypatch.setattr("magna.train.Policy", lambda *args, **kwargs: DummyPolicy())

    class DummyTrainer:
        total_epochs = 0
        global_step = 0

        def __init__(self, train_cfg, vecenv, policy, logger=None):
            return None

        def print_dashboard(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("magna.train.PuffeRL", DummyTrainer)

    from magna.train import train

    train(cfg)

    assert calls["wandb_update"]["values"] == {
        "env.width": 16,
        "env.height": 16,
        "env.depth": 16,
    }
    assert calls["wandb_update"]["allow_val_change"] is True
    assert calls["learned_id_dim"] == 0
