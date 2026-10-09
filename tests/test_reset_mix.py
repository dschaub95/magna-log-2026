import numpy as np
import pufferlib
import pytest
from pydantic import ValidationError

from magna.config import Config, EnvConfig, resolve_reset_mix_counts
from magna.env import MultiCellEnv


def test_resolve_reset_mix_ratios_third_each():
    assert resolve_reset_mix_counts(1 / 3, 1 / 3, 1 / 3, 64) == (22, 21, 21)


def test_resolve_reset_mix_absolute():
    assert resolve_reset_mix_counts(22, 21, 21, 64) == (22, 21, 21)


def test_resolve_reset_mix_default_all_init():
    assert resolve_reset_mix_counts(1, 0, 0, 64) == (64, 0, 0)


def test_resolve_reset_mix_rejects_bad_sum():
    with pytest.raises(ValueError, match="must sum to 1"):
        resolve_reset_mix_counts(0.5, 0.2, 0.1, 64)


def test_resolve_reset_mix_requires_init_env():
    with pytest.raises(ValueError, match="reset_init count must be >= 1"):
        resolve_reset_mix_counts(0, 0.5, 0.5, 64)


def test_config_validates_reset_mix():
    cfg = Config.model_validate(
        {
            "env": {
                "reset_init": 22,
                "reset_rotated_target": 21,
                "reset_interpolated": 21,
            },
            "vector": {"num_envs": 64},
        }
    )
    assert cfg.reset_mix_counts() == (22, 21, 21)


def test_config_rejects_zero_init_bucket():
    with pytest.raises(ValidationError, match="reset_init count must be >= 1"):
        Config.model_validate(
            {
                "env": {
                    "reset_init": 0,
                    "reset_rotated_target": 32,
                    "reset_interpolated": 32,
                },
                "vector": {"num_envs": 64},
            }
        )


def _square_targets(n: int = 4, size: int = 64) -> np.ndarray:
    g = int(np.sqrt(n))
    xs, ys = np.meshgrid(np.linspace(16, size - 16, g), np.linspace(16, size - 16, g))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)


def _neighbors(n: int, k: int = 1):
    idx = np.arange(n, dtype=np.int32)
    neighbors = np.stack([(idx + i + 1) % n for i in range(k)], axis=1)
    distances = np.full((n, k), 8.0, dtype=np.float32)
    return neighbors, distances


def _make_env(
    *,
    reset_mode: int,
    targets: np.ndarray,
    spatial_dim: int | None = None,
    reuse_initial_positions: bool = False,
    fit_init_to_target_bbox: bool = False,
    **kwargs,
):
    targets = np.asarray(targets, dtype=np.float32)
    spatial_dim = targets.shape[-1] if spatial_dim is None else spatial_dim
    n = len(targets)
    neighbors, distances = _neighbors(n)
    kw = {
        "cfg": EnvConfig(
            size=64,
            k=1,
            collision_grid_size=8,
            perception_radius=64,
            lam_nbr_reward=0,
            max_speed=2,
            time_horizon=1,
            log_interval=1,
            reuse_initial_positions=reuse_initial_positions,
            fit_init_to_target_bbox=fit_init_to_target_bbox,
        ),
        "num_agents": n,
        "coordinate_scale": 1.0,
        "spatial_dim": spatial_dim,
        "target_neighbors": neighbors,
        "target_neighbor_distances": distances,
        "reset_mode": reset_mode,
        "track_error_range": True,
    }
    if reset_mode in (1, 2) or fit_init_to_target_bbox:
        kw["target_positions"] = targets
    kw.update(kwargs)
    return MultiCellEnv(**kw)


def _pairwise_d2(pos: np.ndarray) -> np.ndarray:
    tri = np.triu_indices(len(pos), 1)
    return np.sort(np.sum((pos[:, None] - pos[None]) ** 2, axis=-1)[tri])


def test_rotated_target_preserves_pairwise_distances_2d():
    targets = _square_targets()
    env = _make_env(reset_mode=1, targets=targets)
    try:
        env.reset(seed=0)
        got = env.positions()[0]
        np.testing.assert_allclose(_pairwise_d2(got), _pairwise_d2(targets), rtol=1e-4)
    finally:
        env.close()


def test_rotated_target_preserves_pairwise_distances_3d():
    targets = _square_targets()
    t3 = np.column_stack([targets, np.linspace(20, 44, len(targets), dtype=np.float32)])
    env = _make_env(reset_mode=1, targets=t3, spatial_dim=3)
    try:
        env.reset(seed=1)
        got = env.positions()[0]
        np.testing.assert_allclose(_pairwise_d2(got), _pairwise_d2(t3), rtol=1e-4)
        assert not np.allclose(got, t3, atol=1e-3)
    finally:
        env.close()


def test_interpolated_reset_is_finite_in_arena():
    targets = _square_targets(n=9)
    env = _make_env(reset_mode=2, targets=targets)
    try:
        env.reset(seed=0)
        pos = env.positions()[0]
        assert np.isfinite(pos).all()
        assert pos.min() >= 0.0
        assert pos.max() <= 64.0
    finally:
        env.close()


@pytest.mark.parametrize("reset_mode", [1, 2])
def test_generated_reset_modes_reuse_initial_positions(reset_mode):
    targets = _square_targets()
    env = _make_env(
        reset_mode=reset_mode, targets=targets, reuse_initial_positions=True
    )
    try:
        env.reset(seed=0)
        first = env.positions()[0].copy()
        env.reset(seed=1)
        np.testing.assert_array_equal(env.positions()[0], first)
    finally:
        env.close()


def test_reset_mode_requires_target_positions():
    targets = _square_targets()
    neighbors, distances = _neighbors(len(targets))
    with pytest.raises(pufferlib.APIUsageError, match="target_positions"):
        MultiCellEnv(
            EnvConfig(size=64, k=1, perception_radius=64, time_horizon=1),
            num_agents=len(targets),
            coordinate_scale=1.0,
            target_neighbors=neighbors,
            target_neighbor_distances=distances,
            reset_mode=1,
        )


def test_fit_init_to_target_bbox_matches_limiting_axes():
    xs, ys = np.meshgrid(np.linspace(10, 54, 4), np.linspace(24, 40, 3))
    targets = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    env = _make_env(reset_mode=0, targets=targets, fit_init_to_target_bbox=True)
    try:
        env.reset(seed=0)
        pos = env.positions()[0]
        t_lo, t_hi = targets.min(0), targets.max(0)
        p_lo, p_hi = pos.min(0), pos.max(0)
        np.testing.assert_allclose(0.5 * (p_lo + p_hi), 0.5 * (t_lo + t_hi), atol=1e-3)
        t_span, p_span = t_hi - t_lo, p_hi - p_lo
        assert np.all(p_span <= t_span + 1e-3)
        assert np.min(np.abs(p_span - t_span)) < 1e-3
    finally:
        env.close()


def test_fit_init_to_target_bbox_ignores_degenerate_axes():
    targets = np.array([[16.0, 32.0], [32.0, 32.0], [48.0, 32.0]], dtype=np.float32)
    env = _make_env(reset_mode=0, targets=targets, fit_init_to_target_bbox=True)
    try:
        env.reset(seed=0)
        pos = env.positions()[0]
        np.testing.assert_allclose(pos[:, 1], 32.0, atol=1e-3)
        np.testing.assert_allclose(pos[:, 0].min(), 16.0, atol=1e-3)
        np.testing.assert_allclose(pos[:, 0].max(), 48.0, atol=1e-3)
        assert len(np.unique(np.round(pos, 4), axis=0)) > 1
    finally:
        env.close()


def test_init_mode_logs_error_rotated_does_not():
    targets = _square_targets()
    init_env = _make_env(reset_mode=0, targets=targets)
    rot_env = _make_env(reset_mode=1, targets=targets)
    try:
        init_env.reset(seed=0)
        rot_env.reset(seed=0)
        action = np.tile(np.array([[4, 5]], dtype=np.int32), (len(targets), 1))
        _, _, _, _, init_info = init_env.step(action)
        _, _, _, _, rot_info = rot_env.step(action)
        assert init_info and "error" in init_info[0]
        assert rot_info and "error" not in rot_info[0]
        assert "episode_return" in rot_info[0]
    finally:
        init_env.close()
        rot_env.close()
