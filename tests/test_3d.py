import numpy as np
import pufferlib
import pytest
import torch

from magna.config import EnvConfig
from magna.data.processing import (
    build_target_neighbor_pair_distances,
    build_target_neighbors,
    resolve_env_geometry,
)
from magna.env import MultiCellEnv
from magna.policy import Policy
from magna.utils import coords_to_colors


def test_3d_default_init_varies_z():
    num_agents = 16
    neighbors = np.roll(np.arange(num_agents, dtype=np.int32), -1).reshape(-1, 1)
    env = MultiCellEnv(
        EnvConfig(
            size=64,
            k=1,
            collision_grid_size=8,
            perception_radius=64,
            lam_nbr_reward=0,
            max_speed=2,
        ),
        num_agents=num_agents,
        coordinate_scale=1.0,
        spatial_dim=3,
        target_neighbors=neighbors,
        target_neighbor_distances=np.full((num_agents, 1), 2.0, dtype=np.float32),
        initial_positions=None,
    )
    try:
        env.reset(seed=0)
        z = env.positions()[0, :, 2]
        assert np.ptp(z) > 1.0
        assert abs(float(z.mean()) - 32.0) < 2.0
    finally:
        env.close()


def test_gaussian_reset_distribution_resamples_geometry():
    num_agents = 8
    neighbors = np.roll(np.arange(num_agents, dtype=np.int32), -1).reshape(-1, 1)
    env = MultiCellEnv(
        EnvConfig(
            size=64,
            k=1,
            collision_grid_size=8,
            perception_radius=64,
            lam_nbr_reward=0,
            max_speed=2,
            time_horizon=1,
            reset_distribution="gaussian",
        ),
        num_agents=num_agents,
        coordinate_scale=4.0,
        spatial_dim=3,
        target_neighbors=neighbors,
        target_neighbor_distances=np.full((num_agents, 1), 2.0, dtype=np.float32),
    )
    try:
        env.reset(seed=0)
        first = env.positions()[0].copy()
        env.step(np.tile(np.array([[4, 4, 5]], dtype=np.int32), (num_agents, 1)))
        second = env.positions()[0]
        assert np.linalg.norm(first - 32.0, axis=1).max() <= 31.0 * 0.88 + 1e-4
        assert np.linalg.norm(second - 32.0, axis=1).max() <= 31.0 * 0.88 + 1e-4
        tri = np.triu_indices(num_agents, 1)
        first_d2 = np.sort(np.sum((first[:, None] - first[None]) ** 2, axis=-1)[tri])
        second_d2 = np.sort(np.sum((second[:, None] - second[None]) ** 2, axis=-1)[tri])
        assert second_d2 != pytest.approx(first_d2)
    finally:
        env.close()


def test_3d_positions_neighbors_and_movement(make_test_env):
    positions = np.array(
        [[20.0, 20.0, 20.0], [20.0, 20.0, 24.0], [28.0, 20.0, 20.0]],
        dtype=np.float32,
    )
    neighbors, distances, _ = build_target_neighbors(positions, 1)
    np.testing.assert_allclose(distances[:, 0], [4.0, 4.0, 8.0])

    env = make_test_env(
        initial_positions=positions,
        target_neighbors=neighbors,
        target_neighbor_distances=distances,
        spatial_dim=3,
        depth=64,
        collision_iters=0,
        density_grad_obs=False,
        self_pos_delta_obs=False,
    )
    try:
        env.reset()
        before = env.positions()
        env.step(np.tile([4, 8, 3], (3, 1)))
        after = env.positions()
        assert before.shape == (1, 3, 3)
        assert env.single_action_space.shape == (3,)
        assert env.cell_obs_dim == 5
        assert env.neighbor_cell_start == 3
        assert np.all(after[0, :, 2] > before[0, :, 2])
    finally:
        env.close()


def test_3d_matched_signature(make_test_env):
    target = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 6]], dtype=np.float32)
    neighbors, distances, _ = build_target_neighbors(target, 3)
    env = make_test_env(
        num_agents=4,
        k=3,
        agent_radius=1,
        collision_iters=0,
        spatial_dim=3,
        depth=64,
        perception_radius=16,
        reward_mode="interval_split_matched",
        time_horizon=1,
        log_interval=1,
        target_neighbors=neighbors,
        target_neighbor_distances=distances,
        target_neighbor_pair_distances=build_target_neighbor_pair_distances(
            target, neighbors
        ),
        initial_positions=target + 20,
    )
    try:
        env.reset()
        _, rewards, _, _, info = env.step(np.tile([4, 4, 5], (4, 1)))
        assert np.isfinite(rewards).all()
        assert info[0]["error"] == pytest.approx(0.0, abs=1e-5)
    finally:
        env.close()


@pytest.mark.parametrize("local_frame_obs", [False, True])
def test_3d_policy_forward(make_test_env, local_frame_obs):
    env = make_test_env(
        initial_positions=np.array(
            [[20, 20, 20], [24, 20, 24], [28, 20, 28]], dtype=np.float32
        ),
        spatial_dim=3,
        depth=64,
        local_frame_obs=local_frame_obs,
    )
    try:
        obs, _ = env.reset()
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            hidden_size=16,
        )
        logits, _ = policy.forward_eval(
            torch.as_tensor(obs), {"lstm_h": None, "lstm_c": None}
        )
        assert [head.shape for head in logits] == [(3, 9), (3, 9), (3, 6)]
    finally:
        env.close()


def test_3d_rejects_delaunay_neighbors():
    with pytest.raises(ValueError, match="only supported for 2D"):
        build_target_neighbors(
            np.zeros((4, 3), dtype=np.float32), 1, neighbor_graph="delaunay_topk"
        )


def test_3d_ball_requires_equal_depth(make_test_env):
    with pytest.raises(pufferlib.APIUsageError, match="equal width, height, and depth"):
        make_test_env(
            initial_positions=np.zeros((3, 3), dtype=np.float32),
            spatial_dim=3,
            circular=True,
            depth=32,
        )


def test_3d_ball_clips_xyz_and_collision_stays_xy(make_test_env):
    positions = np.array(
        [[63.0, 63.0, 63.0], [20.0, 20.0, 40.0], [20.0, 20.0, 10.0]],
        dtype=np.float32,
    )
    env = make_test_env(
        initial_positions=positions,
        spatial_dim=3,
        depth=64,
        circular=True,
        recenter_on_step=False,
    )
    try:
        env.reset()
        np.testing.assert_array_equal(
            env.positions()[0, 1, :2], env.positions()[0, 2, :2]
        )
        clipped = env.positions()[0]
        assert np.linalg.norm(clipped[0] - 32.0) <= 31.0 + 1e-5
        env.step(np.tile([4, 4, 2], (3, 1)))
        assert np.linalg.norm(env.positions()[0, 1, :2] - env.positions()[0, 2, :2]) > 0
    finally:
        env.close()


def test_3d_unbounded_preserves_outside_initial_positions(make_test_env):
    positions = np.array(
        [[63.0, 63.0, 63.0], [20.0, 20.0, 40.0], [20.0, 20.0, 10.0]],
        dtype=np.float32,
    )
    env = make_test_env(
        initial_positions=positions,
        spatial_dim=3,
        depth=64,
        circular=True,
        bounded=False,
        recenter_on_step=False,
    )
    try:
        env.reset()
        np.testing.assert_allclose(env.positions()[0, 0], positions[0], atol=1e-5)
    finally:
        env.close()


def test_3d_geometry_uses_depth_center():
    positions, _, _ = resolve_env_geometry(
        np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
        coordinate_scale=1,
        width=100,
        height=100,
        depth=200,
        auto_size_buffer=1,
    )
    np.testing.assert_allclose(positions.mean(axis=0), [50, 50, 100])


def test_3d_capture_returns_frame(make_test_env, capture_available):
    env = make_test_env(
        capture_size=96,
        spatial_dim=3,
        depth=64,
        initial_positions=np.array(
            [[20, 20, 20], [24, 20, 24], [28, 20, 28]], dtype=np.float32
        ),
    )
    try:
        frame = env.capture()
        assert frame.shape == (96, 96, 4)
        assert frame.std() > 5
    finally:
        env.close()


def test_3d_default_colors_include_depth():
    colors = coords_to_colors(np.array([[1, 1, 1], [1, 1, 2]], dtype=np.float32))
    np.testing.assert_array_equal(colors[:, :2], np.zeros((2, 2), dtype=np.uint8))
    assert colors[0, 2] < colors[1, 2]


def test_3d_density_log_normalization_includes_z(make_test_env):
    kwargs = {
        "initial_positions": np.array(
            [[20.0, 20.0, 20.0], [24.0, 25.0, 26.0], [28.0, 20.0, 24.0]],
            dtype=np.float32,
        ),
        "spatial_dim": 3,
        "depth": 64,
        "circular": False,
        "collision_iters": 0,
        "density_grad_obs": True,
        "self_pos_delta_obs": False,
        "mutual_obs_bit": False,
    }
    raw = make_test_env(**kwargs, density_grad_log_norm=False)
    logged = make_test_env(**kwargs, density_grad_log_norm=True)
    try:
        raw_obs, _ = raw.reset()
        logged_obs, _ = logged.reset()
        raw_norm = np.linalg.norm(raw_obs[:, -3:], axis=1)
        logged_norm = np.linalg.norm(logged_obs[:, -3:], axis=1)
        np.testing.assert_allclose(logged_norm, np.log1p(raw_norm), atol=1e-7)
    finally:
        raw.close()
        logged.close()
