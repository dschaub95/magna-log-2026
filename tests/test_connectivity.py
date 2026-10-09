import numpy as np
import pytest


def test_reset_rejects_disconnected_initial_positions(make_test_env):
    env = make_test_env(
        enforce_connected=True,
        num_agents=3,
        perception_radius=4.0,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.0], [2.0], [2.0]], dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [22.0, 20.0], [40.0, 40.0]], dtype=np.float32
        ),
    )

    with pytest.raises(ValueError, match="connected component"):
        env.reset()


def test_connectivity_enforcement_can_be_disabled(make_test_env):
    env = make_test_env(
        enforce_connected=False,
        num_agents=3,
        perception_radius=4.0,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.0], [2.0], [2.0]], dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [22.0, 20.0], [40.0, 40.0]], dtype=np.float32
        ),
    )

    try:
        env.reset()
    finally:
        env.close()


def test_step_rolls_back_disconnect_and_rebuilds_observations(make_test_env):
    env = make_test_env(
        enforce_connected=True,
        num_agents=3,
        perception_radius=3.0,
        agent_radius=1,
        collision_iters=0,
        recenter_on_step=False,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.5], [2.5], [2.5]], dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [22.5, 20.0], [25.0, 20.0]], dtype=np.float32
        ),
    )

    try:
        initial_obs = env.reset()[0].copy()
        obs, rewards, _, _, _ = env.step(
            np.array([[4, 5], [4, 5], [4, 4]], dtype=np.int32)
        )
        np.testing.assert_allclose(obs, initial_obs)
        np.testing.assert_allclose(rewards, 0.0)
    finally:
        env.close()


def test_disconnect_penalty_is_added_after_rollback(make_test_env):
    env = make_test_env(
        enforce_connected=True,
        disconnect_penalty=-0.25,
        num_agents=3,
        perception_radius=3.0,
        agent_radius=1,
        collision_iters=0,
        recenter_on_step=False,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.5], [2.5], [2.5]], dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [22.5, 20.0], [25.0, 20.0]], dtype=np.float32
        ),
    )

    try:
        env.reset()
        _, rewards, _, _, _ = env.step(
            np.array([[4, 5], [4, 5], [4, 4]], dtype=np.int32)
        )
        np.testing.assert_allclose(rewards, -0.25)
    finally:
        env.close()


def test_reward_scaler_applies_to_disconnect_penalty(make_test_env):
    env = make_test_env(
        enforce_connected=True,
        disconnect_penalty=-0.25,
        reward_scaler=2.0,
        num_agents=3,
        perception_radius=3.0,
        collision_iters=0,
        recenter_on_step=False,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.full((3, 1), 2.5, dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [22.5, 20.0], [25.0, 20.0]], dtype=np.float32
        ),
    )

    try:
        env.reset()
        _, rewards, _, _, _ = env.step(
            np.array([[4, 5], [4, 5], [4, 4]], dtype=np.int32)
        )
        np.testing.assert_allclose(rewards, -0.5)
    finally:
        env.close()


def test_terminate_on_disconnect_ends_episode(make_test_env):
    env = make_test_env(
        enforce_connected=True,
        terminate_on_disconnect=True,
        disconnect_penalty=-0.25,
        num_agents=3,
        perception_radius=3.0,
        agent_radius=1,
        collision_iters=0,
        recenter_on_step=False,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.5], [2.5], [2.5]], dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [22.5, 20.0], [25.0, 20.0]], dtype=np.float32
        ),
    )

    try:
        initial_obs = env.reset()[0].copy()
        obs, rewards, terminals, _, _ = env.step(
            np.array([[4, 5], [4, 5], [4, 4]], dtype=np.int32)
        )
        assert np.all(terminals)
        np.testing.assert_allclose(obs, initial_obs)
        np.testing.assert_allclose(rewards, -0.25)
    finally:
        env.close()
