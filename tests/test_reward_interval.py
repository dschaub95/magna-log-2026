import numpy as np
import pufferlib
import pytest
from pydantic import ValidationError

from magna.data.processing import build_target_neighbor_pair_distances
from magna.env import MultiCellEnv


def assert_logged_extreme(info, pending_rewards: list[np.ndarray]) -> float:
    rewards = np.concatenate(pending_rewards)
    expected = float(rewards[np.argmax(np.abs(rewards))])
    extreme = info[0]["environment/extreme_reward"]
    assert extreme == pytest.approx(expected)
    pending_rewards.clear()
    return extreme


def _build_env(
    make_test_env,
    *,
    reward_mode: str,
    reward_interval_multiplier: float = 2.0,
    max_speed: float = 2.0,
    log_interval: int = 128,
    enforce_connected: bool = True,
) -> MultiCellEnv:
    return make_test_env(
        width=64,
        height=64,
        agent_radius=1,
        collision_iters=1,
        collision_grid_size=2,
        k=1,
        perception_radius=10.0,
        lam_nbr_reward=0.0,
        reward_mode=reward_mode,
        reward_interval_multiplier=reward_interval_multiplier,
        max_speed=max_speed,
        time_horizon=32,
        reuse_initial_positions=True,
        enforce_connected=enforce_connected,
        log_interval=log_interval,
        num_agents=2,
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.0], [2.0]], dtype=np.float32),
        initial_positions=np.array([[20.0, 20.0], [22.0, 20.0]], dtype=np.float32),
    )


def _step_speed(env: MultiCellEnv, speed0: int, speed1: int) -> np.ndarray:
    actions = np.array([[4, speed0], [4, speed1]], dtype=np.int32)
    _, rewards, _, _, _ = env.step(actions)
    return rewards.copy()


def _build_env_interval_split_three_agents(make_test_env) -> MultiCellEnv:
    return make_test_env(
        width=64,
        height=64,
        agent_radius=1,
        collision_iters=1,
        collision_grid_size=2,
        k=2,
        perception_radius=20.0,
        lam_nbr_reward=0.0,
        reward_mode="interval_split_cached",
        reward_interval_multiplier=1.0,
        max_speed=2.0,
        time_horizon=32,
        reuse_initial_positions=True,
        log_interval=128,
        num_agents=3,
        target_neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        target_neighbor_distances=np.array(
            [[2.0, 2.0], [2.0, np.sqrt(8.0)], [2.0, np.sqrt(8.0)]], dtype=np.float32
        ),
        initial_positions=np.array(
            [[20.0, 20.0], [22.0, 20.0], [20.0, 22.0]], dtype=np.float32
        ),
    )


def _build_env_interval_split_matched_line(
    make_test_env, lam_nbr_reward: float
) -> MultiCellEnv:
    return make_test_env(
        width=64,
        height=64,
        agent_radius=1,
        collision_iters=0,
        collision_grid_size=2,
        k=2,
        perception_radius=20.0,
        lam_nbr_reward=lam_nbr_reward,
        reward_mode="interval_split_matched",
        reward_interval_multiplier=1.0,
        max_speed=2.0,
        time_horizon=32,
        reuse_initial_positions=True,
        enforce_connected=False,
        log_interval=128,
        num_agents=3,
        target_neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        target_neighbor_distances=np.array(
            [[2.0, 5.0], [2.0, 3.0], [5.0, 3.0]], dtype=np.float32
        ),
        initial_positions=np.array(
            [[25.0, 20.0], [22.0, 20.0], [20.0, 20.0]], dtype=np.float32
        ),
    )


def _build_env_interval_split_matched_visibility(make_test_env) -> MultiCellEnv:
    return make_test_env(
        width=64,
        height=64,
        agent_radius=1,
        collision_iters=0,
        collision_grid_size=2,
        k=1,
        perception_radius=4.0,
        lam_nbr_reward=1.0,
        reward_mode="interval_split_matched",
        reward_interval_multiplier=1.0,
        max_speed=2.0,
        time_horizon=32,
        reuse_initial_positions=True,
        enforce_connected=False,
        log_interval=128,
        num_agents=2,
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[2.0], [2.0]], dtype=np.float32),
        initial_positions=np.array([[20.0, 20.0], [25.0, 20.0]], dtype=np.float32),
    )


def _step_speed_three(
    env: MultiCellEnv, speed0: int, speed1: int, speed2: int
) -> np.ndarray:
    actions = np.array([[4, speed0], [4, speed1], [4, speed2]], dtype=np.int32)
    _, rewards, _, _, _ = env.step(actions)
    return rewards.copy()


def _build_hidden_target_env(make_test_env, reward_mode):
    return make_test_env(
        agent_radius=1,
        collision_iters=0,
        k=1,
        perception_radius=10.0,
        lam_nbr_reward=0.0,
        reward_mode=reward_mode,
        reward_interval_multiplier=0.5,
        max_speed=2.0,
        time_horizon=32,
        reuse_initial_positions=True,
        recenter_on_step=False,
        enforce_connected=False,
        num_agents=3,
        target_neighbors=np.array([[1], [0], [0]], dtype=np.int32),
        target_neighbor_distances=np.full((3, 1), 8.0, dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [26.5, 20.0], [28.0, 20.0]], dtype=np.float32
        ),
        initial_headings=np.array([0.0, 0.0, np.pi], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("reward_mode", "expected"),
    [("interval_v1_cached", 2.0), ("interval_split_cached", 1.0)],
)
def test_hidden_target_improvement_is_paid_on_reappearance(
    make_test_env, reward_mode, expected
):
    env = _build_hidden_target_env(make_test_env, reward_mode)
    try:
        env.reset(seed=0)
        assert _step_speed_three(env, 2, 2, 2)[0] == pytest.approx(0.0)
        rewards = [_step_speed_three(env, 2, 4, 4)[0]]
        rewards.extend(_step_speed_three(env, 2, 5, 2)[0] for _ in range(8))

        assert rewards[0] == pytest.approx(0.0)
        assert [reward for reward in rewards if reward] == pytest.approx([expected])
    finally:
        env.close()


@pytest.mark.parametrize(
    ("reward_mode", "expected"),
    [
        ("interval_v1", 2.0),
        ("interval_split", 1.0),
        ("interval_v1_cached", 2.0),
        ("interval_split_cached", 1.0),
    ],
)
def test_first_visibility_uses_missing_baseline(make_test_env, reward_mode, expected):
    env = make_test_env(
        agent_radius=1,
        collision_iters=0,
        k=2,
        perception_radius=10.0,
        lam_nbr_reward=0.0,
        reward_mode=reward_mode,
        reward_interval_multiplier=1.0,
        max_speed=2.0,
        reuse_initial_positions=True,
        enforce_connected=False,
        num_agents=3,
        target_neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        target_neighbor_distances=np.array(
            [[2.0, 8.0], [2.0, 6.0], [8.0, 6.0]], dtype=np.float32
        ),
        initial_positions=np.array(
            [[20.0, 20.0], [22.0, 20.0], [31.0, 20.0]], dtype=np.float32
        ),
        initial_headings=np.array([0.0, 0.0, np.pi], dtype=np.float32),
    )
    try:
        env.reset(seed=0)
        assert _step_speed_three(env, 2, 2, 2)[0] == pytest.approx(0.0)
        assert _step_speed_three(env, 2, 4, 4)[0] == pytest.approx(expected)
    finally:
        env.close()


def test_reward_mode_validation(make_test_env):
    with pytest.raises((pufferlib.APIUsageError, ValidationError)):
        _build_env(make_test_env, reward_mode="unknown_mode")
    with pytest.raises((pufferlib.APIUsageError, ValidationError)):
        _build_env(
            make_test_env,
            reward_mode="interval_v1_cached",
            reward_interval_multiplier=0.0,
        )
    with pytest.raises((pufferlib.APIUsageError, ValidationError)):
        _build_env(
            make_test_env,
            reward_mode="interval_split_cached",
            reward_interval_multiplier=0.0,
        )


def test_interval_reward_crossings_and_hidden_distance_freezing(make_test_env):
    env = _build_env(
        make_test_env, reward_mode="interval_v1_cached", enforce_connected=False
    )
    try:
        env.reset(seed=0)

        # Warmup step initializes previous reward state.
        rewards = _step_speed(env, 2, 2)
        assert rewards[0] == pytest.approx(0.0)

        # Increase cumulative error from bin 0 -> 1.
        rewards = _step_speed(env, 0, 4)
        assert rewards[0] == pytest.approx(-1.0)
        assert rewards[1] == pytest.approx(-1.0)

        # Increase cumulative error from bin 1 -> 2.
        rewards = _step_speed(env, 0, 4)
        assert rewards[0] == pytest.approx(-1.0)
        assert rewards[1] == pytest.approx(-1.0)

        # While hidden, the last visible distance bin remains frozen.
        rewards = _step_speed(env, 0, 4)
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] == pytest.approx(0.0)

        # Stop, then move back toward target distance to cross bins downward (+ rewards).
        rewards = _step_speed(env, 5, 5)
        assert rewards[0] == pytest.approx(0.0)
        rewards = _step_speed(env, 4, 0)
        assert rewards[0] == pytest.approx(0.0)
        rewards = _step_speed(env, 4, 0)
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(1.0)
        rewards = _step_speed(env, 4, 0)
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(1.0)
        # Staying in first bin has no transition reward.
        rewards = _step_speed(env, 4, 0)
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] == pytest.approx(0.0)
    finally:
        env.close()


def test_interval_multiplier_changes_bin_crossing_sensitivity(make_test_env):
    env_fast_bins = _build_env(
        make_test_env,
        reward_mode="interval_v1_cached",
        reward_interval_multiplier=1.0,
    )
    env_slow_bins = _build_env(
        make_test_env,
        reward_mode="interval_v1_cached",
        reward_interval_multiplier=5.0,
    )
    try:
        env_fast_bins.reset(seed=0)
        env_slow_bins.reset(seed=0)

        # Warmup in both envs.
        _step_speed(env_fast_bins, 2, 2)
        _step_speed(env_slow_bins, 2, 2)

        # One outward move crosses more interval boundaries when multiplier is smaller.
        # A larger multiplier widens both the first and subsequent bins.
        # - interval 2.0 (mult=1) => -2 reward
        # - first bin max 5.0 (mult=5) => 0 reward
        rewards_fast = _step_speed(env_fast_bins, 0, 4)
        rewards_slow = _step_speed(env_slow_bins, 0, 4)
        assert rewards_fast[0] == pytest.approx(-2.0)
        assert rewards_slow[0] == pytest.approx(0.0)
    finally:
        env_fast_bins.close()
        env_slow_bins.close()


def test_interval_multiplier_scales_first_bin(make_test_env):
    env = _build_env(
        make_test_env,
        reward_mode="interval_v1_cached",
        reward_interval_multiplier=2.0,
        max_speed=0.1,
    )
    try:
        env.reset(seed=0)

        # Warmup state.
        _step_speed(env, 2, 2)

        # Outward move changes distance by ~0.2.
        # First bin upper bound is multiplier(2) * valid_targets(1) = 2.
        # The ~0.2 cumulative error therefore remains in bin 0.
        rewards = _step_speed(env, 0, 4)
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] == pytest.approx(0.0)
    finally:
        env.close()


def test_interval_split_per_neighbor_bin_transitions_use_mean_scaling(make_test_env):
    env = _build_env_interval_split_three_agents(make_test_env)
    try:
        env.reset(seed=0)

        # Warmup state.
        rewards = _step_speed_three(env, 2, 2, 2)
        assert rewards[0] == pytest.approx(0.0)

        # Move only target-neighbor 1 away from agent 0: one slot crosses bin 0->1.
        # Split mode averages per-slot transitions over k=2, so reward is -0.5.
        rewards = _step_speed_three(env, 2, 4, 2)
        assert rewards[0] == pytest.approx(-0.5)
    finally:
        env.close()


def test_interval_split_matched_uses_sorted_distance_roles(make_test_env):
    env = _build_env_interval_split_matched_line(make_test_env, lam_nbr_reward=0.0)
    try:
        env.reset(seed=0)

        rewards = _step_speed_three(env, 2, 2, 2)
        assert rewards == pytest.approx(np.zeros(3))

        rewards = _step_speed_three(env, 4, 2, 2)
        assert rewards == pytest.approx(np.array([-1.0, -0.5, -0.5]))
    finally:
        env.close()


def test_interval_split_matched_blends_visibility_and_distance_weights(make_test_env):
    env_distance = _build_env_interval_split_matched_line(
        make_test_env, lam_nbr_reward=0.0
    )
    env_neighbor = _build_env_interval_split_matched_line(
        make_test_env, lam_nbr_reward=1.0
    )
    try:
        env_distance.reset(seed=0)
        env_neighbor.reset(seed=0)
        _step_speed_three(env_distance, 2, 2, 2)
        _step_speed_three(env_neighbor, 2, 2, 2)

        rewards_distance = _step_speed_three(env_distance, 4, 2, 2)
        rewards_neighbor = _step_speed_three(env_neighbor, 4, 2, 2)
        assert rewards_distance == pytest.approx(np.array([-1.0, -0.5, -0.5]))
        assert rewards_neighbor == pytest.approx(np.zeros(3))
    finally:
        env_distance.close()
        env_neighbor.close()


def test_interval_split_matched_rewards_visibility_transitions(make_test_env):
    env = _build_env_interval_split_matched_visibility(make_test_env)
    try:
        env.reset(seed=0)
        assert _step_speed(env, 2, 2) == pytest.approx(np.zeros(2))
        assert _step_speed(env, 4, 2) == pytest.approx(np.ones(2))
    finally:
        env.close()


def test_interval_split_matched_penalizes_too_many_neighbors(make_test_env):
    env = make_test_env(
        agent_radius=1,
        collision_iters=0,
        collision_grid_size=2,
        k=2,
        perception_radius=4.0,
        lam_nbr_reward=1.0,
        reward_mode="interval_split_matched",
        time_horizon=32,
        enforce_connected=False,
        num_agents=3,
        target_neighbors=np.array([[1, -1], [0, -1], [0, -1]], dtype=np.int32),
        target_neighbor_distances=np.array([[3.0, 0.0]] * 3, dtype=np.float32),
        initial_positions=np.array(
            [[20.0, 20.0], [23.0, 20.0], [25.0, 20.0]],
            dtype=np.float32,
        ),
    )
    try:
        env.reset(seed=0)
        actions = np.array([[4, 5], [4, 5], [4, 5]], dtype=np.int32)
        _, rewards, _, _, _ = env.step(actions)
        assert rewards == pytest.approx(np.zeros(3))

        actions[0] = [4, 4]
        _, rewards, _, _, _ = env.step(actions)
        assert rewards[0] == pytest.approx(-1.0)
    finally:
        env.close()


def test_interval_split_matched_uses_neighbor_pair_distances(make_test_env):
    target_neighbors = np.array(
        [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]], dtype=np.int32
    )
    target_neighbor_distances = np.array(
        [
            [2.0, 2.0],
            [2.0, np.sqrt(8.0)],
            [2.0, np.sqrt(8.0)],
            [2.0, 2.0],
            [2.0, 4.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )
    target_positions = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [20.0, 0.0], [18.0, 0.0], [22.0, 0.0]],
        dtype=np.float32,
    )
    initial_positions = np.array(
        [
            [20.0, 20.0],
            [18.0, 20.0],
            [22.0, 20.0],
            [50.0, 20.0],
            [52.0, 20.0],
            [50.0, 22.0],
        ],
        dtype=np.float32,
    )

    def build_env(pair_distances):
        return make_test_env(
            width=80,
            height=80,
            agent_radius=1,
            collision_iters=0,
            collision_grid_size=2,
            k=2,
            perception_radius=10.0,
            lam_nbr_reward=0.0,
            reward_mode="interval_split_matched",
            reward_interval_multiplier=1.0,
            max_speed=2.0,
            time_horizon=32,
            reuse_initial_positions=True,
            enforce_connected=False,
            recenter_on_step=False,
            log_interval=128,
            num_agents=6,
            target_neighbors=target_neighbors,
            target_neighbor_distances=target_neighbor_distances,
            target_neighbor_pair_distances=pair_distances,
            initial_positions=initial_positions,
        )

    pair_distances = build_target_neighbor_pair_distances(
        target_positions, target_neighbors
    )
    env_plain = build_env(None)
    env_pairs = build_env(pair_distances)
    try:
        actions = np.tile(np.array([[4, 2]], dtype=np.int32), (6, 1))
        env_plain.reset(seed=0)
        env_pairs.reset(seed=0)
        env_plain.step(actions)
        env_pairs.step(actions)

        actions = actions.copy()
        actions[4] = [4, 4]
        env_plain.step(actions)
        env_pairs.step(actions)

        actions = np.tile(np.array([[4, 2]], dtype=np.int32), (6, 1))
        actions[3] = [0, 3]
        _, plain_rewards, _, _, _ = env_plain.step(actions)
        _, pair_rewards, _, _, _ = env_pairs.step(actions)

        assert pair_rewards != pytest.approx(plain_rewards)
    finally:
        env_plain.close()
        env_pairs.close()


def test_interval_split_matched_logs_anonymous_role_error(make_test_env):
    env = make_test_env(
        width=64,
        height=64,
        agent_radius=1,
        collision_iters=0,
        collision_grid_size=2,
        k=2,
        perception_radius=20.0,
        lam_nbr_reward=0.0,
        reward_mode="interval_split_matched",
        time_horizon=1,
        reuse_initial_positions=True,
        enforce_connected=False,
        log_interval=1,
        num_agents=3,
        target_neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        target_neighbor_distances=np.array(
            [[2.0, 6.0], [2.0, 4.0], [6.0, 4.0]], dtype=np.float32
        ),
        initial_positions=np.array(
            [[26.0, 20.0], [20.0, 20.0], [22.0, 20.0]], dtype=np.float32
        ),
    )
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([[4, 5], [4, 5], [4, 5]], dtype=np.int32))
        assert info[0]["error"] == pytest.approx(0.0, abs=1e-5)
    finally:
        env.close()


def test_interval_split_matched_can_freeze_role_assignment(make_test_env):
    def build_env(match_roles_once: bool):
        return make_test_env(
            width=64,
            height=64,
            agent_radius=1,
            collision_iters=0,
            collision_grid_size=2,
            k=2,
            perception_radius=20.0,
            lam_nbr_reward=0.0,
            reward_mode="interval_split_matched",
            match_roles_once=match_roles_once,
            time_horizon=1,
            reuse_initial_positions=True,
            enforce_connected=False,
            recenter_on_step=False,
            log_interval=1,
            num_agents=3,
            target_neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
            target_neighbor_distances=np.array(
                [[2.0, 6.0], [2.0, 4.0], [6.0, 4.0]], dtype=np.float32
            ),
            initial_positions=np.array(
                [[26.0, 20.0], [20.0, 20.0], [22.0, 20.0]], dtype=np.float32
            ),
        )

    env_dynamic = build_env(False)
    env_fixed = build_env(True)
    try:
        actions = np.array([[4, 5], [4, 5], [4, 4]], dtype=np.int32)
        env_dynamic.reset(seed=0)
        env_fixed.reset(seed=0)
        _, dynamic_rewards, _, _, dynamic_info = env_dynamic.step(actions)
        _, fixed_rewards, _, _, fixed_info = env_fixed.step(actions)

        assert dynamic_rewards == pytest.approx(fixed_rewards)
        assert dynamic_info[0]["error"] == pytest.approx(0.0, abs=1e-5)
        assert fixed_info[0]["error"] > 0.5
    finally:
        env_dynamic.close()
        env_fixed.close()


def test_interval_extreme_reward_is_logged_per_interval(make_test_env):
    env = _build_env(make_test_env, reward_mode="interval_v1_cached", log_interval=1)
    try:
        env.reset(seed=0)

        warmup_actions = [
            (2, 2),  # warmup, reward 0
            (0, 4),  # reward -1
            (0, 4),  # reward -1
            (0, 4),  # reward 0 (hidden distance remains frozen)
            (5, 5),  # reward 0
            (4, 0),  # reward 0
            (4, 0),  # reward +1
        ]
        seen_extrema: list[float] = []
        pending_rewards: list[np.ndarray] = []
        for speed0, speed1 in warmup_actions + [(2, 2)] * 64:
            step_actions = np.array([[4, speed0], [4, speed1]], dtype=np.int32)
            _, rewards, _, _, info = env.step(step_actions)
            pending_rewards.append(rewards.copy())
            if not info:
                continue
            seen_extrema.append(assert_logged_extreme(info, pending_rewards))

        assert seen_extrema
        assert -1.0 in seen_extrema
    finally:
        env.close()
