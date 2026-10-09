import numpy as np
import pufferlib
import pytest
import torch
import torch.nn as nn
from pufferlib.models import LSTMWrapper

from magna.config import Config, EnvConfig
from magna.data.constants import DEFAULT_COORDINATE_SCALE
from magna.env.env import MultiCellEnv
from magna.policy import Policy, PolicyBackbone
from magna.train import QuietPuffeRL


def _build_env(
    make_test_env,
    *,
    mutual_obs_bit: bool,
    num_envs: int = 1,
    perceived_ratio_obs: bool = True,
    density_grad_obs: bool = False,
    density_grad_log_norm: bool = False,
    local_frame_obs: bool = True,
    self_pos_delta_obs: bool = True,
    boundary_dist_obs: bool = False,
    target_neighbor_bit: bool = False,
    comm_select_action: bool = False,
    collision_iters: int = 1,
    perception_radius: float = 10.0,
    initial_headings=None,
) -> MultiCellEnv:
    return make_test_env(
        num_envs=num_envs,
        width=64,
        height=64,
        agent_radius=1,
        collision_iters=collision_iters,
        collision_grid_size=2,
        k=1,
        perception_radius=perception_radius,
        mutual_obs_bit=mutual_obs_bit,
        target_neighbor_bit=target_neighbor_bit,
        perceived_ratio_obs=perceived_ratio_obs,
        density_grad_obs=density_grad_obs,
        density_grad_log_norm=density_grad_log_norm,
        local_frame_obs=local_frame_obs,
        self_pos_delta_obs=self_pos_delta_obs,
        boundary_dist_obs=boundary_dist_obs,
        comm_select_action=comm_select_action,
        lam_nbr_reward=0.0,
        reward_mode="interval_v1",
        reward_interval_multiplier=2.0,
        max_speed=2.0,
        time_horizon=16,
        reuse_initial_positions=True,
        log_interval=128,
        num_agents=3,
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.array([[1.0], [0.1], [0.1]], dtype=np.float32),
        initial_positions=np.array(
            [
                [20.0, 20.0],  # agent 0
                [21.0, 20.0],  # agent 1
                [21.1, 20.0],  # agent 2
            ],
            dtype=np.float32,
        ),
        initial_headings=initial_headings,
    )


def _assert_logits_close(actual, expected):
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e)
        return
    torch.testing.assert_close(actual, expected)


def _build_reference_wrapper(
    env: MultiCellEnv,
    *,
    id_dim: int,
    id_vectors: np.ndarray,
    hidden_size: int,
    message_agg: str = "concat",
) -> LSTMWrapper:
    backbone = PolicyBackbone(
        env,
        n_neighbors=1,
        id_dim=id_dim,
        id_vectors=id_vectors,
        message_agg=message_agg,
        hidden_size=hidden_size,
        comm_enabled=False,
    )
    return LSTMWrapper(
        env, policy=backbone, input_size=hidden_size, hidden_size=hidden_size
    )


def _build_lstm_input(policy, observations, last_hidden):
    return policy._build_lstm_input(
        policy._parse_observations(observations), last_hidden
    )


def _reset_obs(make_test_env, **kwargs) -> np.ndarray:
    env = _build_env(make_test_env, **kwargs)
    try:
        obs, _ = env.reset(seed=7)
        return obs.copy()
    finally:
        env.close()


@pytest.mark.parametrize("mutual_obs_bit", [False, True])
@pytest.mark.parametrize("perceived_ratio_obs", [False, True])
@pytest.mark.parametrize("self_pos_delta_obs", [False, True])
@pytest.mark.parametrize("boundary_dist_obs", [False, True])
def test_observation_shape_depends_on_config_flags(
    make_test_env,
    mutual_obs_bit: bool,
    perceived_ratio_obs: bool,
    self_pos_delta_obs: bool,
    boundary_dist_obs: bool,
):
    env = _build_env(
        make_test_env,
        mutual_obs_bit=mutual_obs_bit,
        perceived_ratio_obs=perceived_ratio_obs,
        self_pos_delta_obs=self_pos_delta_obs,
        boundary_dist_obs=boundary_dist_obs,
    )
    try:
        neighbor_obs_dim = 8 if mutual_obs_bit else 7
        self_obs_dim = (
            1
            + int(perceived_ratio_obs)
            + (3 if self_pos_delta_obs else 0)
            + int(boundary_dist_obs)
        )
        expected_shape = (neighbor_obs_dim + 1 + self_obs_dim,)
        assert env.single_observation_space.shape == expected_shape
    finally:
        env.close()


def test_comm_select_action_adds_binary_action_heads_and_obs_bits(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False, comm_select_action=True)
    try:
        assert env.single_observation_space.shape == (14,)
        assert env.single_action_space.nvec.tolist() == [9, 6, 2]
    finally:
        env.close()


def test_comm_select_action_observes_selected_receivers(make_test_env):
    env = make_test_env(
        num_agents=3,
        k=1,
        agent_radius=1,
        collision_grid_size=2,
        perception_radius=16.0,
        mutual_obs_bit=False,
        comm_select_action=True,
        initial_positions=np.array(
            [[20.0, 20.0], [25.0, 20.0], [27.0, 20.0]], dtype=np.float32
        ),
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.ones((3, 1), dtype=np.float32),
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.array([[4, 2, 0], [4, 2, 1], [4, 2, 0]], dtype=np.int32))

        assert obs[0, 5] == pytest.approx(0.0)
        assert obs[2, 5] == pytest.approx(1.0)
    finally:
        env.close()


def test_comm_select_action_can_select_multiple_neighbors(make_test_env):
    env = make_test_env(
        num_agents=3,
        k=2,
        agent_radius=1,
        collision_grid_size=2,
        perception_radius=16.0,
        mutual_obs_bit=False,
        comm_select_action=True,
        initial_positions=np.array(
            [[20.0, 20.0], [25.0, 20.0], [30.0, 20.0]], dtype=np.float32
        ),
        target_neighbors=np.array([[1, 2], [0, 2], [1, 0]], dtype=np.int32),
        target_neighbor_distances=np.ones((3, 2), dtype=np.float32),
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(
            np.array([[4, 2, 1, 1], [4, 2, 0, 0], [4, 2, 0, 0]], dtype=np.int32)
        )

        neighbors = obs[:, :14].reshape(3, 2, 7)
        for receiver in (1, 2):
            sender_slot = np.where(neighbors[receiver, :, 6] == 1.0)[0].item()
            assert neighbors[receiver, sender_slot, 5] == pytest.approx(1.0)
    finally:
        env.close()


def test_comm_select_action_bit_follows_mutual_obs_bit(make_test_env):
    env = make_test_env(
        num_agents=3,
        k=1,
        agent_radius=1,
        collision_grid_size=2,
        perception_radius=16.0,
        mutual_obs_bit=True,
        comm_select_action=True,
        initial_positions=np.array(
            [[20.0, 20.0], [25.0, 20.0], [27.0, 20.0]], dtype=np.float32
        ),
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.ones((3, 1), dtype=np.float32),
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.array([[4, 2, 0], [4, 2, 1], [4, 2, 0]], dtype=np.int32))

        assert obs[2, 5] == pytest.approx(1.0)
        assert obs[2, 6] == pytest.approx(1.0)
    finally:
        env.close()


def test_reciprocal_topk_mutual_obs_bit_values(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=True)
    try:
        obs, _ = env.reset(seed=0)
        # Neighbor tuple layout when enabled:
        # [dist, sin_bearing, cos_bearing, sin_rel_h, cos_rel_h, speed, mutual_bit, id]
        assert obs[0, 6] == pytest.approx(0.0)
        assert obs[2, 6] == pytest.approx(1.0)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("positions", "perception_radius", "target", "neighbor_id"),
    [
        # Agent 0 sees target neighbor 1.
        (None, 10.0, 1.0, 2.0),
        # Agent 0 closest to 2, but targets 1.
        (
            np.array([[20.0, 20.0], [30.0, 20.0], [21.0, 20.0]], dtype=np.float32),
            10.0,
            0.0,
            3.0,
        ),
        # No neighbor in range.
        (
            np.array([[10.0, 10.0], [40.0, 10.0], [50.0, 10.0]], dtype=np.float32),
            1.0,
            0.0,
            0.0,
        ),
    ],
)
def test_target_neighbor_bit(
    make_test_env, positions, perception_radius, target, neighbor_id
):
    kwargs = {
        "mutual_obs_bit": False,
        "target_neighbor_bit": True,
        "local_frame_obs": True,
        "perception_radius": perception_radius,
    }
    if positions is None:
        env = _build_env(make_test_env, **kwargs)
    else:
        env = make_test_env(
            **kwargs,
            k=1,
            initial_positions=positions,
            target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
            target_neighbor_distances=np.ones((3, 1), dtype=np.float32),
        )
    try:
        obs, _ = env.reset(seed=0)
        assert obs[0, env.target_neighbor_bit_index] == pytest.approx(target)
        assert obs[0, env.neighbor_obs_payload_dim] == pytest.approx(neighbor_id)
    finally:
        env.close()


def test_target_neighbor_bit_bumps_neighbor_obs_dim(make_test_env):
    base = _build_env(make_test_env, mutual_obs_bit=False)
    enabled = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        target_neighbor_bit=True,
    )
    try:
        assert enabled.neighbor_obs_dim == base.neighbor_obs_dim + 1
        bit = enabled.neighbor_cell_start + enabled.cell_obs_dim
        assert enabled.target_neighbor_bit_index == bit
    finally:
        base.close()
        enabled.close()


def test_self_pos_delta_obs_false_omits_only_self_delta_tail_channels(make_test_env):
    env_without_delta = _build_env(
        make_test_env, mutual_obs_bit=False, self_pos_delta_obs=False
    )
    try:
        obs_without_delta, _ = env_without_delta.reset(seed=7)
        obs_without_delta = obs_without_delta.copy()
    finally:
        env_without_delta.close()

    env_with_delta = _build_env(
        make_test_env, mutual_obs_bit=False, self_pos_delta_obs=True
    )
    try:
        obs_with_delta, _ = env_with_delta.reset(seed=7)
        obs_with_delta = obs_with_delta.copy()
    finally:
        env_with_delta.close()

    # self_pos_delta_obs=False should drop only the final 3 self channels.
    assert obs_without_delta.shape[1] + 3 == obs_with_delta.shape[1]
    np.testing.assert_allclose(obs_without_delta, obs_with_delta[:, :-3])


def test_perceived_ratio_obs_false_omits_only_perceived_ratio_channel(make_test_env):
    env_without_ratio = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        self_pos_delta_obs=True,
    )
    try:
        obs_without_ratio, _ = env_without_ratio.reset(seed=7)
        obs_without_ratio = obs_without_ratio.copy()
    finally:
        env_without_ratio.close()

    env_with_ratio = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=True,
        self_pos_delta_obs=True,
    )
    try:
        obs_with_ratio, _ = env_with_ratio.reset(seed=7)
        obs_with_ratio = obs_with_ratio.copy()
    finally:
        env_with_ratio.close()

    # perceived_ratio_obs=False should drop only the perceived_ratio channel.
    assert obs_without_ratio.shape[1] + 1 == obs_with_ratio.shape[1]
    np.testing.assert_allclose(obs_without_ratio[:, :8], obs_with_ratio[:, :8])
    np.testing.assert_allclose(obs_without_ratio[:, 8:], obs_with_ratio[:, 9:])


def test_density_grad_obs_adds_spiky_gradient_tail(make_test_env):
    obs_without = _reset_obs(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        density_grad_obs=False,
        self_pos_delta_obs=False,
    )
    obs_with = _reset_obs(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        density_grad_obs=True,
        self_pos_delta_obs=False,
    )

    assert obs_without.shape[1] + 3 == obs_with.shape[1]
    np.testing.assert_allclose(obs_without, obs_with[:, :-3])
    assert obs_with[0, -3] > 0.0
    assert obs_with[0, -2] == pytest.approx(0.0, abs=1e-6)
    assert obs_with[0, -1] == pytest.approx(1.0, abs=1e-6)
    assert obs_with[2, -3] > 0.0
    assert obs_with[2, -1] < 0.0


def test_world_frame_density_grad_uses_xy_components(make_test_env):
    obs = _reset_obs(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        density_grad_obs=True,
        local_frame_obs=False,
        self_pos_delta_obs=False,
    )

    assert obs.shape[1] == 12
    assert obs[0, -2] > 0.0
    assert obs[0, -1] == pytest.approx(0.0, abs=1e-6)


def test_density_grad_log_norm_scales_direction_and_magnitude(make_test_env):
    obs_raw = _reset_obs(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        density_grad_obs=True,
        density_grad_log_norm=False,
        self_pos_delta_obs=False,
    )
    obs_log = _reset_obs(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        density_grad_obs=True,
        density_grad_log_norm=True,
        self_pos_delta_obs=False,
    )

    assert obs_raw.shape == obs_log.shape
    np.testing.assert_allclose(
        obs_log[:, -3], np.log1p(obs_raw[:, -3]), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(obs_log[:, -2:], obs_raw[:, -2:], rtol=1e-5, atol=1e-7)


def test_density_grad_obs_preserves_small_nonzero_magnitude(make_test_env):
    obs = _reset_obs(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        density_grad_obs=True,
        self_pos_delta_obs=False,
        perception_radius=1000.0,
    )

    assert 0.0 < obs[0, -3] < 1e-6
    np.testing.assert_allclose(np.linalg.norm(obs[0, -2:]), 1.0, atol=1e-5)


def test_boundary_dist_obs_false_omits_only_boundary_tail_channel(make_test_env):
    env_without = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        boundary_dist_obs=False,
    )
    try:
        obs_without, _ = env_without.reset(seed=7)
        obs_without = obs_without.copy()
    finally:
        env_without.close()

    env_with = _build_env(make_test_env, mutual_obs_bit=False, boundary_dist_obs=True)
    try:
        obs_with, _ = env_with.reset(seed=7)
        obs_with = obs_with.copy()
    finally:
        env_with.close()

    assert obs_without.shape[1] + 1 == obs_with.shape[1]
    np.testing.assert_allclose(obs_without, obs_with[:, :-1])


@pytest.mark.parametrize(
    ("circular", "position", "expected"),
    [
        (False, (6.0, 32.0), 0.5),
        (True, (55.0, 32.0), 0.8),
        (True, (32.0, 32.0), 1.0),
    ],
)
def test_boundary_dist_obs_respects_arena_shape_and_caps_at_one(
    make_test_env, circular: bool, position: tuple[float, float], expected: float
):
    env = make_test_env(
        width=64,
        height=64,
        circular=circular,
        agent_radius=1,
        k=1,
        perception_radius=10.0,
        mutual_obs_bit=False,
        perceived_ratio_obs=True,
        self_pos_delta_obs=True,
        boundary_dist_obs=True,
        density_grad_obs=False,
        initial_positions=np.array(
            [position, [40.0, 40.0], [50.0, 50.0]], dtype=np.float32
        ),
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs[0, -1] == pytest.approx(expected)
    finally:
        env.close()


@pytest.mark.parametrize("mutual_obs_bit", [False, True])
@pytest.mark.parametrize("perceived_ratio_obs", [False, True])
@pytest.mark.parametrize("self_pos_delta_obs", [False, True])
@pytest.mark.parametrize("boundary_dist_obs", [False, True])
def test_policy_parses_observations_with_or_without_optional_channels(
    make_test_env,
    mutual_obs_bit: bool,
    perceived_ratio_obs: bool,
    self_pos_delta_obs: bool,
    boundary_dist_obs: bool,
):
    env = _build_env(
        make_test_env,
        mutual_obs_bit=mutual_obs_bit,
        perceived_ratio_obs=perceived_ratio_obs,
        self_pos_delta_obs=self_pos_delta_obs,
        boundary_dist_obs=boundary_dist_obs,
    )
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg="concat",
            hidden_size=16,
        )
        hidden = _build_lstm_input(
            policy.policy,
            torch.as_tensor(obs),
            torch.zeros(env.num_agents, policy.hidden_size),
        )
        assert hidden.shape == (3, 16)
        assert policy.policy.local_num_agents == 3
    finally:
        env.close()


def test_comm_gather_offsets_local_neighbor_ids_per_vector_env(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False, num_envs=2)
    try:
        env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg="concat",
            hidden_size=16,
            comm_enabled=True,
            comm_msg_dim=16,
        )
        neighbor_ids = torch.tensor([[2], [3], [2], [2], [3], [2]])
        last_msg = torch.arange(
            env.num_agents * policy.hidden_size, dtype=torch.float32
        ).view(env.num_agents, policy.hidden_size)
        neighbor_msg, neighbor_mask = policy.policy._gather_neighbor_messages(
            last_msg, neighbor_ids
        )

        num_agents = policy.policy.local_num_agents
        row_offsets = (
            torch.arange(env.num_agents, dtype=neighbor_ids.dtype).view(-1, 1)
            // num_agents
            * num_agents
        )
        padded_ids = torch.where(
            neighbor_ids == 0, neighbor_ids, neighbor_ids + row_offsets
        )
        expected = torch.cat(
            [last_msg.new_zeros(1, policy.hidden_size), last_msg], dim=0
        )[padded_ids]
        torch.testing.assert_close(neighbor_msg, expected)
        assert torch.equal(neighbor_mask, neighbor_ids != 0)
        torch.testing.assert_close(
            neighbor_msg[num_agents:],
            neighbor_msg[:num_agents] + num_agents * policy.hidden_size,
        )
    finally:
        env.close()


def test_vector_comm_receives_same_env_neighbor_messages(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False, num_envs=2)
    try:
        env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg="concat",
            hidden_size=16,
            comm_enabled=True,
        )

        neighbor_ids = torch.tensor([[2], [3], [2], [2], [3], [2]])
        last_msg = torch.arange(
            env.num_agents * policy.hidden_size, dtype=torch.float32
        ).view(env.num_agents, policy.hidden_size)
        neighbor_msg, neighbor_mask = policy.policy._gather_neighbor_messages(
            last_msg, neighbor_ids
        )

        num_agents = policy.policy.local_num_agents
        local_neighbor_ids = neighbor_ids.squeeze(1)
        row_offsets = torch.arange(env.num_agents) // num_agents * num_agents
        expected_rows = local_neighbor_ids - 1 + row_offsets
        expected = last_msg[expected_rows]
        torch.testing.assert_close(neighbor_msg.squeeze(1), expected)
        assert torch.equal(neighbor_mask, neighbor_ids != 0)
        torch.testing.assert_close(
            neighbor_msg[num_agents:].squeeze(1),
            neighbor_msg[:num_agents].squeeze(1) + num_agents * policy.hidden_size,
        )
    finally:
        env.close()


def test_comm_routes_messages_to_neighbor_slots(make_test_env):
    env = make_test_env(
        k=2,
        target_neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        target_neighbor_distances=np.full((3, 2), 1.0, dtype=np.float32),
    )
    try:
        policy = Policy(
            env,
            n_neighbors=2,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg="concat",
            hidden_size=16,
            comm_enabled=True,
        )
        neighbor_ids = torch.tensor([[2, 3], [1, 3], [1, 0]])
        last_msg = torch.arange(48, dtype=torch.float32).reshape(3, 16)

        neighbor_msg, neighbor_mask = policy.policy._gather_neighbor_messages(
            last_msg, neighbor_ids
        )

        torch.testing.assert_close(
            neighbor_msg,
            torch.stack(
                [
                    torch.stack([last_msg[1], last_msg[2]]),
                    torch.stack([last_msg[0], last_msg[2]]),
                    torch.stack([last_msg[0], torch.zeros(16)]),
                ]
            ),
        )
        assert torch.equal(
            neighbor_mask,
            torch.tensor([[True, True], [True, True], [True, False]]),
        )
    finally:
        env.close()


def test_comm_select_action_masks_unselected_neighbor_messages(make_test_env):
    env = make_test_env(
        num_agents=3,
        k=1,
        agent_radius=1,
        collision_grid_size=2,
        perception_radius=16.0,
        mutual_obs_bit=False,
        comm_select_action=True,
        initial_positions=np.array(
            [[20.0, 20.0], [25.0, 20.0], [27.0, 20.0]], dtype=np.float32
        ),
        target_neighbors=np.array([[1], [2], [1]], dtype=np.int32),
        target_neighbor_distances=np.ones((3, 1), dtype=np.float32),
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.array([[4, 2, 0], [4, 2, 1], [4, 2, 0]], dtype=np.int32))
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg="concat",
            hidden_size=16,
            comm_enabled=True,
        )

        class HiddenCellEncoder(nn.Module):
            def forward(self, x):
                self.x = x.detach()
                return x[..., -policy.hidden_size :]

        class CaptureMessageEncoder(nn.Module):
            def forward(self, x):
                self.x = x.detach()
                return x.new_zeros(*x.shape[:-1], policy.hidden_size)

        policy.policy.cell_encoder = HiddenCellEncoder()
        policy.policy.comm_proj = nn.Identity()
        policy.policy.message_encoder = CaptureMessageEncoder()
        policy.policy.update_encoder = nn.Sequential(
            nn.Linear(policy.policy.update_input_dim, policy.hidden_size)
        )
        _build_lstm_input(
            policy.policy,
            torch.as_tensor(obs),
            torch.arange(10, 13, dtype=torch.float32)
            .view(-1, 1)
            .expand(-1, policy.hidden_size),
        )
        received = policy.policy.message_encoder.x[
            :, 0, policy.hidden_size : 2 * policy.hidden_size
        ]

        torch.testing.assert_close(received[0], torch.zeros(policy.hidden_size))
        torch.testing.assert_close(received[2], torch.full((policy.hidden_size,), 11.0))
    finally:
        env.close()


def test_comm_forward_eval_uses_lstm_hidden_state(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg="concat",
            hidden_size=16,
            comm_enabled=True,
        )
        state = {"lstm_h": None, "lstm_c": None}

        _, values = policy.forward_eval(torch.as_tensor(obs), state)

        assert values.shape == (env.num_agents, 1)
        assert state["lstm_h"].shape == (env.num_agents, policy.hidden_size)
    finally:
        env.close()


def _build_vector_comm_training_case(tmp_path, trainer_cls=QuietPuffeRL):
    local_num_agents = 3
    env_cfg = EnvConfig.model_validate(
        {
            "num_envs": 2,
            "width": 64,
            "height": 64,
            "agent_radius": 1,
            "collision_iters": 1,
            "collision_grid_size": 2,
            "k": 1,
            "perception_radius": 10.0,
            "lam_nbr_reward": 0.0,
            "reward_mode": "interval_v1",
            "max_speed": 2.0,
            "time_horizon": 16,
            "reuse_initial_positions": True,
            "log_interval": 128,
        }
    )
    env_kwargs = {
        "cfg": env_cfg,
        "num_agents": local_num_agents,
        "coordinate_scale": DEFAULT_COORDINATE_SCALE,
        "target_neighbors": np.array([[1], [2], [1]], dtype=np.int32),
        "target_neighbor_distances": np.array([[1.0], [0.1], [0.1]], dtype=np.float32),
        "initial_positions": np.array(
            [
                [[20.0, 20.0], [21.0, 20.0], [21.1, 20.0]],
                [[20.0, 30.0], [22.0, 30.0], [22.2, 30.0]],
            ],
            dtype=np.float32,
        ),
    }
    vecenv = pufferlib.vector.make(
        MultiCellEnv,
        num_envs=1,
        backend=pufferlib.vector.Serial,
        seed=0,
        env_kwargs=env_kwargs,
    )

    class RecordingPolicy(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_groups = []

        def forward(self, observations, state):
            if observations.ndim == 3:
                self.training_groups.append(
                    (
                        observations[:, 0, self.policy.neighbor_obs_end]
                        .detach()
                        .cpu()
                        .round()
                        .long(),
                        observations[:, 0, 0].detach().cpu(),
                    )
                )
            return super().forward(observations, state)

    policy = RecordingPolicy(
        vecenv.driver_env,
        n_neighbors=1,
        id_dim=2,
        id_vectors=np.zeros((local_num_agents, 2), dtype=np.float32),
        message_agg="concat",
        hidden_size=16,
        comm_enabled=True,
    )
    train_cfg = Config.model_validate(
        {
            "train": {
                "device": "cpu",
                "optimizer": "adam",
                "batch_size": 12,
                "minibatch_size": 6,
                "max_minibatch_size": 6,
                "bptt_horizon": 2,
                "total_timesteps": 12,
                "checkpoint_interval": 100,
                "data_dir": str(tmp_path),
                "print_dashboard": False,
                "generate_rollout_video": False,
            }
        }
    ).train.model_dump()
    trainer = trainer_cls(train_cfg, vecenv, policy)
    return local_num_agents, vecenv, policy, trainer


def _assert_training_groups_are_whole_envs(policy, local_num_agents):
    expected_group = torch.tensor([1, 2, 3])
    expected_distances = [
        torch.tensor([0.1, 0.01, -0.01]),
        torch.tensor([0.2, 0.02, -0.02]),
    ]
    assert policy.training_groups
    for self_ids, neighbor_distances in policy.training_groups:
        assert self_ids.numel() % local_num_agents == 0
        for group_ids, group_distances in zip(
            self_ids.view(-1, local_num_agents),
            neighbor_distances.view(-1, local_num_agents),
            strict=True,
        ):
            torch.testing.assert_close(group_ids, expected_group)
            assert any(
                torch.allclose(group_distances, distances, rtol=0.0, atol=1e-6)
                for distances in expected_distances
            )


def test_vector_comm_training_minibatches_keep_local_agent_groups(tmp_path):
    local_num_agents, vecenv, policy, trainer = _build_vector_comm_training_case(
        tmp_path
    )
    try:
        trainer.evaluate()
        trainer.train()
        _assert_training_groups_are_whole_envs(policy, local_num_agents)
    finally:
        vecenv.close()
        trainer.utilization.stop()


def test_original_training_sampler_would_mix_vector_comm_groups(tmp_path):
    class OriginalSamplerQuietPuffeRL(QuietPuffeRL):
        def _sample_minibatch_indices(self, prio_weights, anneal_beta):
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            return idx, (self.segments * prio_probs[idx, None]) ** -anneal_beta

    local_num_agents, vecenv, policy, trainer = _build_vector_comm_training_case(
        tmp_path, OriginalSamplerQuietPuffeRL
    )
    try:
        trainer.evaluate()
        torch.manual_seed(0)
        trainer.train()
        assert policy.training_groups
        with pytest.raises(AssertionError):
            _assert_training_groups_are_whole_envs(policy, local_num_agents)
    finally:
        vecenv.close()
        trainer.utilization.stop()


def test_policy_id_feature_encoder_disabled_by_default(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=4,
            id_vectors=np.ones((3, 4), dtype=np.float32),
            hidden_size=16,
        )
        assert policy.policy.id_encoder is None
        assert policy.policy.id_dim == 4
        assert policy.policy.id_table.shape == (4, 4)
        assert policy.policy.encoder_layers == 1
    finally:
        env.close()


def test_mask_nontarget_id_features_masks_non_target_neighbor_ids(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False, target_neighbor_bit=True)
    try:
        obs, _ = env.reset(seed=0)
        obs = obs.copy()
        # Force non-target agent 2 into the slot with target bit cleared.
        obs[0, env.neighbor_obs_payload_dim] = 3.0
        obs[0, env.neighbor_obs_dim] = 1.0
        obs[0, env.target_neighbor_bit_index] = 0.0
        id_vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=id_vectors,
            mask_nontarget_id_features=True,
            hidden_size=16,
        )
        parsed = policy.policy._parse_observations(torch.as_tensor(obs[:1]))
        assert parsed.neighbor_ids[0, 0].item() == 3
        torch.testing.assert_close(
            parsed.neighbor_id_embed[0, 0], torch.zeros(2, dtype=torch.float32)
        )
    finally:
        env.close()


@pytest.mark.parametrize(("id_dim", "expected_dim"), [(2, 6), (0, 4)])
def test_policy_supports_static_and_learned_id_features(
    make_test_env, id_dim, expected_dim
):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=id_dim,
            id_vectors=np.ones((3, id_dim), dtype=np.float32),
            learned_id_dim=4,
            hidden_size=16,
        )
        parsed = policy.policy._parse_observations(torch.as_tensor(obs))

        assert policy.policy.id_dim == expected_dim
        assert parsed.self_id_embed.shape == (3, expected_dim)
        torch.testing.assert_close(policy.policy.learned_id.weight[0], torch.zeros(4))
    finally:
        env.close()


@pytest.mark.parametrize("init", ["zeros", "gaussian", "orthogonal"])
def test_policy_learned_id_initialization(make_test_env, init: str):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=0,
            id_vectors=np.empty((3, 0), dtype=np.float32),
            learned_id_dim=2,
            learned_id_init=init,
            hidden_size=16,
        )
        weight = policy.policy.learned_id.weight

        torch.testing.assert_close(weight[0], torch.zeros(2))
        assert torch.count_nonzero(weight[1:]) == (0 if init == "zeros" else 6)
    finally:
        env.close()


def test_policy_no_comms_matches_original_lstm_wrapper_forward_eval(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        id_vectors = np.ones((3, 4), dtype=np.float32)
        hidden_size = 16
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=4,
            id_vectors=id_vectors,
            hidden_size=hidden_size,
            message_agg="concat",
            comm_enabled=False,
        )
        reference = _build_reference_wrapper(
            env,
            id_dim=4,
            id_vectors=id_vectors,
            hidden_size=hidden_size,
            message_agg="concat",
        )
        reference.load_state_dict(policy.state_dict(), strict=True)

        obs_t = torch.as_tensor(obs)
        state = {"lstm_h": None, "lstm_c": None}
        ref_state = {"lstm_h": None, "lstm_c": None}

        logits, values = policy.forward_eval(obs_t, state)
        ref_logits, ref_values = reference.forward_eval(obs_t, ref_state)

        _assert_logits_close(logits, ref_logits)
        torch.testing.assert_close(values, ref_values)
        torch.testing.assert_close(state["hidden"], ref_state["hidden"])
        torch.testing.assert_close(state["lstm_h"], ref_state["lstm_h"])
        torch.testing.assert_close(state["lstm_c"], ref_state["lstm_c"])
    finally:
        env.close()


def test_policy_no_comms_matches_original_lstm_wrapper_forward(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        id_vectors = np.ones((3, 4), dtype=np.float32)
        hidden_size = 16
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=4,
            id_vectors=id_vectors,
            hidden_size=hidden_size,
            message_agg="concat",
            comm_enabled=False,
        )
        reference = _build_reference_wrapper(
            env,
            id_dim=4,
            id_vectors=id_vectors,
            hidden_size=hidden_size,
            message_agg="concat",
        )
        reference.load_state_dict(policy.state_dict(), strict=True)

        obs_t = torch.as_tensor(obs).unsqueeze(1).repeat(1, 4, 1)
        state = {"lstm_h": None, "lstm_c": None}
        ref_state = {"lstm_h": None, "lstm_c": None}

        logits, values = policy(obs_t, state)
        ref_logits, ref_values = reference(obs_t, ref_state)

        _assert_logits_close(logits, ref_logits)
        torch.testing.assert_close(values, ref_values)
        torch.testing.assert_close(state["hidden"], ref_state["hidden"])
        torch.testing.assert_close(state["lstm_h"], ref_state["lstm_h"])
        torch.testing.assert_close(state["lstm_c"], ref_state["lstm_c"])
    finally:
        env.close()


@pytest.mark.parametrize("message_agg", ["concat", "mean", "attn"])
def test_policy_id_feature_encoder_projects_only_id_vectors(
    make_test_env, message_agg: str
):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=32,
            id_vectors=np.ones((3, 32), dtype=np.float32),
            id_feature_encoder=True,
            message_agg=message_agg,
            hidden_size=16,
        )
        hidden = _build_lstm_input(
            policy.policy,
            torch.as_tensor(obs),
            torch.zeros(env.num_agents, policy.hidden_size),
        )

        assert policy.policy.id_encoder is not None
        assert policy.policy.id_input_dim == 32
        assert policy.policy.id_dim == 16
        assert policy.policy.id_table.shape == (4, 32)
        torch.testing.assert_close(
            policy.policy.id_table[0],
            torch.zeros(32, dtype=policy.policy.id_table.dtype),
        )
        assert hidden.shape == (3, 16)
    finally:
        env.close()


@pytest.mark.parametrize("message_agg", ["concat", "mean"])
def test_policy_null_msg_can_be_learnable(make_test_env, message_agg: str):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        obs[0, :7] = 0
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=4,
            id_vectors=np.ones((3, 4), dtype=np.float32),
            null_msg_learnable=True,
            message_agg=message_agg,
            hidden_size=16,
        )

        hidden = _build_lstm_input(
            policy.policy, torch.as_tensor(obs[:1]), torch.zeros(1, policy.hidden_size)
        )
        hidden.sum().backward()

        assert policy.policy.null_msg.grad is not None
        assert policy.policy.null_msg.grad.abs().sum() > 0
    finally:
        env.close()


def test_policy_null_msg_does_not_replace_non_empty_pool(make_test_env):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=4,
            id_vectors=np.ones((3, 4), dtype=np.float32),
            null_msg_learnable=True,
            message_agg="mean",
            hidden_size=16,
        )

        hidden = _build_lstm_input(
            policy.policy, torch.as_tensor(obs[:1]), torch.zeros(1, policy.hidden_size)
        )
        hidden.sum().backward()

        assert policy.policy.null_msg.grad.abs().sum() == 0
    finally:
        env.close()


def test_attention_message_agg_uses_self_dot_product():
    class Dummy:
        pass

    policy = Dummy()
    policy.n_neighbors = 3
    policy.neighbor_cell_start = 5
    policy.cell_obs_dim = 1
    policy.hidden_size = 4
    policy.comm_enabled = True
    policy._cell_input = lambda id_embed, cell_obs, comm_msg: torch.cat(
        [id_embed, cell_obs, comm_msg], dim=-1
    )
    policy.cell_encoder = lambda x: x[..., -4:]
    policy.message_encoder = lambda x: x[..., 4:8]
    policy.update_encoder = lambda x: x[:, 4:8]
    policy.attn_query = nn.Identity()
    policy.message_agg = "attn"
    self_cell = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    neighbor_obs = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    neighbor_msg = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [9.0, 9.0, 9.0, 9.0]]]
    )
    zeros = torch.zeros(1, 3, 0)

    pooled = PolicyBackbone._encode_view(
        policy,
        self_cell,
        torch.zeros(1, 0),
        zeros,
        neighbor_obs,
        torch.tensor([[1, 2, 0]]),
        neighbor_msg,
    )
    scores = torch.tensor([[0.0, 2.0 / 4**0.5, torch.finfo(torch.float32).min]])
    expected = (neighbor_msg * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(
        PolicyBackbone._encode_view(
            policy,
            self_cell,
            torch.zeros(1, 0),
            zeros,
            torch.zeros_like(neighbor_obs),
            torch.zeros(1, 3, dtype=torch.long),
            neighbor_msg,
        ),
        torch.zeros_like(self_cell),
    )


def test_message_agg_masks_padding_by_neighbor_id():
    class Dummy:
        pass

    policy = Dummy()
    policy.n_neighbors = 2
    policy.neighbor_cell_start = 5
    policy.cell_obs_dim = 1
    policy.hidden_size = 4
    policy.comm_enabled = True
    policy._cell_input = lambda id_embed, cell_obs, comm_msg: torch.cat(
        [id_embed, cell_obs, comm_msg], dim=-1
    )
    policy.cell_encoder = lambda x: x[..., -4:]
    policy.message_encoder = lambda x: x[..., 4:8]
    policy.update_encoder = lambda x: x[:, 4:8]
    policy.message_agg = "mean"
    neighbor_obs = torch.tensor([[[0.0] * 7, [0.0] * 7]])
    neighbor_msg = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [9.0, 9.0, 9.0, 9.0]]])

    torch.testing.assert_close(
        PolicyBackbone._encode_view(
            policy,
            torch.zeros(1, 4),
            torch.zeros(1, 0),
            torch.zeros(1, 2, 0),
            neighbor_obs,
            torch.tensor([[1, 0]]),
            neighbor_msg,
        ),
        torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    )


@pytest.mark.parametrize("cell_encoder", ["linear", "raw"])
@pytest.mark.parametrize("message_agg", ["concat", "mean", "attn"])
def test_policy_cell_encoder_modes_parse_observations(
    make_test_env, message_agg: str, cell_encoder: str
):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            message_agg=message_agg,
            cell_encoder=cell_encoder,
            hidden_size=16,
            comm_enabled=True,
            comm_msg_dim=4,
        )
        hidden = _build_lstm_input(
            policy.policy,
            torch.as_tensor(obs),
            torch.zeros(env.num_agents, policy.hidden_size),
        )

        assert isinstance(
            policy.policy.cell_encoder,
            nn.Linear if cell_encoder == "linear" else nn.Identity,
        )
        assert hidden.shape == (env.num_agents, policy.hidden_size)
    finally:
        env.close()


@pytest.mark.parametrize("message_agg", ["concat", "mean", "attn"])
def test_policy_encoder_layers_configurable(make_test_env, message_agg: str):
    env = _build_env(make_test_env, mutual_obs_bit=False)
    try:
        policy = Policy(
            env,
            n_neighbors=1,
            id_dim=8,
            id_vectors=np.ones((3, 8), dtype=np.float32),
            id_feature_encoder=True,
            message_agg=message_agg,
            hidden_size=16,
            encoder_layers=3,
        )

        for encoder in (
            policy.policy.cell_encoder,
            policy.policy.message_encoder,
            policy.policy.update_encoder,
        ):
            assert (
                len([layer for layer in encoder if isinstance(layer, nn.Linear)]) == 3
            )

        id_linears = [
            layer for layer in policy.policy.id_encoder if isinstance(layer, nn.Linear)
        ]
        assert len(id_linears) == 3
    finally:
        env.close()


def test_initial_headings_affect_ego_neighbor_bearings(make_test_env):
    angle = np.deg2rad(20.0)
    env = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        self_pos_delta_obs=False,
        collision_iters=0,
        initial_headings=np.full(3, angle, dtype=np.float32),
    )
    try:
        obs, _ = env.reset(seed=0)
        np.testing.assert_allclose(
            obs[:, :3],
            [
                [0.1, -np.sin(angle), np.cos(angle)],
                [0.01, -np.sin(angle), np.cos(angle)],
                [0.01, np.sin(angle), -np.cos(angle)],
            ],
            atol=1e-5,
        )
    finally:
        env.close()


def test_ego_neighbor_obs_use_local_bearing_and_relative_heading(make_test_env):
    env = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        self_pos_delta_obs=False,
        collision_iters=0,
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.full((3, 2), [8, 5], dtype=np.int32))
        angle = np.deg2rad(20.0)
        np.testing.assert_allclose(
            obs[:, :3],
            [
                [0.1, -np.sin(angle), np.cos(angle)],
                [0.01, -np.sin(angle), np.cos(angle)],
                [0.01, np.sin(angle), -np.cos(angle)],
            ],
            atol=1e-5,
        )
        np.testing.assert_allclose(obs[:, 3:5], [[0.0, 1.0]] * 3, atol=1e-5)
    finally:
        env.close()


def test_ego_neighbor_relative_heading_rotates_with_ego_heading(make_test_env):
    ego_heading = np.deg2rad(20.0)
    neighbor_heading = np.deg2rad(50.0)
    env = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        self_pos_delta_obs=False,
        collision_iters=0,
        initial_headings=np.array([ego_heading, neighbor_heading, ego_heading]),
    )
    try:
        obs, _ = env.reset(seed=0)
        rel = neighbor_heading - ego_heading
        np.testing.assert_allclose(obs[0, 3:5], [np.sin(rel), np.cos(rel)], atol=1e-5)
    finally:
        env.close()


def test_global_rotation_preserves_local_observations_and_action_logits(make_test_env):
    positions = np.array([[28.0, 30.0], [34.0, 29.0], [33.0, 35.0]], dtype=np.float32)
    headings = np.array([0.2, 1.1, 2.4], dtype=np.float32)
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    rotated_positions = (positions - 32.0) @ rotation.T + 32.0

    kwargs = {
        "local_frame_obs": True,
        "mutual_obs_bit": True,
        "perceived_ratio_obs": True,
        "self_pos_delta_obs": False,
        "collision_iters": 0,
    }
    env = make_test_env(
        initial_positions=positions, initial_headings=headings, **kwargs
    )
    rotated_env = make_test_env(
        initial_positions=rotated_positions,
        initial_headings=headings + angle,
        **kwargs,
    )
    try:
        obs, _ = env.reset(seed=0)
        rotated_obs, _ = rotated_env.reset(seed=0)
        np.testing.assert_allclose(rotated_obs, obs, atol=1e-6)

        policy = PolicyBackbone(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            hidden_size=16,
        )
        logits, _ = policy.decode_actions(
            policy.encode_observations(torch.as_tensor(obs))
        )
        rotated_logits, _ = policy.decode_actions(
            policy.encode_observations(torch.as_tensor(rotated_obs))
        )
        _assert_logits_close(rotated_logits, logits)
    finally:
        env.close()
        rotated_env.close()


def test_world_frame_observes_xy_absolute_headings_and_xy_displacement(make_test_env):
    headings = np.array([0.2, 1.1, 2.4], dtype=np.float32)
    env = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        local_frame_obs=False,
        self_pos_delta_obs=True,
        collision_iters=0,
        initial_headings=headings,
    )
    try:
        obs, _ = env.reset(seed=0)
        np.testing.assert_allclose(
            obs[0, :5],
            [0.1, 0.0, np.sin(headings[1]), np.cos(headings[1]), 0.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            obs[0, 7:10], [np.sin(headings[0]), np.cos(headings[0]), 0.0], atol=1e-6
        )

        policy = PolicyBackbone(
            env,
            n_neighbors=1,
            id_dim=2,
            id_vectors=np.zeros((3, 2), dtype=np.float32),
            hidden_size=16,
        )
        parsed = policy._parse_observations(torch.as_tensor(obs))
        torch.testing.assert_close(
            parsed.neighbor_obs[0, 0, :5], torch.as_tensor(obs[0, :5])
        )
        torch.testing.assert_close(
            parsed.self_cell_obs[0], torch.as_tensor(obs[0, 7:10])
        )

        obs, *_ = env.step(np.full((3, 2), [8, 3], dtype=np.int32))
        final_headings = headings + np.deg2rad(20.0)
        np.testing.assert_allclose(
            obs[:, 10:12],
            np.column_stack([np.cos(final_headings), np.sin(final_headings)]) / 10.0,
            atol=1e-5,
        )

        logits, _ = policy.decode_actions(
            policy.encode_observations(torch.as_tensor(obs))
        )
        assert all(head.shape[0] == 3 for head in logits)
    finally:
        env.close()


def test_self_pos_delta_obs_is_ego_polar_after_step(make_test_env):
    env = _build_env(
        make_test_env,
        mutual_obs_bit=False,
        perceived_ratio_obs=False,
        self_pos_delta_obs=True,
        collision_iters=0,
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.full((3, 2), [8, 3], dtype=np.int32))
        np.testing.assert_allclose(obs[:, 9:12], [[0.1, 0.0, 1.0]] * 3, atol=1e-5)
    finally:
        env.close()
