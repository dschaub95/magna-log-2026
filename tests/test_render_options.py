import numpy as np
import pufferlib
import pytest
from pydantic import ValidationError


def test_recenter_on_step_false_steps_without_error(make_test_env):
    env = make_test_env(recenter_on_step=False)
    try:
        env.reset()
        actions = np.array([[4, 4], [4, 4], [4, 4]], dtype=np.int32)
        env.step(actions)
    finally:
        env.close()


def test_uniform_hex_cell_color_is_broadcast_to_all_agents(make_test_env):
    env = make_test_env(cell_colors="#0f7f2a")
    try:
        expected = np.array([15, 127, 42], dtype=np.uint8)
        assert env._cell_colors.shape == (1, 3, 3)
        np.testing.assert_array_equal(env._cell_colors[0], np.tile(expected, (3, 1)))
    finally:
        env.close()


def test_uniform_rgb_triplet_is_broadcast_to_all_agents(make_test_env):
    env = make_test_env(cell_colors=np.array([7, 8, 9]))
    try:
        expected = np.array([7, 8, 9], dtype=np.uint8)
        assert env._cell_colors.shape == (1, 3, 3)
        np.testing.assert_array_equal(env._cell_colors[0], np.tile(expected, (3, 1)))
    finally:
        env.close()


def test_per_agent_cell_colors_are_preserved(make_test_env):
    cell_colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        dtype=np.uint8,
    )
    env = make_test_env(cell_colors=cell_colors)
    try:
        assert env._cell_colors.shape == (1, 3, 3)
        np.testing.assert_array_equal(env._cell_colors[0], cell_colors)
    finally:
        env.close()


def test_per_env_cell_colors_are_preserved(make_test_env):
    cell_colors = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
        ],
        dtype=np.uint8,
    )
    env = make_test_env(num_envs=2, cell_colors=cell_colors)
    try:
        assert env._cell_colors.shape == (2, 3, 3)
        np.testing.assert_array_equal(env._cell_colors, cell_colors)
    finally:
        env.close()


def test_flattened_per_env_cell_colors_are_reshaped_correctly(make_test_env):
    flat_cell_colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ],
        dtype=np.uint8,
    )
    env = make_test_env(num_envs=2, cell_colors=flat_cell_colors)
    try:
        expected = flat_cell_colors.reshape(2, 3, 3)
        assert env._cell_colors.shape == (2, 3, 3)
        np.testing.assert_array_equal(env._cell_colors, expected)
    finally:
        env.close()


def test_integer_valued_float_cell_colors_are_accepted(make_test_env):
    cell_colors = np.array(
        [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]],
        dtype=np.float32,
    )
    env = make_test_env(cell_colors=cell_colors)
    try:
        assert env._cell_colors.dtype == np.uint8
        np.testing.assert_array_equal(
            env._cell_colors[0],
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )
    finally:
        env.close()


def test_cell_sprite_path_is_stored_without_rendering(make_test_env):
    env = make_test_env(cell_sprite_path="data/custom-cell.png")
    try:
        assert env._cell_sprite_path == "data/custom-cell.png"
        assert env._cell_colors is None
    finally:
        env.close()


def test_draw_cell_ids_flag_is_stored(make_test_env):
    env = make_test_env(draw_cell_ids=True)
    try:
        assert env._draw_cell_ids is True
    finally:
        env.close()


def test_draw_obs_graph_flag_is_stored(make_test_env):
    env = make_test_env(draw_obs_graph=True)
    try:
        assert env._draw_obs_graph is True
    finally:
        env.close()


def test_draw_perception_radius_flag_is_stored(make_test_env):
    env = make_test_env(draw_perception_radius=True)
    try:
        assert env._draw_perception_radius is True
    finally:
        env.close()


def test_circular_flag_is_stored(make_test_env):
    env = make_test_env(circular=True)
    try:
        assert env.circular is True
    finally:
        env.close()


def test_unbounded_reset_preserves_outside_initial_positions(make_test_env):
    initial_positions = np.array([[-40.0, 32.0], [120.0, 32.0]], dtype=np.float32)
    env = make_test_env(
        bounded=False,
        circular=True,
        num_agents=2,
        perception_radius=1000.0,
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[56.0], [56.0]], dtype=np.float32),
        initial_positions=initial_positions,
    )
    try:
        env.reset()
        np.testing.assert_allclose(env.positions()[0], initial_positions)
    finally:
        env.close()


def test_unbounded_step_allows_outward_drift(make_test_env):
    env = make_test_env(
        bounded=False,
        circular=True,
        recenter_on_step=False,
        num_agents=2,
        k=1,
        perception_radius=1000.0,
        max_speed=8.0,
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[1.0], [1.0]], dtype=np.float32),
        initial_positions=np.array([[32.0, 32.0], [33.0, 32.0]], dtype=np.float32),
    )
    try:
        env.reset()
        for _ in range(8):
            env.step(np.array([[4, 4], [4, 4]], dtype=np.int32))
        assert env.positions()[0, 0, 0] > 63.0
    finally:
        env.close()


def test_boundary_penalty_applies_only_to_clipped_agents(make_test_env):
    env = make_test_env(
        circular=False,
        boundary_penalty=-0.25,
        recenter_on_step=False,
        collision_iters=0,
        num_agents=2,
        max_speed=8.0,
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[20.0], [20.0]], dtype=np.float32),
        initial_positions=np.array([[62.0, 32.0], [40.0, 32.0]], dtype=np.float32),
        initial_headings=np.array([0.0, 0.0], dtype=np.float32),
    )
    try:
        env.reset()
        _, rewards, _, _, _ = env.step(np.array([[4, 4], [4, 4]], dtype=np.int32))
        np.testing.assert_allclose(rewards, [-0.25, 0.0])
    finally:
        env.close()


def test_capture_size_defaults_to_768(make_test_env):
    env = make_test_env()
    try:
        assert env.capture_size == 768
    finally:
        env.close()


def test_capture_size_override_is_stored(make_test_env):
    env = make_test_env(capture_size=320)
    try:
        assert env.capture_size == 320
    finally:
        env.close()


@pytest.mark.parametrize("capture_size", [0, -1, 12.5, True, None])
def test_invalid_capture_size_raises(make_test_env, capture_size):
    with pytest.raises(pufferlib.APIUsageError):
        make_test_env(capture_size=capture_size)


@pytest.mark.parametrize("circular", [None])
def test_invalid_circular_type_raises(make_test_env, circular):
    with pytest.raises((pufferlib.APIUsageError, ValidationError)):
        make_test_env(circular=circular)


def test_capture_returns_square_capture_size(make_test_env, capture_available):
    env = make_test_env(capture_size=96)
    try:
        frame = env.capture()
        assert frame.shape == (96, 96, 4)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_circular_requires_matching_width_and_height(make_test_env):
    with pytest.raises(
        (pufferlib.APIUsageError, ValidationError),
    ):
        make_test_env(width=80, height=64, circular=True)


def test_circular_capture_renders_petri_dish_palette(make_test_env, capture_available):
    env = make_test_env(circular=True, capture_size=96)
    try:
        frame = env.capture()
        outside = frame[5, 5]
        rim = frame[0, frame.shape[1] // 2]
        center = frame[frame.shape[0] // 2, frame.shape[1] // 2]

        assert int(outside[0]) > 220
        assert int(outside[1]) > 220
        assert int(outside[2]) > 220
        assert int(outside[3]) == 255

        assert int(rim[0]) < 160
        assert int(rim[1]) < 170
        assert int(rim[2]) < 175
        assert int(rim[3]) == 255

        assert int(center[0]) >= 215
        assert int(center[1]) >= 220
        assert int(center[2]) >= 205
        assert int(center[3]) == 255
    finally:
        env.close()


def test_circular_reset_clips_outside_initial_positions_into_dish(make_test_env):
    env = make_test_env(
        circular=True,
        num_agents=2,
        perception_radius=1000.0,
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[56.0], [56.0]], dtype=np.float32),
        initial_positions=np.array([[-40.0, 32.0], [120.0, 32.0]], dtype=np.float32),
    )
    try:
        obs, _ = env.reset()
        obs = obs.reshape(2, -1)
        np.testing.assert_allclose(obs[0, 0], 62.0 / 1000.0, atol=1e-4)
        np.testing.assert_allclose(obs[0, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(obs[0, 2:4], [0.0, 1.0], atol=1e-4)
        np.testing.assert_allclose(obs[1, 0], -62.0 / 1000.0, atol=1e-4)
        np.testing.assert_allclose(obs[1, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(obs[1, 2:4], [0.0, 1.0], atol=1e-4)
    finally:
        env.close()


def test_positions_returns_current_coordinates_copy(make_test_env):
    expected = np.array([[20.0, 20.0], [24.0, 20.0], [28.0, 20.0]], dtype=np.float32)
    env = make_test_env(initial_positions=expected)
    try:
        env.reset()
        positions = env.positions()
        np.testing.assert_array_equal(positions[0], expected)
        positions[:] = 0
        np.testing.assert_array_equal(env.positions()[0], expected)
    finally:
        env.close()


def test_set_positions_updates_coordinates(make_test_env):
    env = make_test_env(
        initial_positions=np.array(
            [[20.0, 20.0], [24.0, 20.0], [28.0, 20.0]], dtype=np.float32
        )
    )
    try:
        env.reset()
        updated = np.array([[30.0, 31.0], [34.0, 32.0], [38.0, 33.0]], dtype=np.float32)
        before = env.observations.copy()
        after = env.set_positions(updated)
        np.testing.assert_allclose(env.positions()[0], updated)
        assert after is env.observations
        assert not np.allclose(before, after)
    finally:
        env.close()


def test_invalid_cell_colors_raise(make_test_env):
    with pytest.raises(pufferlib.APIUsageError):
        make_test_env(cell_colors="not-a-color")


def test_cell_sprite_path_and_cell_colors_are_mutually_exclusive(make_test_env):
    with pytest.raises(pufferlib.APIUsageError):
        make_test_env(cell_sprite_path="data/custom-cell.png", cell_colors="#ff0000")


@pytest.mark.parametrize(
    "cell_colors",
    [
        np.array([[12.5, 0.0, 0.0]], dtype=np.float32),
        np.array([[-1, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.int32),
        np.array([[256, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.int32),
        b"#ff0000",
    ],
)
def test_invalid_cell_color_values_raise(make_test_env, cell_colors):
    with pytest.raises(pufferlib.APIUsageError):
        make_test_env(cell_colors=cell_colors)


def test_invalid_cell_sprite_path_type_raises(make_test_env):
    with pytest.raises(pufferlib.APIUsageError):
        make_test_env(cell_sprite_path=123)


@pytest.mark.parametrize("draw_cell_ids", [1, 0, "true", None])
def test_invalid_draw_cell_ids_type_raises(make_test_env, draw_cell_ids):
    with pytest.raises(pufferlib.APIUsageError):
        make_test_env(draw_cell_ids=draw_cell_ids)


@pytest.mark.parametrize("draw_obs_graph", ["not_bool", {}])
def test_invalid_draw_obs_graph_type_raises(make_test_env, draw_obs_graph):
    with pytest.raises(ValidationError):
        make_test_env(draw_obs_graph=draw_obs_graph)


@pytest.mark.parametrize("draw_perception_radius", ["not_bool", {}])
def test_invalid_draw_perception_radius_type_raises(
    make_test_env, draw_perception_radius
):
    with pytest.raises(ValidationError):
        make_test_env(draw_perception_radius=draw_perception_radius)


def test_cell_colors_buffers_match_expected_values(make_test_env):
    cell_colors = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    env = make_test_env(num_envs=2, cell_colors=cell_colors)
    try:
        np.testing.assert_array_equal(env._cell_colors, cell_colors)
        assert env._cell_colors.flags.c_contiguous
        assert env._cell_colors[0].flags.c_contiguous
        assert env._cell_colors[1].flags.c_contiguous
    finally:
        env.close()
