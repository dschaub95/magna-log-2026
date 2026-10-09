import numpy as np
import pytest

from magna.data.processing import (
    build_target_neighbor_pair_distances,
    build_target_neighbors,
    resolve_perception_radius,
)


def _runtime_neighbors_from_obs(
    obs: np.ndarray,
    k: int,
    neighbor_obs_dim: int,
) -> np.ndarray:
    id_offset = neighbor_obs_dim - 1
    neighbors = np.full((obs.shape[0], k), -1, dtype=np.int32)
    for agent in range(obs.shape[0]):
        for slot in range(k):
            neighbor_id = int(round(obs[agent, slot * neighbor_obs_dim + id_offset]))
            if neighbor_id > 0:
                neighbors[agent, slot] = neighbor_id - 1
    return neighbors


def _assert_neighbor_sets_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    for agent, (actual_row, expected_row) in enumerate(
        zip(actual, expected, strict=True)
    ):
        np.testing.assert_array_equal(
            np.sort(actual_row[actual_row >= 0]),
            np.sort(expected_row[expected_row >= 0]),
            err_msg=f"neighbor set mismatch for agent {agent}",
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_delaunay_runtime_neighbors_match_python(make_test_env, seed: int):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 10.0, (10, 2)).astype(np.float32)
    k = 9
    positions = points + 20.0
    max_dist = float(
        np.linalg.norm(positions[:, None] - positions[None], axis=-1).max()
    )
    perception_radius = max_dist * 2.0
    py_neighbors, py_distances, k_eff = build_target_neighbors(
        points,
        k=k,
        neighbor_graph="delaunay_topk",
        perception_radius=perception_radius,
    )
    env = make_test_env(
        num_agents=points.shape[0],
        k=k_eff,
        neighbor_graph="delaunay_topk",
        width=64,
        height=64,
        initial_positions=positions,
        target_neighbors=py_neighbors,
        target_neighbor_distances=py_distances,
        perception_radius=perception_radius,
        mutual_obs_bit=False,
    )
    try:
        obs, _ = env.reset(seed=0)
        runtime_neighbors = _runtime_neighbors_from_obs(
            obs, k_eff, env.neighbor_obs_dim
        )
        _assert_neighbor_sets_equal(runtime_neighbors, py_neighbors)
    finally:
        env.close()


def test_delaunay_target_neighbors_accepted_by_env(make_test_env):
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    neighbors, distances, k_eff = build_target_neighbors(
        points, k=10, neighbor_graph="delaunay_topk"
    )
    env = make_test_env(
        num_agents=4,
        k=k_eff,
        neighbor_graph="delaunay_topk",
        initial_positions=points * 10 + 20,
        target_neighbors=neighbors,
        target_neighbor_distances=distances,
    )
    try:
        obs, _ = env.reset(seed=0)
        obs, rewards, terminals, truncations, _ = env.step(
            np.tile(np.array([[4, 2]], dtype=np.int32), (4, 1))
        )
        assert np.isfinite(obs).all()
        assert np.isfinite(rewards).all()
        assert not terminals.any()
        assert not truncations.any()
    finally:
        env.close()


def test_delaunay_env_does_not_emit_qhull_warnings_for_two_agents(make_test_env, capfd):
    env = make_test_env(
        num_agents=2,
        k=1,
        neighbor_graph="delaunay_topk",
        initial_positions=np.array([[20.0, 20.0], [24.0, 20.0]], dtype=np.float32),
        target_neighbors=np.array([[1], [0]], dtype=np.int32),
        target_neighbor_distances=np.array([[4.0], [4.0]], dtype=np.float32),
    )
    try:
        obs, _ = env.reset(seed=0)
        assert np.all(obs[:, 7] > 0.0)
    finally:
        env.close()
    assert "QH" not in capfd.readouterr().err


def test_build_target_neighbors_excludes_self_and_clamps_k():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    neighbors, distances, k_eff = build_target_neighbors(points, k=10)

    assert k_eff == 3
    assert neighbors.shape == (4, 3)
    assert distances.shape == (4, 3)
    for i in range(points.shape[0]):
        assert i not in neighbors[i]
    assert neighbors.dtype == np.int32
    assert distances.dtype == np.float32


def test_build_target_neighbors_tie_breaks_by_index():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    neighbors, distances, k_eff = build_target_neighbors(points, k=2)

    assert k_eff == 2
    np.testing.assert_array_equal(neighbors[0], np.array([1, 2], dtype=np.int32))
    np.testing.assert_allclose(distances[0], np.array([1.0, 1.0], dtype=np.float32))


def test_build_target_neighbor_pair_distances():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    neighbors = np.array([[3, 0, 2], [0, 2, 3], [0, 1, -1], [1, -1, -1]], np.int32)

    distances = build_target_neighbor_pair_distances(points, neighbors)

    assert distances.shape == (4, 3)
    assert distances.dtype == np.float32
    np.testing.assert_allclose(distances[0], [4.0, np.sqrt(20.0), 2.0])
    np.testing.assert_allclose(distances[1], [2.0, 4.0, np.sqrt(20.0)])
    assert np.isinf(distances[3]).all()


def test_build_delaunay_target_neighbors_zero_pad_to_k():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    neighbors, distances, k_eff = build_target_neighbors(
        points, k=10, neighbor_graph="delaunay_topk"
    )

    assert k_eff == 3
    assert neighbors.shape == (4, 3)
    assert np.all(distances[neighbors < 0] == 0.0)
    assert np.any(neighbors == -1)
    for i in range(points.shape[0]):
        valid = neighbors[i] >= 0
        assert i not in neighbors[i][valid]
        assert valid.sum() >= 2


def test_build_target_neighbors_respects_perception_radius():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    neighbors, distances, _ = build_target_neighbors(points, k=3, perception_radius=1.1)
    valid = neighbors >= 0
    assert np.any(~valid)
    assert np.all(distances[valid] <= 1.1)
    assert (valid.sum(axis=1) == 2).all()


def test_build_target_neighbors_raises_when_radius_isolates_agent():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]], dtype=np.float32
    )
    with pytest.raises(ValueError, match="no target neighbors"):
        build_target_neighbors(points, k=3, perception_radius=2.0)


def test_resolve_perception_radius():
    neighbors = np.array([[1, -1], [0, -1]], dtype=np.int32)
    distances = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert (
        resolve_perception_radius(
            configured_perception_radius=5.0,
            target_neighbor_distances=distances,
            target_neighbors=neighbors,
        )
        == 5.0
    )
    with pytest.raises(ValueError, match="exceeds configured perception_radius"):
        resolve_perception_radius(
            configured_perception_radius=0.5,
            target_neighbor_distances=distances,
            target_neighbors=neighbors,
        )


def test_build_target_neighbors_rejects_invalid_input():
    with pytest.raises(ValueError, match="num_agents must be > 1"):
        build_target_neighbors(np.array([[0.0, 0.0]], dtype=np.float32), k=1)
