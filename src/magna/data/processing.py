import numpy as np
from scipy.spatial import Delaunay, KDTree


def _positions(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (n_positions, 2 or 3)")
    return x


def _pair_circle(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    center = (a + b) * 0.5
    return center.astype(np.float32), float(np.linalg.norm(a - center))


def _triple_circle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, float] | None:
    ax, ay, bx, by, cx, cy = map(float, (*a, *b, *c))
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) <= 1e-12:
        return None
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d
    uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d
    center = np.array([ux, uy], dtype=np.float32)
    return center, float(np.linalg.norm(a - center))


def _in_circle(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
    return float(np.linalg.norm(point - center)) <= radius + 1e-5


def _center(
    positions: np.ndarray,
    center: np.ndarray,
    *,
    width: int,
    height: int,
    depth: int | None = None,
) -> np.ndarray:
    env_center = np.array(
        [
            float(width) * 0.5,
            float(height) * 0.5,
            float(depth or min(width, height)) * 0.5,
        ],
        dtype=np.float32,
    )[: positions.shape[1]]
    offset = env_center - center
    return np.ascontiguousarray(positions + offset, dtype=np.float32)


def apply_coordinate_scale(
    positions: np.ndarray, *, coordinate_scale: float = 1.0
) -> np.ndarray:
    if coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be > 0")
    scaled = np.asarray(positions, dtype=np.float32) * float(coordinate_scale)
    return np.ascontiguousarray(scaled, dtype=np.float32)


def center_positions_in_env(
    positions: np.ndarray, *, width: int, height: int, depth: int | None = None
) -> np.ndarray:
    positions = _positions(positions)
    bbox_center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    return _center(positions, bbox_center, width=width, height=height, depth=depth)


def center_positions_in_circle(
    positions: np.ndarray, *, width: int, height: int, source_center: np.ndarray
) -> np.ndarray:
    positions = _positions(positions)
    if positions.shape[1] != 2:
        raise ValueError("center_positions_in_circle requires 2D positions")
    source_center = np.asarray(source_center, dtype=np.float32)
    if source_center.shape != (2,):
        raise ValueError("source_center must have shape (2,)")
    return _center(positions, source_center, width=width, height=height)


def minimum_enclosing_circle(positions: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.array(_positions(positions), copy=True)
    if points.shape[1] != 2:
        raise ValueError("minimum_enclosing_circle requires 2D positions")
    if not len(points):
        raise ValueError("positions must contain at least 1 row")
    points = points[np.random.default_rng(0).permutation(points.shape[0])]
    center, radius = points[0].copy(), 0.0
    for i, p in enumerate(points):
        if _in_circle(p, center, radius):
            continue
        center, radius = p.copy(), 0.0
        for j, q in enumerate(points[:i]):
            if _in_circle(q, center, radius):
                continue
            center, radius = _pair_circle(p, q)
            for r in points[:j]:
                if _in_circle(r, center, radius):
                    continue
                center, radius = _triple_circle(p, q, r) or max(
                    (_pair_circle(p, q), _pair_circle(p, r), _pair_circle(q, r)),
                    key=lambda item: item[1],
                )
    return center.astype(np.float32), float(radius)


def _fixed_dimensions(
    *, width: int | None, height: int | None, size: int | str | None
) -> tuple[int, int] | None:
    if width is not None and height is not None:
        return int(width), int(height)
    if size == "auto":
        return None
    if size is None:
        raise ValueError("width/height or size must be provided")
    size = int(size)
    return size, size


def _auto_square_size(
    positions: np.ndarray, agent_radius: int, *, auto_size_buffer: float
) -> int:
    span = positions.max(axis=0) - positions.min(axis=0)
    size = int(np.ceil(float(np.max(span)) + 2.0 * auto_size_buffer))
    minimum_size = 2 * int(agent_radius)
    return max(size, minimum_size)


def compute_min_scaled_intercell_distance(
    positions: np.ndarray, *, coordinate_scale: float
) -> float:
    scaled = _positions(
        apply_coordinate_scale(positions, coordinate_scale=coordinate_scale)
    )
    if scaled.shape[0] <= 1:
        raise ValueError("positions must contain at least 2 rows")
    distances, _ = KDTree(scaled).query(scaled, k=2, workers=-1)
    return float(np.min(np.asarray(distances, dtype=np.float32)[:, 1]))


def warn_if_targets_too_close(
    positions: np.ndarray,
    *,
    coordinate_scale: float,
    agent_radius: int,
    collision_iters: int = 1,
) -> None:
    if collision_iters == 0:
        return
    min_distance = compute_min_scaled_intercell_distance(
        np.asarray(positions)[:, :2], coordinate_scale=coordinate_scale
    )
    min_achievable_distance = 2.0 * float(agent_radius)
    if min_distance >= min_achievable_distance:
        return
    recommended_radius = min_distance * 0.5
    print(
        "⚠️ Warning smallest inter-cell distance "
        f"({min_distance:.3f}) is smaller than the minimal achievable distance "
        f"for agent_radius={agent_radius} ({min_achievable_distance:.3f}) after "
        f"applying coordinate_scale={float(coordinate_scale):.3f}; "
        f"recommended agent_radius <= {max(1, int(np.floor(recommended_radius)))} "
        f"(exact max {recommended_radius:.3f})."
    )


def build_target_neighbors(
    target_positions: np.ndarray,
    k: int,
    *,
    neighbor_graph: str = "radius_topk",
    perception_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(target_positions, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] not in (2, 3):
        raise ValueError("target_positions must have shape (num_agents, 2 or 3)")
    if k <= 0:
        raise ValueError("k must be > 0")
    if x.shape[0] <= 1:
        raise ValueError("num_agents must be > 1")
    if neighbor_graph not in {"radius_topk", "delaunay_topk"}:
        raise ValueError("neighbor_graph must be 'radius_topk' or 'delaunay_topk'")
    if x.shape[1] == 3 and neighbor_graph == "delaunay_topk":
        raise ValueError("delaunay_topk is only supported for 2D positions")

    n = x.shape[0]
    k_eff = min(k, n - 1)
    if neighbor_graph == "delaunay_topk":
        indptr, vertex_indices = Delaunay(x).vertex_neighbor_vertices
        neighbor_lists = [vertex_indices[indptr[i] : indptr[i + 1]] for i in range(n)]
    else:
        _, indices = KDTree(x).query(x, k=k_eff + 1)
        neighbor_lists = [indices[i, 1 : k_eff + 1] for i in range(n)]

    neighbors = np.full((n, k_eff), -1, dtype=np.int32)
    distances = np.zeros((n, k_eff), dtype=np.float32)
    for i, nbrs in enumerate(neighbor_lists):
        nbr_distances = np.linalg.norm(x[nbrs] - x[i], axis=1).astype(np.float32)
        if perception_radius is not None:
            nbrs = nbrs[nbr_distances <= perception_radius]
            nbr_distances = nbr_distances[nbr_distances <= perception_radius]
        order = np.lexsort((nbrs, nbr_distances))
        take = min(k_eff, len(nbrs))
        neighbors[i, :take] = nbrs[order][:take]
        distances[i, :take] = nbr_distances[order][:take]
    if np.any(np.all(neighbors < 0, axis=1)):
        raise ValueError(
            "perception_radius leaves at least one agent with no target neighbors"
        )
    degree = (neighbors >= 0).sum(axis=1)
    print(
        f"target neighbor graph ({neighbor_graph}, k={k_eff}): "
        f"degree min={degree.min()} mean={degree.mean():.2f} max={degree.max()}"
    )
    return neighbors, distances, k_eff


def build_target_neighbor_pair_distances(
    target_positions: np.ndarray, target_neighbors: np.ndarray
) -> np.ndarray:
    x = _positions(target_positions)
    neighbors = np.asarray(target_neighbors, dtype=np.int32)
    if neighbors.ndim != 2 or neighbors.shape[0] != x.shape[0]:
        raise ValueError("target_neighbors must have shape (num_agents, k)")

    valid = neighbors >= 0
    coords = x[np.where(valid, neighbors, 0)]
    distances = np.linalg.norm(coords[:, :, None] - coords[:, None], axis=-1)
    distances[~(valid[:, :, None] & valid[:, None, :])] = np.inf
    pair_distances = distances[:, *np.triu_indices(neighbors.shape[1], k=1)]
    return np.ascontiguousarray(pair_distances, dtype=np.float32)


def resolve_perception_radius(
    *,
    configured_perception_radius: float,
    target_neighbor_distances: np.ndarray,
    target_neighbors: np.ndarray | None = None,
    epsilon: float = 1e-3,
) -> float:
    distances = np.asarray(target_neighbor_distances, dtype=np.float32)
    if target_neighbors is not None:
        valid = np.asarray(target_neighbors) >= 0
        if not np.any(valid):
            raise ValueError("target_neighbors has no valid entries")
        distances = distances[valid]
    if float(np.max(distances)) > configured_perception_radius + epsilon:
        raise ValueError(
            "target neighbor distance exceeds configured perception_radius "
            f"({configured_perception_radius})"
        )
    return configured_perception_radius


def resolve_env_geometry(
    positions: np.ndarray,
    *,
    coordinate_scale: float,
    width: int | None = None,
    height: int | None = None,
    depth: int | None = None,
    size: int | str | None = None,
    agent_radius: int = 0,
    circular: bool = False,
    auto_size_buffer: float,
) -> tuple[np.ndarray, int, int]:
    scaled = _positions(
        apply_coordinate_scale(positions, coordinate_scale=coordinate_scale)
    )
    fixed = _fixed_dimensions(width=width, height=height, size=size)
    if fixed is not None:
        width, height = fixed
        return (
            center_positions_in_env(scaled, width=width, height=height, depth=depth),
            width,
            height,
        )
    if circular and scaled.shape[1] == 2:
        source_center, radius = minimum_enclosing_circle(scaled)
        size = max(
            int(np.ceil(2.0 * radius + 2.0 * auto_size_buffer)),
            2 * int(agent_radius),
        )
        return (
            center_positions_in_circle(
                scaled, width=size, height=size, source_center=source_center
            ),
            size,
            size,
        )
    if circular:
        center = (scaled.min(axis=0) + scaled.max(axis=0)) * 0.5
        radius = float(np.linalg.norm(scaled - center, axis=1).max())
        size = max(
            int(np.ceil(2.0 * radius + 2.0 * auto_size_buffer)),
            2 * int(agent_radius),
        )
        return _center(scaled, center, width=size, height=size, depth=size), size, size
    size = _auto_square_size(scaled, agent_radius, auto_size_buffer=auto_size_buffer)
    return (
        center_positions_in_env(scaled, width=size, height=size, depth=size),
        size,
        size,
    )


def build_default_id_features_from_targets(targets: np.ndarray) -> np.ndarray:
    targets = _positions(targets)
    centered = targets - targets.mean(axis=0, keepdims=True)
    max_norm = float(np.max(np.linalg.norm(centered, axis=1))) if centered.size else 0.0
    if max_norm <= 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    return (centered / max_norm).astype(np.float32)


def resolve_id_features(
    *,
    loaded_id_features: np.ndarray | None,
    targets: np.ndarray,
    learned_id_dim: int = 0,
) -> np.ndarray:
    if loaded_id_features is not None:
        return np.ascontiguousarray(loaded_id_features, dtype=np.float32)
    if learned_id_dim:
        return np.empty((len(_positions(targets)), 0), dtype=np.float32)
    return build_default_id_features_from_targets(targets)
