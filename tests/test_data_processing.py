import anndata as ad
import numpy as np
import pytest
from tests.conftest import write_adata

from magna.config import Config
from magna.data.constants import (
    DEFAULT_AGENT_RADIUS,
    DEFAULT_COORDINATE_SCALE,
    DEFAULT_GRID_SIZE,
)
from magna.data.io import load_adata_cell_positions_and_colors, load_spatial_data
from magna.data.lattice import (
    build_lattice_target_positions,
    generate_lattice_adata,
)
from magna.data.processing import (
    apply_coordinate_scale,
    center_positions_in_env,
    compute_min_scaled_intercell_distance,
    minimum_enclosing_circle,
    resolve_env_geometry,
    warn_if_targets_too_close,
)


def _processed_targets(cfg: Config) -> np.ndarray:
    adata = load_spatial_data(
        cfg.data.path, target_obsm_key=cfg.data.target_obsm_key or cfg.data.obsm_key
    )
    return resolve_env_geometry(
        load_adata_cell_positions_and_colors(adata, obsm_key=cfg.data.obsm_key)[0],
        coordinate_scale=cfg.data.coordinate_scale,
        width=cfg.env.width,
        height=cfg.env.height,
        auto_size_buffer=cfg.env.auto_size_buffer,
    )[0]


def test_basic_position_processing_primitives(tmp_path, adata):
    path = write_adata(tmp_path, adata)
    scaled = apply_coordinate_scale(
        load_adata_cell_positions_and_colors(ad.read_h5ad(path))[0],
        coordinate_scale=4.0,
    )
    centered = center_positions_in_env(
        build_lattice_target_positions(grid_size=2, spacing=20.0), width=100, height=60
    )

    np.testing.assert_allclose(
        scaled, np.array([[32.0, 48.0], [64.0, 80.0], [112.0, 144.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        centered,
        np.array(
            [[40.0, 20.0], [60.0, 20.0], [40.0, 40.0], [60.0, 40.0]], dtype=np.float32
        ),
    )


def test_distance_warning_and_circle_helpers(capsys):
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    assert compute_min_scaled_intercell_distance(
        positions, coordinate_scale=4.0
    ) == pytest.approx(4.0)

    warn_if_targets_too_close(
        np.array([[0.0, 0.0], [2.0, 0.0], [7.0, 0.0]], dtype=np.float32),
        coordinate_scale=1.5,
        agent_radius=2,
    )
    assert "recommended agent_radius <= 1" in capsys.readouterr().out
    warn_if_targets_too_close(
        positions, coordinate_scale=1.0, agent_radius=2, collision_iters=0
    )
    assert capsys.readouterr().out == ""

    center, radius = minimum_enclosing_circle(
        np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]], dtype=np.float32)
    )
    np.testing.assert_allclose(center, np.array([1.0, np.sqrt(3.0) / 3.0]))
    np.testing.assert_allclose(radius, 2.0 / np.sqrt(3.0))


def test_resolve_env_geometry_fixed_and_auto_modes():
    auto_size_buffer = 3.0
    fixed, width, height = resolve_env_geometry(
        np.array([[0.0, 0.0], [10.0, 0.0], [2.0, 8.0]], dtype=np.float32),
        coordinate_scale=1.0,
        width=20,
        height=20,
        circular=True,
        auto_size_buffer=auto_size_buffer,
    )
    auto_square, auto_width, auto_height = resolve_env_geometry(
        np.array([[0.0, 0.0], [10.0, 4.0], [3.0, 1.0]], dtype=np.float32),
        coordinate_scale=1.0,
        size="auto",
        agent_radius=2,
        auto_size_buffer=auto_size_buffer,
    )
    auto_circle, circle_width, circle_height = resolve_env_geometry(
        np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]], dtype=np.float32),
        coordinate_scale=1.0,
        size="auto",
        agent_radius=1,
        circular=True,
        auto_size_buffer=auto_size_buffer,
    )

    assert (width, height) == (20, 20)
    np.testing.assert_allclose(fixed.min(axis=0), np.array([5.0, 6.0]))
    np.testing.assert_allclose(fixed.max(axis=0), np.array([15.0, 14.0]))
    assert (auto_width, auto_height) == (16, 16)
    np.testing.assert_allclose(auto_square.min(axis=0), np.array([3.0, 6.0]))
    np.testing.assert_allclose(auto_square.max(axis=0), np.array([13.0, 10.0]))
    assert auto_width - auto_square.max(axis=0).max() == pytest.approx(auto_size_buffer)
    assert (
        circle_height
        == circle_width
        == int(np.ceil((2.0 * (2.0 / np.sqrt(3.0))) + 2.0 * auto_size_buffer))
    )
    env_center = np.array([circle_width * 0.5, circle_height * 0.5], dtype=np.float32)
    max_target_radius = float(np.max(np.linalg.norm(auto_circle - env_center, axis=1)))
    assert (circle_width * 0.5) - max_target_radius >= auto_size_buffer - 1e-5


def test_resolve_env_geometry_matches_config_pipeline(tmp_path, adata):
    adata.obsm["spatial"] = np.array([[0, 0], [12, 0], [0, 12]], dtype=np.float32)
    cfg = Config.model_validate(
        {"data": {"path": str(write_adata(tmp_path, adata)), "coordinate_scale": 4.0}}
    ).resolve_dependent_defaults()
    np.testing.assert_array_equal(
        _processed_targets(cfg),
        np.array([[360, 360], [408, 360], [360, 408]], dtype=np.float32),
    )


def test_default_lattice_pipeline_matches_previous_coordinates():
    adata = generate_lattice_adata(
        grid_size=DEFAULT_GRID_SIZE,
        spacing=3.0 * DEFAULT_AGENT_RADIUS / DEFAULT_COORDINATE_SCALE,
    )
    targets, _, _ = resolve_env_geometry(
        np.asarray(adata.obsm["spatial"], dtype=np.float32),
        coordinate_scale=DEFAULT_COORDINATE_SCALE,
        width=768,
        height=768,
        auto_size_buffer=Config().env.auto_size_buffer,
    )
    old_spacing = 3.0 * DEFAULT_AGENT_RADIUS
    start = (768.0 - (DEFAULT_GRID_SIZE - 1) * old_spacing) * 0.5
    grid_x, grid_y = np.meshgrid(
        np.arange(DEFAULT_GRID_SIZE, dtype=np.float32),
        np.arange(DEFAULT_GRID_SIZE, dtype=np.float32),
        indexing="xy",
    )
    expected = np.stack(
        (start + grid_x * old_spacing, start + grid_y * old_spacing), axis=-1
    ).reshape(DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE, 2)
    np.testing.assert_allclose(targets, expected)


def test_config_rejects_non_positive_coordinate_scale(tmp_path, adata):
    with pytest.raises(ValueError, match="greater than 0"):
        Config.model_validate(
            {"data": {"path": str(write_adata(tmp_path, adata)), "coordinate_scale": 0}}
        )
