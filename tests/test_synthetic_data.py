import anndata as ad
import numpy as np
import pytest
from tests.conftest import write_adata

from magna.config import Config
from magna.data.init_positions import build_init_positions
from magna.data.io import (
    load_adata_cell_positions_and_colors,
)
from magna.data.lattice import (
    build_lattice_target_positions,
    generate_lattice_adata,
)


def test_lattice_generators_cover_positions_and_persistence(tmp_path):
    np.testing.assert_allclose(
        build_lattice_target_positions(grid_size=2, spacing=20.0),
        np.array(
            [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]], dtype=np.float32
        ),
    )
    cube = build_lattice_target_positions(grid_size=2, spacing=20.0, spatial_dim=3)
    assert cube.shape == (8, 3)
    np.testing.assert_allclose(cube.max(axis=0), [20.0, 20.0, 20.0])

    path = tmp_path / "square.h5ad"
    adata = generate_lattice_adata(grid_size=3, spacing=10.0, save_path=path)
    assert adata.n_obs == 9 and path.exists()
    targets, _, _, _ = load_adata_cell_positions_and_colors(ad.read_h5ad(path))
    np.testing.assert_allclose(np.diff(np.unique(targets[:, 0])), [10.0, 10.0])
    np.testing.assert_allclose(np.diff(np.unique(targets[:, 1])), [10.0, 10.0])

    cube_path = tmp_path / "cube.h5ad"
    cube_adata = generate_lattice_adata(
        grid_size=3, spacing=10.0, spatial_dim=3, save_path=cube_path
    )
    assert cube_adata.n_obs == 27 and cube_path.exists()
    cube_targets, _, _, _ = load_adata_cell_positions_and_colors(
        ad.read_h5ad(cube_path)
    )
    assert cube_targets.shape == (27, 3)
    np.testing.assert_allclose(np.diff(np.unique(cube_targets[:, 2])), [10.0, 10.0])


def test_config_resolves_h5ad_agent_count_and_minibatch(tmp_path, adata):
    cfg = Config.model_validate(
        {
            "data": {
                "path": str(write_adata(tmp_path, adata)),
                "coordinate_scale": 4.0,
            },
            "vector": {"num_envs": 8},
            "train": {"bptt_horizon": 16, "auto_minibatch_divisor": 2},
        }
    ).resolve_dependent_defaults()
    assert cfg.train.minibatch_size == 8 * 3 * 16 // 2
    assert cfg.train.max_minibatch_size == cfg.train.minibatch_size


def test_data_path_overrides_recompute_auto_minibatch(tmp_path, adata):
    from magna.config import (
        apply_cli_overrides,
        finalize_with_wandb_overrides,
        load_yaml_config,
    )
    from magna.data.lattice import generate_lattice_adata

    path_a = tmp_path / "a.yaml"
    path_b = write_adata(tmp_path, adata, name="b.h5ad")
    generate_lattice_adata(grid_size=4, spacing=10.0, save_path=tmp_path / "a.h5ad")
    path_a.write_text(
        "data:\n"
        f"  path: {tmp_path / 'a.h5ad'}\n"
        "vector:\n"
        "  num_envs: 8\n"
        "train:\n"
        "  minibatch_size: auto\n"
        "  max_minibatch_size: match_minibatch\n"
        "  bptt_horizon: 16\n"
        "  auto_minibatch_divisor: 2\n"
    )
    loaded = load_yaml_config(str(path_a), resolve=False)
    assert loaded.train.minibatch_size == "auto"
    cfg = apply_cli_overrides(loaded, {"data.path": str(path_b)})
    assert cfg.train.minibatch_size == 8 * 3 * 16 // 2

    class DummyRun:
        config = {"data.path": str(path_b)}

    cfg = finalize_with_wandb_overrides(loaded, DummyRun(), override_keys={"data.path"})
    assert cfg.train.minibatch_size == 8 * 3 * 16 // 2


def test_auto_minibatch_rounds_down_to_bptt_horizon(tmp_path, adata):
    cfg = Config.model_validate(
        {
            "data": {"path": str(write_adata(tmp_path, adata))},
            "vector": {"num_envs": 1},
            "train": {"bptt_horizon": 96, "auto_minibatch_divisor": 2},
        }
    ).resolve_dependent_defaults()
    assert cfg.train.minibatch_size % cfg.train.bptt_horizon == 0


def test_auto_minibatch_ignores_stale_max_cap(tmp_path, adata):
    cfg = Config.model_validate(
        {
            "data": {"path": str(write_adata(tmp_path, adata))},
            "vector": {"num_envs": 1},
            "train": {
                "bptt_horizon": 96,
                "max_minibatch_size": 65536,
            },
        }
    ).resolve_dependent_defaults()
    assert cfg.train.minibatch_size % cfg.train.bptt_horizon == 0
    assert cfg.train.max_minibatch_size == cfg.train.minibatch_size


def test_minibatch_must_divide_bptt_horizon(tmp_path, adata):
    with pytest.raises(
        ValueError, match="minibatch_size 65536 must be divisible by bptt_horizon 96"
    ):
        Config.model_validate(
            {
                "data": {"path": str(write_adata(tmp_path, adata))},
                "model": {"comm_enabled": False},
                "train": {
                    "minibatch_size": 65536,
                    "max_minibatch_size": 65536,
                    "bptt_horizon": 96,
                },
            }
        ).resolve_dependent_defaults()


@pytest.mark.parametrize(
    ("train", "expected_minibatch", "expected_max_minibatch"),
    [
        ({"minibatch_size": 20, "max_minibatch_size": 10}, 18, 6),
        ({"batch_size": 18, "minibatch_size": 20}, 18, 18),
    ],
)
def test_comm_minibatch_alignment_respects_caps(
    tmp_path, adata, train, expected_minibatch, expected_max_minibatch
):
    cfg = Config.model_validate(
        {
            "data": {"path": str(write_adata(tmp_path, adata))},
            "model": {"comm_enabled": True},
            "train": {"device": "cpu", "bptt_horizon": 2, **train},
        }
    ).resolve_dependent_defaults()

    assert cfg.train.minibatch_size == expected_minibatch
    assert cfg.train.max_minibatch_size == expected_max_minibatch


@pytest.mark.parametrize("distribution", ["perturbed", "uniform", "cluster"])
def test_generate_lattice_adata_can_attach_init_obsm(tmp_path, distribution):
    path = tmp_path / "square_init.h5ad"
    generate_lattice_adata(
        grid_size=4, spacing=12.0, init_distribution=distribution, save_path=path
    )
    targets, initial, _, _ = load_adata_cell_positions_and_colors(
        ad.read_h5ad(path), init_obsm_key="spatial_init"
    )
    assert initial is not None and initial.shape == targets.shape


def test_build_init_positions_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="Unknown init distribution"):
        build_init_positions(
            build_lattice_target_positions(grid_size=2, spacing=10.0),
            "invalid",  # type: ignore[arg-type]
        )
