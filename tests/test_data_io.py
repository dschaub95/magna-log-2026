import anndata as ad
import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.spatial import KDTree
from tests.conftest import write_adata

from magna.data.io import (
    load_adata_cell_positions_and_colors,
    load_adata_id_features,
    load_spatial_data,
)
from magna.data.processing import resolve_id_features


def _dummy_ann_data(embedding):
    return type(
        "DummyAnnData",
        (),
        {
            "n_obs": 3,
            "X": np.zeros((3, 2), dtype=np.float32),
            "obsm": {"embedding": embedding},
        },
    )()


def test_load_adata_cell_positions_and_colors_supports_main_optional_inputs(adata):
    adata.obsm["custom"] = np.array(
        [[10.0, 14.0, 1.0], [18.0, 22.0, 2.0], [26.0, 30.0, 3.0]], dtype=np.float32
    )
    adata.obsm["spatial_target"] = np.array(
        [[40.0, 44.0], [48.0, 52.0], [56.0, 60.0]], dtype=np.float32
    )
    adata.obsm["spatial_init"] = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
    )
    adata.obsm["heading_init"] = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    adata.obs["cell_type"] = ["A", "B", "A"]

    targets, initial, colors, headings = load_adata_cell_positions_and_colors(
        adata,
        obsm_key="custom",
        target_obsm_key="spatial_target",
        init_obsm_key="spatial_init",
        init_heading_obsm_key="heading_init",
        color_key="cell_type",
    )

    np.testing.assert_allclose(targets, adata.obsm["spatial_target"])
    np.testing.assert_allclose(initial, adata.obsm["spatial_init"])
    np.testing.assert_allclose(headings, adata.obsm["heading_init"][:, 0])
    assert colors.tolist() == ["A", "B", "A"]


@pytest.mark.parametrize(
    ("mutate", "kwargs", "error", "match"),
    [
        (
            lambda adata: adata.obsm.pop("spatial"),
            {},
            KeyError,
            "missing obsm\\['spatial'\\]",
        ),
        (
            lambda adata: adata.obsm.__setitem__(
                "spatial", np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
            ),
            {},
            ValueError,
            "must have shape",
        ),
        (
            lambda adata: adata.obsm.__setitem__(
                "heading", np.ones((3, 2), dtype=np.float32)
            ),
            {"init_heading_obsm_key": "heading"},
            ValueError,
            "must have shape",
        ),
        (
            lambda adata: None,
            {"color_key": "cell_type"},
            KeyError,
            "missing obs\\['cell_type'\\]",
        ),
    ],
)
def test_load_adata_cell_positions_and_colors_errors(
    adata, mutate, kwargs, error, match
):
    mutate(adata)
    with pytest.raises(error, match=match):
        load_adata_cell_positions_and_colors(adata, **kwargs)


@pytest.mark.parametrize("source", ["X", "embedding"])
def test_load_adata_id_features_loads_dense_or_sparse_sources(adata, source):
    expected = np.asarray(adata.X, dtype=np.float32)
    adata.X = sparse.csr_matrix(expected)
    adata.obsm["embedding"] = sparse.csr_matrix(
        np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    )

    loaded = load_adata_id_features(adata, id_features=source)

    np.testing.assert_allclose(
        loaded,
        expected
        if source == "X"
        else np.asarray(adata.obsm["embedding"].toarray(), dtype=np.float32),
    )


def test_load_adata_id_features_errors(adata):
    adata.obsm.clear()
    with pytest.raises(KeyError, match="missing obsm\\['embedding'\\]"):
        load_adata_id_features(adata, id_features="embedding")

    adata.obsm["embedding"] = np.zeros((3, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="id features must have shape"):
        load_adata_id_features(adata, id_features="embedding")

    with pytest.raises(ValueError, match="id features must have shape"):
        load_adata_id_features(
            _dummy_ann_data(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            id_features="embedding",
        )


def test_resolve_id_features_falls_back_to_target_derived_features():
    targets = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    id_features = resolve_id_features(loaded_id_features=None, targets=targets)
    assert id_features.shape == targets.shape
    assert id_features.dtype == np.float32


def test_resolve_id_features_uses_no_static_features_for_learned_ids():
    targets = np.zeros((3, 2), dtype=np.float32)
    id_features = resolve_id_features(
        loaded_id_features=None, targets=targets, learned_id_dim=4
    )
    assert id_features.shape == (3, 0)


def test_loaders_work_after_roundtrip_h5ad(tmp_path, adata):
    path = write_adata(tmp_path, adata)
    targets, _, _, _ = load_adata_cell_positions_and_colors(ad.read_h5ad(path))
    np.testing.assert_allclose(targets, adata.obsm["spatial"])


def test_load_pt_coordinates_normalizes_source_spacing(tmp_path):
    path = tmp_path / "coordinates.pt"
    torch.save(torch.tensor([[0.0, 0.0], [2.0, 0.0], [8.0, 0.0]]), path)

    coordinates = load_spatial_data(str(path)).obsm["spatial"]

    distances, _ = KDTree(coordinates).query(coordinates, k=2)
    assert distances[:, 1].mean() == pytest.approx(12.0)


def test_load_pt_coordinates_rejects_invalid_payload(tmp_path):
    path = tmp_path / "coordinates.pt"
    torch.save(torch.zeros(3), path)

    with pytest.raises(ValueError, match="shape"):
        load_spatial_data(str(path))


def test_load_h5ad_coordinates_uses_same_spacing_normalization(tmp_path, adata):
    adata.obsm["spatial_init"] = adata.obsm["spatial"] * 2
    loaded = load_spatial_data(
        str(write_adata(tmp_path, adata)), init_obsm_key="spatial_init"
    )
    coordinates = loaded.obsm["spatial"]

    distances, _ = KDTree(coordinates).query(coordinates, k=2)
    assert distances[:, 1].mean() == pytest.approx(12.0)
    np.testing.assert_allclose(loaded.obsm["spatial_init"], coordinates * 2)


def test_load_h5ad_preserves_coordinates_already_spaced_at_target(tmp_path):
    coordinates = np.array([[0.0, 0.0], [12.0, 0.0], [24.0, 0.0]])
    path = write_adata(tmp_path, ad.AnnData(obsm={"spatial": coordinates}))

    np.testing.assert_array_equal(
        load_spatial_data(str(path)).obsm["spatial"], coordinates
    )
