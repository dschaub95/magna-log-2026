from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import KDTree

from .constants import DEFAULT_OBSM_KEY, DEFAULT_TARGET_SPACING


def load_pt_coordinates(path: str, *, mmap: bool = False) -> torch.Tensor:
    coordinates = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if (
        not isinstance(coordinates, torch.Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[1] not in (2, 3)
        or len(coordinates) < 2
    ):
        raise ValueError(
            ".pt coordinate files must contain a tensor with shape (n_cells, 2 or 3)"
        )
    return coordinates


def load_spatial_data(
    path: str,
    *,
    target_obsm_key: str = DEFAULT_OBSM_KEY,
    init_obsm_key: str | None = None,
    target_spacing: float = DEFAULT_TARGET_SPACING,
) -> ad.AnnData:
    if Path(path).suffix.lower() == ".pt":
        coordinates = load_pt_coordinates(path)
        adata = ad.AnnData(
            obsm={target_obsm_key: np.asarray(coordinates.float().numpy())}
        )
    else:
        adata = ad.read_h5ad(path)

    coordinates = _obsm(adata, target_obsm_key, None)
    if len(coordinates) < 2:
        raise ValueError("Resolved target positions must contain at least 2 cells")
    mean_spacing = float(KDTree(coordinates).query(coordinates, k=2)[0][:, 1].mean())
    if mean_spacing <= 0:
        raise ValueError(
            "Target coordinates must have positive nearest-neighbor spacing"
        )
    scale = 1.0
    if not np.isclose(mean_spacing, target_spacing):
        scale = target_spacing / mean_spacing
        print(
            f"Normalizing input coordinate spacing from {mean_spacing:.3f} "
            f"to {target_spacing:.3f} (scale={scale:.3f})."
        )
    adata.obsm[target_obsm_key] = coordinates * scale
    if init_obsm_key is not None:
        adata.obsm[init_obsm_key] = _obsm(adata, init_obsm_key, None) * scale
    return adata


def _obsm(adata: ad.AnnData, key: str, cols: int | None) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"AnnData object is missing obsm['{key}']")
    values = np.asarray(adata.obsm[key], dtype=np.float32)
    if cols == 1:
        if values.ndim not in (1, 2) or values.ndim == 2 and values.shape[1] != 1:
            raise ValueError(
                f"adata.obsm['{key}'] must have shape (n_cells,) or (n_cells, 1), got {values.shape}"
            )
        values = values.reshape(-1)
    elif cols is None and (values.ndim != 2 or values.shape[1] not in (2, 3)):
        raise ValueError(
            f"adata.obsm['{key}'] must have shape (n_cells, 2) or (n_cells, 3), got {values.shape}"
        )
    elif cols is not None and (values.ndim != 2 or values.shape[1] < cols):
        raise ValueError(
            f"adata.obsm['{key}'] must have shape (n_cells, >={cols}), got {values.shape}"
        )
    if cols not in (None, 1):
        values = values[:, :cols]
    return np.ascontiguousarray(values, dtype=np.float32)


def _obs(adata: ad.AnnData, key: str) -> np.ndarray:
    if key not in adata.obs:
        raise KeyError(f"AnnData object is missing obs['{key}']")
    return np.asarray(adata.obs[key], dtype=object).copy()


def _features(values) -> np.ndarray:
    values = values.toarray() if sparse.issparse(values) else values
    values = np.asarray(values, dtype=np.float32)
    return np.ascontiguousarray(values, dtype=np.float32)


def load_adata_cell_positions_and_colors(
    adata: ad.AnnData,
    *,
    obsm_key: str = DEFAULT_OBSM_KEY,
    target_obsm_key: str | None = None,
    init_obsm_key: str | None = None,
    init_heading_obsm_key: str | None = None,
    color_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    targets = _obsm(adata, target_obsm_key or obsm_key, None)
    if len(targets) <= 1:
        raise ValueError("Resolved target positions must contain at least 2 cells")
    init_positions = (
        None if init_obsm_key is None else _obsm(adata, init_obsm_key, None)
    )
    init_headings = (
        None
        if init_heading_obsm_key is None
        else _obsm(adata, init_heading_obsm_key, 1)
    )
    colors = None if color_key is None else _obs(adata, color_key)
    return targets, init_positions, colors, init_headings


def load_adata_id_features(
    adata: ad.AnnData, *, id_features: str | None = None
) -> np.ndarray | None:
    if id_features is None:
        return None
    values = adata.X if id_features == "X" else adata.obsm.get(id_features)
    if values is None:
        raise KeyError(f"AnnData object is missing obsm['{id_features}']")
    features = _features(values)
    if features.ndim != 2 or not features.shape[1]:
        raise ValueError(
            f"id features must have shape (n_cells, >=1), got {features.shape}"
        )
    return features
