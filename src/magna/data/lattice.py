from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from .constants import (
    DEFAULT_INIT_OBSM_KEY,
    DEFAULT_OBSM_KEY,
)
from .init_positions import InitDistribution, build_init_positions


def build_lattice_target_positions(
    *,
    grid_size: int,
    spacing: float,
    spatial_dim: int = 2,
) -> np.ndarray:
    if spatial_dim not in (2, 3):
        raise ValueError("spatial_dim must be 2 or 3")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    grid = np.arange(grid_size, dtype=np.float32)
    axes = np.meshgrid(*([grid] * spatial_dim), indexing="xy")
    return (np.stack(axes, axis=-1) * spacing).reshape(-1, spatial_dim)


def generate_lattice_adata(
    *,
    grid_size: int,
    spacing: float,
    spatial_dim: int = 2,
    obsm_key: str = DEFAULT_OBSM_KEY,
    init_obsm_key: str | None = None,
    init_distribution: InitDistribution | None = None,
    init_seed: int | None = 0,
    init_std: float | None = None,
    init_margin: float = 0.5,
    save_path: str | Path | None = None,
) -> ad.AnnData:
    positions = build_lattice_target_positions(
        grid_size=grid_size, spacing=spacing, spatial_dim=spatial_dim
    )
    adata = ad.AnnData(X=sparse.csc_matrix((positions.shape[0], 0), dtype=np.float32))
    adata.obsm[obsm_key] = positions
    if init_distribution is not None:
        adata.obsm[init_obsm_key or DEFAULT_INIT_OBSM_KEY] = build_init_positions(
            positions,
            init_distribution,
            seed=init_seed,
            std=init_std,
            margin=init_margin,
        )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(save_path)
    return adata
