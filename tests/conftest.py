import ctypes
import ctypes.util
import os

import anndata as ad
import numpy as np
import pytest

from magna.config import EnvConfig
from magna.data.constants import DEFAULT_COORDINATE_SCALE
from magna.env import MultiCellEnv


def write_adata(tmp_path, adata, name: str = "cells.h5ad"):
    path = tmp_path / name
    adata.write_h5ad(path)
    return path


def _display_is_usable() -> bool:
    display = os.environ.get("DISPLAY")
    if not display:
        return False

    lib_name = ctypes.util.find_library("X11")
    if lib_name is None:
        return False

    x11 = ctypes.CDLL(lib_name)
    open_display = x11.XOpenDisplay
    open_display.argtypes = [ctypes.c_char_p]
    open_display.restype = ctypes.c_void_p
    close_display = x11.XCloseDisplay
    close_display.argtypes = [ctypes.c_void_p]
    close_display.restype = ctypes.c_int

    handle = open_display(display.encode())
    if not handle:
        return False
    close_display(handle)
    return True


@pytest.fixture
def capture_available() -> None:
    if "DISPLAY" not in os.environ:
        pytest.skip("capture requires a usable DISPLAY")
    if not _display_is_usable():
        pytest.skip(f"capture requires a usable DISPLAY, got {os.environ['DISPLAY']}")


@pytest.fixture
def make_test_env():
    def _make_test_env(
        *,
        num_agents: int = 3,
        target_neighbors=None,
        target_neighbor_distances=None,
        target_neighbor_pair_distances=None,
        initial_positions=None,
        initial_headings=None,
        cell_sprite_path=None,
        cell_colors=None,
        draw_cell_ids: bool = False,
        capture_size: int = 768,
        render_mode=None,
        comm_select_action: bool = False,
        buf=None,
        seed: int = 0,
        perception_radius=None,
        **cfg_overrides,
    ) -> MultiCellEnv:
        if target_neighbors is None:
            target_neighbors = np.array([[1], [2], [1]], dtype=np.int32)
        if target_neighbor_distances is None:
            target_neighbor_distances = np.array(
                [[2.0], [2.0], [2.0]], dtype=np.float32
            )
        if initial_positions is None:
            initial_positions = np.array(
                [[20.0, 20.0], [24.0, 20.0], [28.0, 20.0]], dtype=np.float32
            )

        overrides = dict(cfg_overrides)
        if overrides.get("bounded") is False and "collision_iters" not in overrides:
            overrides["collision_iters"] = 0
        coordinate_scale = float(
            overrides.pop("coordinate_scale", DEFAULT_COORDINATE_SCALE)
        )
        cfg = EnvConfig.model_validate(
            {
                "num_envs": 1,
                "width": 64,
                "height": 64,
                "agent_radius": 4,
                "collision_iters": 1,
                "collision_grid_size": 8,
                "k": 1,
                "perception_radius": 16.0,
                "lam_nbr_reward": 0.0,
                "reward_mode": "interval_v1",
                "max_speed": 2.0,
                "time_horizon": 16,
                "reuse_initial_positions": True,
                "verbose": 0,
                "log_interval": 128,
                **overrides,
            }
        )

        return MultiCellEnv(
            cfg,
            num_agents=num_agents,
            coordinate_scale=coordinate_scale,
            target_neighbors=target_neighbors,
            target_neighbor_distances=target_neighbor_distances,
            target_neighbor_pair_distances=target_neighbor_pair_distances,
            initial_positions=initial_positions,
            initial_headings=initial_headings,
            cell_sprite_path=cell_sprite_path,
            cell_colors=cell_colors,
            draw_cell_ids=draw_cell_ids,
            capture_size=capture_size,
            render_mode=render_mode,
            comm_select_action=comm_select_action,
            buf=buf,
            seed=seed,
            perception_radius=perception_radius,
        )

    return _make_test_env


@pytest.fixture
def adata():
    adata = ad.AnnData(
        X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32)
    )
    adata.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(
        np.float32
    )
    adata.obsm["spatial"] = np.array(
        [[8.0, 12.0], [16.0, 20.0], [28.0, 36.0]], dtype=np.float32
    )

    return adata
