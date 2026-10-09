"""A simple sample environment. Use this as a template for your own envs."""

import atexit
import os
import shutil
import subprocess
from numbers import Integral

import gymnasium
import numpy as np
import pufferlib
from pufferlib import APIUsageError

from magna.config import EnvConfig
from magna.env import binding

REWARD_MODE_TO_ID = {
    "interval_v1": 0,
    "interval_split": 1,
    "interval_split_matched": 2,
    "interval_v1_cached": 3,
    "interval_split_cached": 4,
}

NEIGHBOR_GRAPH_TO_ID = {
    "radius_topk": 0,
    "delaunay_topk": 1,
}

RESET_DISTRIBUTION_TO_ID = {
    "sunflower": 0,
    "gaussian": 1,
}


def _parse_color_string(color: str) -> tuple[int, int, int]:
    s = color.strip()
    if len(s) == 4 and s.startswith("#"):
        s = "".join(c * 2 for c in s[1:])
    elif len(s) == 7 and s.startswith("#"):
        s = s[1:]
    else:
        raise APIUsageError("cell_colors strings must be #RGB or #RRGGBB")
    try:
        n = int(s, 16)
    except ValueError as exc:
        raise APIUsageError(
            "cell_colors hex strings must be valid hexadecimal colors"
        ) from exc
    return n >> 16, (n >> 8) & 255, n & 255


def _normalize_cell_colors(
    cell_colors, *, num_envs: int, num_agents: int
) -> np.ndarray | None:
    if cell_colors is None:
        return None
    if isinstance(cell_colors, str):
        return np.full(
            (num_envs, num_agents, 3), _parse_color_string(cell_colors), np.uint8
        )
    if isinstance(cell_colors, (bytes, bytearray)):
        raise APIUsageError("invalid cell_colors type")

    x = np.asarray(cell_colors)
    if np.any((x < 0) | (x > 255)):
        raise APIUsageError("invalid cell_colors range")
    if np.issubdtype(x.dtype, np.floating) and not np.allclose(x, np.round(x)):
        raise APIUsageError("invalid cell_colors dtype")

    x = np.round(x).astype(np.uint8, copy=False)
    if x.shape == (3,):
        x = np.broadcast_to(x, (num_envs, num_agents, 3))
    elif x.shape == (num_agents, 3):
        x = np.broadcast_to(x, (num_envs, num_agents, 3))
    elif x.shape == (num_envs * num_agents, 3):
        x = x.reshape(num_envs, num_agents, 3)
    elif x.shape != (num_envs, num_agents, 3):
        raise APIUsageError("invalid cell_colors shape")
    return np.ascontiguousarray(x)


class MultiCellEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        cfg: EnvConfig,
        *,
        num_agents,
        coordinate_scale,
        spatial_dim=None,
        target_neighbors=None,
        target_neighbor_distances=None,
        target_neighbor_pair_distances=None,
        perception_radius=None,
        initial_positions=None,
        initial_headings=None,
        target_positions=None,
        reset_mode=0,
        cell_sprite_path=None,
        cell_colors=None,
        draw_cell_ids=False,
        draw_obs_graph=None,
        draw_perception_radius=None,
        capture_size=768,
        render_agent_radius=None,
        render_mode=None,
        comm_select_action=False,
        track_error_init=False,
        track_error_range=False,
        buf=None,
        seed=0,
    ):
        if num_agents <= 1:
            raise APIUsageError("num_agents must be > 1")
        if cfg.width is None or cfg.height is None:
            raise APIUsageError("cfg.width and cfg.height must both be set")

        env_count = cfg.num_envs
        k = min(int(cfg.k), num_agents - 1)
        perception_radius = (
            cfg.perception_radius if perception_radius is None else perception_radius
        )
        if perception_radius <= 0:
            raise APIUsageError("perception_radius must be > 0")

        def coerce(value, name, width, *, optional=False, dtype=np.float32):
            if value is None:
                if optional:
                    return None
                raise APIUsageError(f"{name} is required")
            value = np.asarray(value, dtype=dtype)
            if value.ndim == 1 and width == 1:
                value = value.reshape(-1, 1)
            if value.shape == (num_agents, width):
                value = np.broadcast_to(value, (env_count, num_agents, width))
            elif value.shape == (env_count * num_agents, width):
                value = value.reshape(env_count, num_agents, width)
            elif value.shape != (env_count, num_agents, width):
                raise APIUsageError(
                    f"{name} must have shape ({num_agents}, {width}), "
                    f"({env_count}*{num_agents}, {width}), "
                    f"or ({env_count}, {num_agents}, {width})"
                )
            return np.ascontiguousarray(value)

        target_neighbors = coerce(
            target_neighbors, "target_neighbors", k, dtype=np.int32
        )
        target_neighbor_distances = coerce(
            target_neighbor_distances, "target_neighbor_distances", k
        )
        if target_neighbor_pair_distances is not None:
            target_neighbor_pair_distances = np.asarray(
                target_neighbor_pair_distances, dtype=np.float32
            )
            pair_count = k * (k - 1) // 2
            if target_neighbor_pair_distances.shape == (num_agents, pair_count):
                target_neighbor_pair_distances = np.broadcast_to(
                    target_neighbor_pair_distances,
                    (env_count, num_agents, pair_count),
                )
            elif target_neighbor_pair_distances.shape != (
                env_count,
                num_agents,
                pair_count,
            ):
                raise APIUsageError(
                    "target_neighbor_pair_distances must have shape "
                    f"({num_agents}, {pair_count}) or "
                    f"({env_count}, {num_agents}, {pair_count})"
                )
            target_neighbor_pair_distances = np.ascontiguousarray(
                target_neighbor_pair_distances
            )
        initial_headings = coerce(
            initial_headings, "initial_headings", 1, optional=True
        )
        if target_positions is not None:
            target_positions = np.asarray(target_positions, dtype=np.float32)
            inferred_dim = target_positions.shape[-1]
            spatial_dim = inferred_dim if spatial_dim is None else spatial_dim
            if spatial_dim != inferred_dim:
                raise APIUsageError("spatial_dim must match target_positions")
            target_positions = coerce(target_positions, "target_positions", spatial_dim)
        reset_mode = int(reset_mode)
        if reset_mode not in (0, 1, 2):
            raise APIUsageError("reset_mode must be 0, 1, or 2")
        if reset_mode in (1, 2) and target_positions is None:
            raise APIUsageError("target_positions required for reset_mode")
        if initial_positions is not None:
            initial_positions = np.asarray(initial_positions, dtype=np.float32)
            inferred_dim = initial_positions.shape[-1]
            spatial_dim = inferred_dim if spatial_dim is None else spatial_dim
            if spatial_dim != inferred_dim:
                raise APIUsageError("spatial_dim must match initial_positions")
            if spatial_dim not in (2, 3):
                raise APIUsageError("initial_positions must have 2 or 3 coordinates")
            initial_positions = coerce(
                initial_positions, "initial_positions", spatial_dim
            )
        else:
            spatial_dim = spatial_dim or 2
        if spatial_dim not in (2, 3):
            raise APIUsageError("spatial_dim must be 2 or 3")
        if spatial_dim == 3 and cfg.circular and cfg.depth != cfg.width:
            raise APIUsageError(
                "circular 3D arenas require equal width, height, and depth"
            )
        if np.any(target_neighbors >= num_agents) or np.any(
            (target_neighbors != -1) & (target_neighbors < 0)
        ):
            raise APIUsageError(
                "target_neighbors entries must be in [0, num_agents) or -1"
            )

        if cell_sprite_path is not None and not isinstance(cell_sprite_path, str):
            raise APIUsageError("cell_sprite_path must be a string or None")
        if cell_sprite_path is not None and cell_colors is not None:
            raise APIUsageError(
                "cell_sprite_path and cell_colors are mutually exclusive"
            )
        if not isinstance(draw_cell_ids, (bool, np.bool_)):
            raise APIUsageError("draw_cell_ids must be a boolean")
        if draw_obs_graph is not None and not isinstance(
            draw_obs_graph, (bool, np.bool_)
        ):
            raise APIUsageError("draw_obs_graph must be a boolean")
        if draw_perception_radius is not None and not isinstance(
            draw_perception_radius, (bool, np.bool_)
        ):
            raise APIUsageError("draw_perception_radius must be a boolean")
        if (
            isinstance(capture_size, (bool, np.bool_))
            or not isinstance(capture_size, Integral)
            or capture_size <= 0
        ):
            raise APIUsageError("capture_size must be a positive integer")
        capture_size = int(capture_size)
        normalized_cell_colors = _normalize_cell_colors(
            cell_colors, num_envs=env_count, num_agents=num_agents
        )

        comm_select_action = bool(comm_select_action)
        local_frame_obs = bool(cfg.local_frame_obs)
        vec_dim = spatial_dim + int(local_frame_obs)
        neighbor_obs_dim = (
            3 * spatial_dim
            + local_frame_obs
            + int(cfg.mutual_obs_bit)
            + int(cfg.target_neighbor_bit)
            + int(comm_select_action)
        )
        cell_obs_dim = 1 if local_frame_obs else 2 * spatial_dim - 1
        self_measures_dim = (
            cell_obs_dim
            + int(cfg.perceived_ratio_obs)
            + vec_dim * int(cfg.self_pos_delta_obs)
            + int(cfg.boundary_dist_obs)
            + vec_dim * int(cfg.density_grad_obs)
        )
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_obs_payload_dim = neighbor_obs_dim - 1
        self.self_obs_payload_dim = self_measures_dim
        self.self_obs_dim = self_measures_dim + 1
        self.cell_obs_dim = cell_obs_dim
        self.vec_obs_dim = vec_dim
        self.neighbor_cell_start = (
            3 * spatial_dim - 1 if local_frame_obs else spatial_dim
        )
        self.target_neighbor_bit_index = (
            self.neighbor_cell_start + cell_obs_dim + int(cfg.mutual_obs_bit)
            if cfg.target_neighbor_bit
            else -1
        )
        # Local layout uses ego-polar vectors and relative headings; world layout uses
        # dx/dy vectors and absolute headings. Optional bits and ids follow each payload.
        # id indices use 0 for padding and 1..num_agents for real agents.
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(k * self.neighbor_obs_dim + self.self_obs_dim,),
            dtype=np.float32,
        )
        # 2D actions: [heading, speed]; 3D actions: [heading, pitch, speed].
        # - heading/pitch 0..8 => {-20,-15,...,+20} degrees
        # - speed_action 0..4 => delta {-2,-1,0,+1,+2}, 5 => set speed to 0
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [9] * (spatial_dim - 1) + [6] + [2] * k * comm_select_action
        )

        self.render_mode = render_mode
        self.num_agents = cfg.num_envs * num_agents
        self.agents_per_env = num_agents
        self.log_interval = cfg.log_interval
        self._track_error_init = bool(track_error_init)
        self._track_error_range = bool(track_error_range)
        self.bounded = bool(cfg.bounded)
        self.circular = bool(cfg.circular)
        self.mutual_obs_bit = bool(cfg.mutual_obs_bit)
        self.perceived_ratio_obs = bool(cfg.perceived_ratio_obs)
        self.density_grad_obs = bool(cfg.density_grad_obs)
        self.local_frame_obs = local_frame_obs
        self.self_pos_delta_obs = bool(cfg.self_pos_delta_obs)
        self.boundary_dist_obs = bool(cfg.boundary_dist_obs)
        self.comm_select_action = comm_select_action
        self._xvfb_proc = None
        self._interval_extreme_reward = 0.0
        self._target_neighbors = target_neighbors
        self._target_neighbor_distances = target_neighbor_distances
        self._target_neighbor_pair_distances = target_neighbor_pair_distances
        self._initial_positions = initial_positions
        self._initial_headings = initial_headings
        self._target_positions = target_positions
        self._reset_mode = reset_mode
        self._cell_sprite_path = cell_sprite_path
        self._cell_colors = normalized_cell_colors
        self._draw_cell_ids = draw_cell_ids
        self._draw_obs_graph = bool(
            cfg.draw_obs_graph if draw_obs_graph is None else draw_obs_graph
        )
        self._draw_perception_radius = bool(
            cfg.draw_perception_radius
            if draw_perception_radius is None
            else draw_perception_radius
        )
        self.capture_size = capture_size
        self.spatial_dim = spatial_dim

        super().__init__(buf)
        base_env_kwargs = cfg.model_dump(
            exclude={
                "size",
                "reset_init",
                "reset_rotated_target",
                "reset_interpolated",
            }
        )
        base_env_kwargs.update(
            {
                "width": cfg.width,
                "height": cfg.height,
                "depth": cfg.depth or cfg.width,
                "spatial_dim": spatial_dim,
                "capture_size": self.capture_size,
                "render_agent_radius": float(
                    render_agent_radius or cfg.render_agent_radius or cfg.agent_radius
                ),
                "num_agents": num_agents,
                "k": k,
                "perception_radius": perception_radius,
                "neighbor_graph": NEIGHBOR_GRAPH_TO_ID[cfg.neighbor_graph],
                "reset_distribution": RESET_DISTRIBUTION_TO_ID[cfg.reset_distribution],
                "reset_mode": self._reset_mode,
                "coordinate_scale": coordinate_scale,
                "reward_mode": REWARD_MODE_TO_ID[cfg.reward_mode],
                "draw_cell_ids": self._draw_cell_ids,
                "draw_obs_graph": self._draw_obs_graph,
                "draw_perception_radius": self._draw_perception_radius,
                "comm_select_action": self.comm_select_action,
            }
        )
        c_envs = []
        for i in range(cfg.num_envs):
            env_target_neighbors = self._target_neighbors[i]
            env_target_neighbor_distances = self._target_neighbor_distances[i]
            env_kwargs = dict(base_env_kwargs)
            env_kwargs["target_neighbors"] = env_target_neighbors
            env_kwargs["target_neighbor_distances"] = env_target_neighbor_distances
            env_kwargs["target_neighbor_pair_distances"] = (
                None
                if self._target_neighbor_pair_distances is None
                else self._target_neighbor_pair_distances[i]
            )
            if self._initial_positions is not None:
                env_kwargs["initial_positions"] = self._initial_positions[i]
            if self._initial_headings is not None:
                env_kwargs["initial_headings"] = self._initial_headings[i].reshape(-1)
            if self._target_positions is not None:
                env_kwargs["target_positions"] = self._target_positions[i]
            if self._cell_sprite_path is not None:
                env_kwargs["cell_sprite_path"] = self._cell_sprite_path
            if self._cell_colors is not None:
                env_kwargs["cell_colors"] = self._cell_colors[i]
            c_env = binding.env_init(
                self.observations[i * num_agents : (i + 1) * num_agents],
                self.actions[i * num_agents : (i + 1) * num_agents],
                self.rewards[i * num_agents : (i + 1) * num_agents],
                self.terminals[i * num_agents : (i + 1) * num_agents],
                self.truncations[i * num_agents : (i + 1) * num_agents],
                seed,
                **env_kwargs,
            )
            c_envs.append(c_env)

        self._c_envs = c_envs
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self._interval_extreme_reward = 0.0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        idx = np.argmax(np.abs(self.rewards))
        if abs(self.rewards[idx]) > abs(self._interval_extreme_reward):
            self._interval_extreme_reward = float(self.rewards[idx])

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log and not self._track_error_init:
                log.pop("error_init", None)
            if log and not self._track_error_range:
                log.pop("error_min", None)
                log.pop("error_max", None)
            if log:
                log["environment/extreme_reward"] = self._interval_extreme_reward
                self._interval_extreme_reward = 0.0
                info.append(log)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        self._ensure_xvfb()
        binding.vec_render(self.c_envs, 0)

    def capture(self, env_id=0):
        self._ensure_xvfb()
        return binding.vec_capture(self.c_envs, env_id)

    def positions(self):
        return np.stack([binding.env_get(env)["positions"] for env in self._c_envs])

    def set_positions(self, positions):
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim == 2:
            pos = pos[None]
        for env, p in zip(self._c_envs, pos, strict=True):
            binding.env_put(env, positions=p)
        return self.observations

    def close(self):
        binding.vec_close(self.c_envs)
        self._shutdown_xvfb()

    def _ensure_xvfb(self):
        if os.environ.get("DISPLAY") or self._xvfb_proc:
            return
        if shutil.which("Xvfb") is None:
            raise RuntimeError(
                "DISPLAY is not set and Xvfb is not installed. Install Xvfb or set DISPLAY."
            )

        # -displayfd picks a free display and writes it once the server is ready.
        r, w = os.pipe()
        self._xvfb_proc = subprocess.Popen(
            ["Xvfb", "-displayfd", str(w), "-screen", "0", "1920x1080x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(w,),
        )
        os.close(w)
        with os.fdopen(r) as f:
            os.environ["DISPLAY"] = f":{f.readline().strip()}"
        atexit.register(self._shutdown_xvfb)

    def _shutdown_xvfb(self):
        if self._xvfb_proc is None:
            return
        self._xvfb_proc.terminate()
        self._xvfb_proc = None
