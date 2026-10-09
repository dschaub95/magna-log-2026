import inspect
import os
import pprint
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import cv2
import imageio.v2 as imageio
import numpy as np
import ot
import pufferlib
import torch
import wandb
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

from magna.config import Config, EnvConfig, apply_cli_overrides, load_wandb_config
from magna.data.io import (
    load_adata_cell_positions_and_colors,
    load_adata_id_features,
    load_spatial_data,
)
from magna.data.processing import (
    build_target_neighbor_pair_distances,
    build_target_neighbors,
    resolve_env_geometry,
    resolve_id_features,
    resolve_perception_radius,
    warn_if_targets_too_close,
)
from magna.env import MultiCellEnv
from magna.policy import Policy
from magna.utils import resolve_cell_colors, seed_everything

DEFAULT_INFERENCE_STEPS = 1024


def _gw(points: np.ndarray, target: np.ndarray) -> float:
    gw2 = ot.gromov.gromov_wasserstein2(
        ot.dist(points, points, metric="euclidean"),
        ot.dist(target, target, metric="euclidean"),
        ot.unif(len(points)),
        ot.unif(len(target)),
    )
    return float(np.sqrt(max(gw2, 0.0)))


def formation_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape[-1] != prediction.shape[-1]:
        raise ValueError("Coordinate arrays must share the same dimension.")
    scale = float(pdist(target).max())
    if scale == 0.0:
        raise ValueError("Target formation has zero pairwise-distance norm.")
    target, prediction = target / scale, prediction / scale
    out = {"gw": _gw(prediction, target)}
    if len(target) != len(prediction):
        return out
    target_dist, prediction_dist = pdist(target), pdist(prediction)
    error = prediction_dist - target_dist
    out.update(
        {
            "rel_err": float(np.linalg.norm(error) / np.linalg.norm(target_dist)),
            "mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error**2)),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }
    )
    return out


METRIC_KEYS = ("rel_err", "mae", "mse", "rmse", "gw")


def _downsample_frame_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        raise ValueError("max_side must be > 0")
    scale = max_side / max(frame.shape[:2])
    size = tuple(max(1, round(dim * scale)) for dim in frame.shape[1::-1])
    return (
        frame if scale >= 1 else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    )


def _resolve_rollout_steps(env_kwargs: dict, n_steps: int | None) -> int:
    if n_steps is not None:
        return int(n_steps)
    cfg = env_kwargs.get("cfg")
    time_horizon = getattr(cfg, "time_horizon", env_kwargs.get("time_horizon"))
    if time_horizon is None or int(time_horizon) <= 0:
        return DEFAULT_INFERENCE_STEPS
    return int(time_horizon)


def run_policy_inference(
    *,
    env_kwargs: dict,
    policy,
    device: str,
    seed: int,
    n_steps: int | None = None,
    n_rollouts: int = 1,
    video_path: str | Path | None = None,
    fps: int = 30,
    codec: str = "libx264",
    macro_block_size: int = 4,
    max_frame_side: int = 1024,
    collect_errors: bool = True,
    show_progress: bool = True,
    greedy: bool = False,
    return_positions: Literal["last", "all"] = "last",
    targets: np.ndarray | None = None,
) -> dict:
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be >= 1")

    rollout_steps = _resolve_rollout_steps(env_kwargs, n_steps)
    capture_video = video_path is not None
    keep_all = return_positions == "all"
    all_errors: list[float] = []
    all_error_mins: list[float] = []
    all_error_maxes: list[float] = []
    rollout_frames: list[list[np.ndarray]] = []
    rollout_metrics: list[dict[str, float | int | None]] = []
    rollout_positions: list[np.ndarray] = []

    for rollout_idx in range(n_rollouts):
        rollout_seed = int(seed + rollout_idx)
        seed_everything(rollout_seed)
        env = MultiCellEnv(**env_kwargs)
        obs, _ = env.reset(seed=rollout_seed)
        state: dict[str, torch.Tensor | None] = {"lstm_h": None, "lstm_c": None}
        frames: list[np.ndarray] = []
        final_positions = env.positions()[0]
        positions = [final_positions] if keep_all else []
        errors: list[float] = []
        error_mins: list[float] = []
        error_maxes: list[float] = []
        steps_taken = 0
        try:
            for _ in tqdm(range(rollout_steps), disable=not show_progress):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=device)
                    logits, _ = policy.forward_eval(obs_t, state)
                    if greedy:
                        action_heads = [head.argmax(dim=1) for head in logits]
                        action = torch.stack(action_heads, dim=1)
                    else:
                        action, _, _ = pufferlib.pytorch.sample_logits(logits)
                obs, _, terminals, truncations, info = env.step(action.cpu().numpy())
                steps_taken += 1
                final_positions = env.positions()[0]
                if keep_all:
                    positions.append(final_positions)
                if capture_video:
                    frame = env.capture()
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    frames.append(_downsample_frame_max_side(frame, max_frame_side))
                if collect_errors and info:
                    if isinstance(info[0], dict) and "error" in info[0]:
                        error = float(info[0]["error"])
                        errors.append(error)
                        all_errors.append(error)
                        error_min = float(info[0].get("error_min", error))
                        error_max = float(info[0].get("error_max", error))
                        error_mins.append(error_min)
                        error_maxes.append(error_max)
                        all_error_mins.append(error_min)
                        all_error_maxes.append(error_max)
                if bool(np.any(terminals)) or bool(np.any(truncations)):
                    break
        finally:
            env.close()
        rollout_frames.append(frames)
        rollout_positions.append(np.stack(positions) if keep_all else final_positions)

        geo = formation_metrics(targets, final_positions) if targets is not None else {}
        rollout_metrics.append(
            {
                "num_steps": steps_taken,
                "error_mean": float(np.mean(errors)) if errors else None,
                "error_last": float(errors[-1]) if errors else None,
                "error_min": min(error_mins) if error_mins else None,
                "error_max": max(error_maxes) if error_maxes else None,
                **geo,
            }
        )

    resolved_video_path = None
    if capture_video:
        resolved_video_path = Path(video_path)
        resolved_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            resolved_video_path,
            fps=fps,
            codec=codec,
            macro_block_size=macro_block_size,
        )
        try:
            for step_idx in range(max(map(len, rollout_frames))):
                frames = [
                    rollout[min(step_idx, len(rollout) - 1)]
                    for rollout in rollout_frames
                ]
                writer.append_data(
                    frames[0] if n_rollouts == 1 else np.concatenate(frames, axis=1)
                )
        finally:
            writer.close()

    if n_rollouts > 1:
        extras = {"per_rollout": rollout_metrics}
    elif rollout_metrics and any(k in rollout_metrics[0] for k in METRIC_KEYS):
        extras = {k: rollout_metrics[0].get(k) for k in METRIC_KEYS}
    else:
        extras = dict.fromkeys(METRIC_KEYS)
    return {
        "video_path": str(resolved_video_path) if resolved_video_path else None,
        "num_steps": max(metric["num_steps"] for metric in rollout_metrics),
        "error_mean": float(np.mean(all_errors)) if all_errors else None,
        "error_last": float(all_errors[-1]) if all_errors else None,
        "error_min": min(all_error_mins) if all_error_mins else None,
        "error_max": max(all_error_maxes) if all_error_maxes else None,
        **extras,
        "positions": rollout_positions,
    }


def run_eval(
    cfg: Config,
    checkpoint: str | Path,
    *,
    flat_overrides: dict[str, Any] | None = None,
    adata: ad.AnnData | None = None,
    video_path: str | Path = "outputs/videos/rollout_eval.mp4",
    n_rollouts: int = 1,
    greedy: bool = False,
    return_positions: Literal["last", "all"] = "last",
    verbose: bool = False,
    autoreset: bool = True,
    **env_kwargs: Any,
) -> dict[str, Any]:
    overrides = flat_overrides or {}
    cfg = apply_cli_overrides(cfg, overrides)
    (
        targets,
        initial_positions,
        target_obsm_key,
        initial_obsm_key,
        color_values,
        loaded_id_features,
        initial_headings,
    ) = load_eval_data(cfg, adata=adata)

    if verbose:
        print("==== Eval Config ====")
        pprint.pprint(cfg.model_dump())
        print("=====================")

    if (
        initial_positions is not None or initial_headings is not None
    ) and "env.reuse_initial_positions" not in overrides:
        env_kwargs["reuse_initial_positions"] = True
    if env_kwargs.get("cell_colors") is None:
        env_kwargs["cell_colors"] = resolve_cell_colors(targets, color_values)
    if not autoreset:
        env_kwargs["time_horizon"] = -1

    print(f"Checkpoint: {checkpoint}")

    n_steps = _resolve_rollout_steps({"cfg": cfg.env}, None) if not autoreset else None

    env, policy, resolved_seed, inference_env_kwargs = build_eval_env_and_policy(
        cfg,
        targets=targets,
        initial_positions=initial_positions,
        initial_headings=initial_headings,
        loaded_id_features=loaded_id_features,
        **env_kwargs,
    )
    try:
        load_policy_weights(policy, checkpoint, device=cfg.train.device)
        inference = run_policy_inference(
            env_kwargs=inference_env_kwargs,
            policy=policy,
            device=cfg.train.device,
            seed=resolved_seed,
            n_steps=n_steps,
            n_rollouts=n_rollouts,
            video_path=video_path,
            greedy=greedy,
            return_positions=return_positions,
            targets=targets,
        )
    finally:
        env.close()

    positions = inference.pop("positions")
    return {
        "cfg": cfg,
        "checkpoint_path": str(checkpoint),
        "seed": resolved_seed,
        "video_path": inference["video_path"],
        "positions": positions,
        "metrics": inference,
        "target_obsm_key": target_obsm_key,
        "initial_obsm_key": initial_obsm_key,
    }


def resolve_wandb_run_path(
    run_id: str, project: str = "magna", entity: str | None = None
) -> str:
    if "/" in run_id:
        return run_id

    entity = entity or os.environ.get("WANDB_ENTITY")
    entity = entity or getattr(wandb.Api(), "default_entity", None)
    if entity:
        return f"{entity}/{project}/{run_id}"
    raise ValueError(
        "Could not infer W&B entity. Pass `entity=...` or a full `run_id` "
        "as 'entity/project/run_id'."
    )


def load_cfg_from_wandb_run(
    run_id: str,
    *,
    project: str = "magna",
    entity: str | None = None,
    resolve: bool = True,
) -> Config:
    run_path = resolve_wandb_run_path(run_id, project=project, entity=entity)
    return load_wandb_config(run_path, resolve=resolve)


def load_policy_weights(
    policy: Policy, checkpoint_path: str | Path | None, device: str
) -> None:
    if checkpoint_path is None:
        print("No checkpoint path provided, skipping weight loading.")
    else:
        state_dict = torch.load(str(checkpoint_path), map_location=device)
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
        policy.load_state_dict(state_dict)
    policy.eval()


def load_eval_data(
    cfg: Config,
    *,
    adata: ad.AnnData | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    str,
    str | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if adata is None:
        adata = load_spatial_data(
            cfg.data.path,
            target_obsm_key=cfg.data.target_obsm_key or cfg.data.obsm_key,
            init_obsm_key=cfg.data.init_obsm_key,
        )
    target_obsm_key = cfg.data.target_obsm_key or cfg.data.obsm_key
    initial_obsm_key = cfg.data.init_obsm_key
    initial_heading_obsm_key = cfg.data.init_heading_obsm_key
    raw_targets, raw_initial_positions, color_values, raw_initial_headings = (
        load_adata_cell_positions_and_colors(
            adata,
            obsm_key=cfg.data.obsm_key,
            target_obsm_key=target_obsm_key,
            init_obsm_key=initial_obsm_key,
            init_heading_obsm_key=initial_heading_obsm_key,
            color_key=cfg.data.color_key,
        )
    )
    loaded_id_features = load_adata_id_features(adata, id_features=cfg.data.id_features)
    warn_if_targets_too_close(
        raw_targets,
        coordinate_scale=cfg.data.coordinate_scale,
        agent_radius=cfg.env.agent_radius,
        collision_iters=cfg.env.collision_iters,
    )
    targets, cfg.env.width, cfg.env.height = resolve_env_geometry(
        raw_targets,
        coordinate_scale=cfg.data.coordinate_scale,
        width=cfg.env.width,
        height=cfg.env.height,
        depth=cfg.env.depth,
        size=cfg.env.size,
        agent_radius=cfg.env.agent_radius,
        circular=cfg.env.circular,
        auto_size_buffer=cfg.env.auto_size_buffer,
    )
    initial_positions = None
    if raw_initial_positions is not None:
        if raw_initial_positions.shape != raw_targets.shape:
            raise ValueError(
                f"obsm['{initial_obsm_key}'] must match the resolved target positions shape"
            )
        initial_positions, _, _ = resolve_env_geometry(
            raw_initial_positions,
            coordinate_scale=cfg.data.coordinate_scale,
            width=cfg.env.width,
            height=cfg.env.height,
            depth=cfg.env.depth,
            circular=cfg.env.circular,
            auto_size_buffer=cfg.env.auto_size_buffer,
        )
    return (
        targets,
        initial_positions,
        target_obsm_key,
        initial_obsm_key if initial_positions is not None else None,
        color_values,
        loaded_id_features,
        raw_initial_headings,
    )


def build_eval_env_kwargs(
    cfg: Config,
    targets: np.ndarray,
    *,
    initial_positions: np.ndarray | None = None,
    initial_headings: np.ndarray | None = None,
    reuse_initial_positions: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    target_neighbors, target_neighbor_distances, _ = build_target_neighbors(
        targets,
        k=cfg.env.k,
        neighbor_graph=cfg.env.neighbor_graph,
        perception_radius=cfg.env.perception_radius,
    )
    reward_mode = kwargs.get("reward_mode", cfg.env.reward_mode)
    target_neighbor_pair_distances = (
        build_target_neighbor_pair_distances(targets, target_neighbors)
        if reward_mode == "interval_split_matched"
        else None
    )
    env_config_fields = EnvConfig.model_fields
    env_overrides = {k: v for k, v in kwargs.items() if k in env_config_fields}
    perception_radius = resolve_perception_radius(
        configured_perception_radius=cfg.env.perception_radius,
        target_neighbor_distances=target_neighbor_distances,
        target_neighbors=target_neighbors,
    )
    env_cfg = EnvConfig.model_validate(
        {
            **cfg.env.model_dump(),
            "num_envs": 1,
            "perception_radius": perception_radius,
            "reuse_initial_positions": reuse_initial_positions,
            "log_interval": 1,
            **env_overrides,
        }
    )
    ctor_params = inspect.signature(MultiCellEnv.__init__).parameters
    env_kwargs = {
        "cfg": env_cfg,
        "num_agents": int(targets.shape[0]),
        "coordinate_scale": kwargs.get("coordinate_scale", cfg.data.coordinate_scale),
        "spatial_dim": targets.shape[1],
        "target_neighbors": target_neighbors,
        "target_neighbor_distances": target_neighbor_distances,
        "target_neighbor_pair_distances": target_neighbor_pair_distances,
        "target_positions": targets,
        "initial_positions": initial_positions,
        "initial_headings": initial_headings,
        "comm_select_action": cfg.model.comm_select_action,
    }
    env_kwargs.update(
        {k: v for k, v in kwargs.items() if k in ctor_params and k not in env_kwargs}
    )
    env_kwargs["reset_mode"] = 0
    return env_kwargs


def build_eval_env_and_policy(
    cfg: Config,
    targets: np.ndarray,
    *,
    initial_positions: np.ndarray | None = None,
    initial_headings: np.ndarray | None = None,
    reuse_initial_positions: bool = False,
    loaded_id_features: np.ndarray | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> tuple[MultiCellEnv, Policy, int, dict[str, Any]]:
    if loaded_id_features is None and cfg.data.id_features is not None:
        adata = load_spatial_data(
            cfg.data.path, target_obsm_key=cfg.data.target_obsm_key or cfg.data.obsm_key
        )
        loaded_id_features = load_adata_id_features(
            adata, id_features=cfg.data.id_features
        )
    id_features = resolve_id_features(
        loaded_id_features=loaded_id_features,
        targets=targets,
        learned_id_dim=cfg.model.learned_id_dim,
    )
    learned_id_dim = cfg.model.learned_id_dim
    if cfg.env.reward_mode == "interval_split_matched":
        id_features = np.ones((len(targets), 1), dtype=np.float32)
        learned_id_dim = 0
    env_kwargs = build_eval_env_kwargs(
        cfg,
        targets,
        initial_positions=initial_positions,
        initial_headings=initial_headings,
        reuse_initial_positions=reuse_initial_positions,
        **kwargs,
    )
    env = MultiCellEnv(**env_kwargs)
    policy = Policy(
        env,
        n_neighbors=cfg.env.k,
        id_dim=int(id_features.shape[-1]),
        id_vectors=id_features,
        id_feature_encoder=cfg.model.id_feature_encoder,
        learned_id_dim=learned_id_dim,
        learned_id_init=cfg.model.learned_id_init,
        mask_nontarget_id_features=cfg.model.mask_nontarget_id_features,
        null_msg_learnable=cfg.model.null_msg_learnable,
        hidden_size=cfg.model.hidden_size,
        encoder_layers=cfg.model.encoder_layers,
        message_agg=cfg.model.message_agg,
        cell_encoder=cfg.model.cell_encoder,
        comm_enabled=cfg.model.comm_enabled,
        comm_msg_dim=cfg.model.comm_msg_dim,
        action_stay_bias=cfg.model.action_stay_bias,
    ).to(cfg.train.device)
    resolved_seed = int(cfg.train.seed if seed is None else seed)
    return env, policy, resolved_seed, env_kwargs
