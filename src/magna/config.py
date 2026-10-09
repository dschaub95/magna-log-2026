import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
import yaml
from pydantic import BaseModel, Field, model_validator

from magna.data.constants import DEFAULT_COORDINATE_SCALE

_RESET_MIX_RATIO_EPS = 1e-6


def resolve_reset_mix_counts(
    reset_init: float,
    reset_rotated_target: float,
    reset_interpolated: float,
    num_envs: int,
) -> tuple[int, int, int]:
    """Resolve bucket weights to integer env counts (ratios summing to 1, or absolute)."""
    weights = (
        float(reset_init),
        float(reset_rotated_target),
        float(reset_interpolated),
    )
    total = sum(weights)
    if abs(total - 1.0) <= _RESET_MIX_RATIO_EPS:
        raw = [w * num_envs for w in weights]
        counts = [int(x) for x in raw]
        rem = num_envs - sum(counts)
        order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:rem]:
            counts[i] += 1
    elif abs(total - num_envs) <= _RESET_MIX_RATIO_EPS:
        counts = [round(w) for w in weights]
        if any(
            abs(w - count) > _RESET_MIX_RATIO_EPS
            for w, count in zip(weights, counts, strict=True)
        ):
            raise ValueError("absolute reset mix weights must be integers")
    else:
        raise ValueError(
            "reset_init + reset_rotated_target + reset_interpolated must sum to 1 "
            "(ratios) or vector.num_envs (absolute counts)"
        )
    if counts[0] < 1:
        raise ValueError("resolved reset_init count must be >= 1")
    return counts[0], counts[1], counts[2]


class DataConfig(BaseModel):
    path: str = "data/pattern_formation/Grid2d/target_coord.pt"
    obsm_key: str = "spatial"  # Base AnnData obsm key used for coordinates.
    target_obsm_key: str | None = None  # Optional explicit target-coordinate obsm key.
    init_obsm_key: str | None = None  # Optional explicit initial-position obsm key.
    init_heading_obsm_key: str | None = (
        None  # Optional per-cell initial heading obsm key (radians).
    )
    id_features: str | None = None  # Optional static ID feature source: X or obsm key.
    color_key: str | None = None  # Optional obs key used to derive render colors.
    coordinate_scale: float = Field(
        default=DEFAULT_COORDINATE_SCALE, gt=0
    )  # Env units per spatial pixel coordinate.

    @model_validator(mode="after")
    def validate_path_fields(self):
        if self.path:
            suffix = Path(self.path).suffix.lower()
            if suffix not in {".h5ad", ".pt"}:
                raise ValueError("data.path must point to a .h5ad or .pt file")
        return self


class VectorEnvConfig(BaseModel):
    num_envs: int = Field(default=64, gt=0)  # Number of parallel env instances.
    num_workers: int = Field(default=16, ge=1)  # Worker processes for vector env.
    batch_size: int | Literal["auto"] = "auto"  # Env stepping batch size.
    zero_copy: bool = True  # Use shared-memory style transfers when possible.
    seed: int = 42  # Base RNG seed for vectorized execution.
    backend: Literal["multiprocessing"] = "multiprocessing"  # Vector backend type.


class EnvConfig(BaseModel):
    num_envs: int = Field(default=1, gt=0)  # Internal env copies inside one env.
    size: int | Literal["auto"] = 768  # Default square arena size in world units.
    auto_size_buffer: float = Field(
        default=3.0, gt=0
    )  # Padding on each side when size is auto.
    bounded: bool = True  # Clip agents to the arena boundary each step.
    circular: bool = True  # Use an inscribed circular arena instead of the full square.
    width: int | None = Field(
        default=None, gt=0
    )  # Optional explicit arena width override.
    height: int | None = Field(
        default=None, gt=0
    )  # Optional explicit arena height override.
    depth: int | None = Field(default=None, gt=0)  # 3D arena depth; defaults to width.
    agent_radius: int = Field(default=1, gt=0)  # Agent collision radius.
    render_agent_radius: float | None = Field(
        default=4, gt=0
    )  # Optional render-only cell radius override.
    draw_obs_graph: bool = False  # Draw directed observation edges in render/capture.
    draw_perception_radius: bool = (
        False  # Draw each agent's perception radius in render/capture.
    )
    collision_iters: int = Field(default=0, ge=0)  # Solver passes per step.
    collision_grid_size: int | None = Field(
        default=None, gt=0
    )  # Spatial grid cell size override.
    k: int = Field(default=8, gt=0)  # Number of neighbors used in signals.
    perception_radius: float = Field(default=136.0, gt=0)  # Sensing distance cap.
    neighbor_graph: Literal["radius_topk", "delaunay_topk"] = (
        "radius_topk"  # Neighbor selection within perception_radius.
    )
    mutual_obs_bit: bool = (
        True  # Add per-neighbor bit for reciprocal top-k observation.
    )
    target_neighbor_bit: bool = (
        False  # Add per-neighbor bit for target-graph correspondence.
    )
    perceived_ratio_obs: bool = False  # Include self perceived-neighbor ratio channel.
    density_grad_obs: bool = True  # Include the NPA-style Spiky density gradient.
    density_grad_log_norm: bool = (
        False  # Log-normalize density grad obs: log(1 + ||g||) * g / ||g||.
    )
    local_frame_obs: bool = False  # Express directional observations in the ego frame.
    self_pos_delta_obs: bool = True  # Include displacement since the prior observation.
    boundary_dist_obs: bool = (
        False  # Include self distance-to-boundary / perception_radius (capped at 1).
    )
    boundary_penalty: float = Field(
        default=0.0, le=0.0
    )  # Reward added to each agent that hits the arena boundary.
    enforce_connected: bool = (
        False  # Require agents to remain one perception-connected component.
    )
    disconnect_penalty: float = 0.0  # Reward added to each agent after rollback.
    terminate_on_disconnect: bool = False  # End episode after a disconnected rollback.
    lam_nbr_reward: float = Field(default=0.7, ge=0.0, le=1.0)  # Reward blend weight.
    reward_scaler: float = Field(default=1.0, gt=0.0)  # Final reward multiplier.
    reward_min_distance: float = Field(
        default=2.0, ge=0.0
    )  # Lower bound for reward distance-error normalization.
    reward_mode: Literal[
        "interval_v1",
        "interval_split",
        "interval_split_matched",
        "interval_v1_cached",
        "interval_split_cached",
    ] = "interval_v1"  # Reward implementation.
    reward_interval_multiplier: float = Field(
        default=2.0, gt=0
    )  # Interval size factor.
    match_roles_once: bool = False  # Freeze matched reward role assignment per episode.

    max_speed: float | None = Field(default=None, gt=0)  # Max agent velocity.
    time_horizon: int = 1024  # Episode length; use -1 for infinite horizon.
    reset_distribution: Literal["sunflower", "gaussian"] = "sunflower"
    fit_init_to_target_bbox: bool = (
        False  # Uniformly scale packed init to fit within the target bounding box.
    )
    reset_init: float = Field(
        default=1.0, ge=0.0
    )  # Standard packed-init env share/count.
    reset_rotated_target: float = Field(
        default=0.0, ge=0.0
    )  # Rotated-target init env share/count.
    reset_interpolated: float = Field(
        default=0.0, ge=0.0
    )  # Interpolated init env share/count.
    reuse_initial_positions: bool = False  # Reuse stored start positions on reset.
    recenter_on_step: bool = True  # Translate all agents each step so the swarm centroid matches arena center.
    log_interval: int = Field(default=128, gt=0)  # Env-side metric logging period.
    verbose: int = 0  # C binding log verbosity level.

    @model_validator(mode="after")
    def validate_time_horizon(self):
        if self.time_horizon == 0 or self.time_horizon < -1:
            raise ValueError("time_horizon must be > 0 or -1 for infinite horizon")
        return self

    @model_validator(mode="after")
    def validate_compatible_flags(self):
        if self.density_grad_log_norm and not self.density_grad_obs:
            raise ValueError("density_grad_log_norm requires density_grad_obs")
        if self.target_neighbor_bit and self.reward_mode == "interval_split_matched":
            raise ValueError(
                "target_neighbor_bit is incompatible with interval_split_matched"
            )
        return self

    @model_validator(mode="after")
    def validate_bounded_flags(self):
        if self.bounded:
            return self
        if self.boundary_dist_obs:
            raise ValueError("boundary_dist_obs requires bounded=True")
        if self.boundary_penalty != 0.0:
            raise ValueError("boundary_penalty requires bounded=True")
        if self.collision_iters > 0:
            raise ValueError("collision_iters > 0 requires bounded=True")
        return self

    @model_validator(mode="after")
    def resolve_square_dimensions(self):
        if self.size != "auto":
            self.width = self.width or self.size
            self.height = self.height or self.size
        if self.width is None and self.height is None:
            return self
        if self.width is None or self.height is None:
            raise ValueError(
                "resolved env.width and env.height must both be set for square arenas"
            )
        if self.width != self.height:
            raise ValueError(
                "resolved env.width and env.height must match for square arenas"
            )
        self.depth = self.depth or self.width
        return self


class ModelConfig(BaseModel):
    hidden_size: int = 128  # Shared MLP trunk width.
    encoder_layers: int = Field(default=1, gt=0)  # Shared encoder MLP depth.
    id_feature_encoder: bool = False  # Apply a trainable encoder to static ID features.
    learned_id_dim: int = Field(
        default=0, ge=0
    )  # Trainable per-cell ID embedding width.
    learned_id_init: Literal["zeros", "gaussian", "orthogonal"] = "gaussian"
    mask_nontarget_id_features: bool = False  # ID features only for target neighbors.
    null_msg_learnable: bool = False  # Learn missing-neighbor message values.
    message_agg: Literal["mean", "sum", "max", "concat", "attn"] = (
        "concat"  # Set aggregation mode.
    )
    cell_encoder: Literal["mlp", "linear", "raw"] = (
        "raw"  # Cell feature transform before messages.
    )
    comm_enabled: bool = True  # Enable agent-to-agent learned communication.
    comm_msg_dim: int = Field(default=32, gt=0)  # Recurrent communication bottleneck.
    comm_select_action: bool = False  # Add binary action heads for message receivers.
    action_stay_bias: float = 0.0  # Initial logit bias toward no-turn and stop actions.

    @model_validator(mode="after")
    def validate_model_options(self):
        if self.comm_select_action and not self.comm_enabled:
            raise ValueError("comm_select_action requires comm_enabled")
        return self


class TrainConfig(BaseModel):
    seed: int = 42  # Global training seed.
    torch_deterministic: bool = True  # Request deterministic torch kernels.
    cpu_offload: bool = False  # Offload select training ops/states to CPU.
    device: str = "cuda"  # Device used for policy and tensors.
    optimizer: Literal["muon", "adam"] = "muon"  # Optimizer choice.
    precision: Literal["float32", "float16", "bfloat16"] = "float32"  # Compute dtype.
    total_timesteps: int = 200_000_000  # Overall training budget.
    learning_rate: float = 0.015  # Initial optimizer LR.
    embedding_learning_rate: float = 5e-3  # AdamW LR for learned IDs under Muon.
    embedding_weight_decay: float = 0.0  # AdamW decay for learned IDs under Muon.
    embedding_pretrain_steps: int = Field(default=10_000, ge=0)
    embedding_pretrain_progbar: bool = False
    embedding_pretrain_learning_rate: float = Field(default=0.01, gt=0)
    embedding_pretrain_min_learning_rate: float = Field(default=1e-4, ge=0)
    embedding_pretrain_weight_decay: float = Field(default=1e-4, ge=0)
    embedding_pretrain_margin: float = Field(default=0.5, ge=0)
    embedding_pretrain_edge_coef: float = Field(default=10_000.0, ge=0)
    embedding_pretrain_order_coef: float = Field(default=1.0, ge=0)
    anneal_lr: bool = True  # Cosine-anneal LR over training.
    min_lr_ratio: float = 0.0  # Minimum LR as fraction of initial LR.
    lr_cosine_power: float = Field(default=1.0, gt=0)  # >1 stretches high-LR phase.
    gamma: float = 0.99  # Discount factor.
    gae_lambda: float = 0.90  # GAE trace decay.
    update_epochs: int = 1  # controls num iterations over same batch during training
    clip_coef: float = 0.2  # PPO policy clipping range.
    vf_coef: float = 2.0  # Value loss weight.
    vf_clip_coef: float = 0.2  # Value target clipping range.
    max_grad_norm: float = 1.5  # Global gradient norm clip.
    ent_coef: float = 0.02  # Entropy bonus weight.
    adam_beta1: float = 0.95  # Adam/Muon beta1.
    adam_beta2: float = 0.999  # Adam/Muon beta2.
    adam_eps: float = 1e-12  # Adam/Muon epsilon.

    data_dir: str = "outputs/runs"  # Run output directory (checkpoints).
    checkpoint_interval: int = 200  # Save cadence in trainer updates.
    batch_size: int | Literal["auto"] = "auto"  # Rollout batch size for learner.
    minibatch_size: int | Literal["auto"] = "auto"  # SGD minibatch size.
    max_minibatch_size: int | Literal["match_minibatch"] = "match_minibatch"  # Cap.
    auto_minibatch_divisor: int = 4  # Divisor in auto minibatch formula.
    bptt_horizon: int = 64  # Sequence length for recurrent unroll.
    compile: bool = False  # Enable torch.compile.
    compile_mode: str = "max-autotune-no-cudagraphs"  # torch.compile mode.
    compile_fullgraph: bool = True  # Require full graph compilation.

    vtrace_rho_clip: float = 1.0  # V-trace rho clipping threshold.
    vtrace_c_clip: float = 1.0  # V-trace c clipping threshold.

    prio_alpha: float = 0.8  # Priority replay exponent.
    prio_beta0: float = 0.2  # Initial IS correction exponent.

    # n_epochs: int = 100_000_000
    use_rnn: bool = True  # Enable recurrent policy path.
    replay_segment_state: bool = False  # Train from rollout segment initial state.
    env: str = "MultiCellEnv"  # Trainer-side env name metadata.
    n_repeat: int = Field(default=1, ge=1)  # Number of full training repeats.
    print_dashboard: bool = True  # Print the PuffeRL dashboard after each repeat.
    print_config: bool = True  # Print the config before training.
    generate_rollout_video: bool = True  # Generate rollout video after training.
    track_error_init: bool = False  # Log initial distance error after each reset.

    @model_validator(mode="after")
    def validate_auto_minibatch(self):
        if self.auto_minibatch_divisor <= 0:
            raise ValueError("auto_minibatch_divisor must be > 0")
        return self


class Config(BaseModel):
    project: str = "magna"  # Weights & Biases project name.
    run_name: str | None = None  # Optional explicit run name.
    group: str | None = None  # Optional W&B group for related runs.
    notes: str | None = None  # Optional run notes text.
    wandb_mode: Literal["online", "offline", "disabled"] = "online"  # W&B mode.
    wandb_dir: str = "outputs"  # W&B local output directory.
    save_model: bool = False  # Save model to W&B.

    data: DataConfig = Field(default_factory=DataConfig)  # AnnData target settings.
    env: EnvConfig = Field(default_factory=EnvConfig)  # Environment settings.
    vector: VectorEnvConfig = Field(
        default_factory=VectorEnvConfig
    )  # Vector env settings.
    model: ModelConfig = Field(default_factory=ModelConfig)  # Policy model settings.
    train: TrainConfig = Field(default_factory=TrainConfig)  # Training hyperparams.

    @model_validator(mode="after")
    def validate_comm_requires_rnn(self):
        if self.model.comm_enabled and not self.train.use_rnn:
            raise ValueError("comm_enabled requires use_rnn")
        if self.train.embedding_pretrain_steps and (
            self.model.learned_id_dim and self.model.learned_id_init == "zeros"
        ):
            raise ValueError("embedding pretraining requires nonzero initialization")
        if self.model.mask_nontarget_id_features and not self.env.target_neighbor_bit:
            raise ValueError("mask_nontarget_id_features requires target_neighbor_bit")
        return self

    @model_validator(mode="after")
    def validate_reset_mix(self):
        self.reset_mix_counts()
        return self

    def reset_mix_counts(self) -> tuple[int, int, int]:
        return resolve_reset_mix_counts(
            self.env.reset_init,
            self.env.reset_rotated_target,
            self.env.reset_interpolated,
            self.vector.num_envs,
        )

    def resolve_dependent_defaults(self):
        if self.env.max_speed is None:
            self.env.max_speed = float(self.env.agent_radius)
        if self.env.collision_grid_size is None:
            self.env.collision_grid_size = self.env.agent_radius * 2
        if not self.data.path:
            return self

        if Path(self.data.path).suffix.lower() == ".pt":
            from magna.data.io import load_pt_coordinates

            num_agents = len(load_pt_coordinates(self.data.path, mmap=True))
        else:
            import anndata as ad

            num_agents = int(ad.read_h5ad(self.data.path, backed="r").n_obs)

        minibatch_auto = self.train.minibatch_size == "auto"
        max_matches_minibatch = self.train.max_minibatch_size == "match_minibatch" or (
            minibatch_auto
        )

        if minibatch_auto:
            h = self.train.bptt_horizon
            raw = max(
                1,
                self.vector.num_envs
                * num_agents
                * h
                // self.train.auto_minibatch_divisor,
            )
            self.train.minibatch_size = max(h, raw // h * h)

        if max_matches_minibatch:
            self.train.max_minibatch_size = self.train.minibatch_size

        if self.model.comm_enabled:
            self._align_train_minibatch_for_comm(
                num_agents, minibatch_auto, max_matches_minibatch
            )

        h = self.train.bptt_horizon
        mb = min(self.train.minibatch_size, self.train.max_minibatch_size)
        if isinstance(mb, int) and mb % h:
            raise ValueError(
                f"minibatch_size {mb} must be divisible by bptt_horizon {h}"
            )

        if not torch.cuda.is_available() and self.train.device != "cpu":
            self.train.device = "cpu"
            print("Warning: No CUDA device available. Using CPU.")

        return self

    def _align_train_minibatch_for_comm(
        self, num_agents: int, minibatch_auto: bool, max_matches_minibatch: bool
    ) -> None:
        """Minibatch segments must divide local_num_agents (PuffeRL._validate_comm_layout)."""
        h = self.train.bptt_horizon
        mb = self.train.minibatch_size
        max_mb = self.train.max_minibatch_size
        if not isinstance(mb, int) or not isinstance(max_mb, int):
            return

        batch_size = (
            self.train.batch_size
            if isinstance(self.train.batch_size, int)
            else self.vector.num_envs * self.env.num_envs * num_agents * h
        )
        group = num_agents * h
        cap = batch_size if max_matches_minibatch else min(max_mb, batch_size)
        if cap < group:
            raise ValueError("comm_enabled minibatch is too small")

        eff = max(group, min(mb, cap) // group * group)
        if max_matches_minibatch:
            mb = max_mb = eff
        elif minibatch_auto or mb <= max_mb:
            mb = eff
        else:
            max_mb = eff
            mb = max(eff, min(mb, batch_size) // eff * eff)
        self.train.minibatch_size = mb
        self.train.max_minibatch_size = max_mb


def _ensure_external_env_defaults(
    config_dict: dict[str, Any] | None, *, flat: bool
) -> dict[str, Any]:
    if config_dict is None:
        return {}

    normalized = dict(config_dict)
    if flat:
        if "env.circular" not in normalized:
            normalized["env.circular"] = False
        return normalized

    env_cfg = normalized.get("env")
    if not (isinstance(env_cfg, dict) and "circular" in env_cfg):
        env_cfg = dict(env_cfg) if isinstance(env_cfg, dict) else {}
        env_cfg["circular"] = False
        normalized["env"] = env_cfg
    return normalized


def _parse_value(raw: str) -> Any:
    # JSON parses: 1e-5, 10, true/false/null, [..], {..}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _next_cli_token_is_value(token: str | None) -> bool:
    """True if token should pair with a preceding ``--key`` (space form)."""
    if token is None:
        return False
    if not token.startswith("-"):
        return True
    if token.startswith("--"):
        return False
    # Negative numbers: -1, -.5, -1e-5 (not a short-flag stack like -abc)
    return len(token) > 1 and (token[1].isdigit() or token[1] == ".")


def parse_dot_overrides(argv: Iterable[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse ``--key=value`` and ``--key value`` into a flat override dict.

    Space-separated values are accepted when the next token is unambiguously a
    value (paths, strings, positive numbers, or negative numbers like ``-1``).
    """
    overrides: dict[str, Any] = {}
    rest: list[str] = []
    _dot_kv = re.compile(r"^--([A-Za-z0-9_.-]+)=(.*)$")
    _dot_key = re.compile(r"^--([A-Za-z0-9_.-]+)$")
    argv_list = list(argv)
    i = 0
    while i < len(argv_list):
        a = argv_list[i]
        m = _dot_kv.match(a)
        if m:
            k, v = m.group(1), _parse_value(m.group(2))
            overrides[k] = v
            i += 1
            continue
        m_key = _dot_key.match(a)
        if (
            m_key
            and i + 1 < len(argv_list)
            and _next_cli_token_is_value(argv_list[i + 1])
        ):
            k = m_key.group(1)
            v = _parse_value(argv_list[i + 1])
            overrides[k] = v
            i += 2
            continue
        rest.append(a)
        i += 1
    return overrides, rest


def unflatten(flat: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in flat.items():
        cur = out
        parts = k.split(sep)
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def flatten(nested: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Convert a nested dictionary to a flat dictionary with dot-separated keys."""
    out: dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                _flatten(v, new_key)
        else:
            out[prefix] = obj

    _flatten(nested)
    return out


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k not in out:
            print(f"⚠️  Warning: Unknown config key '{k}' ignored")
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def wandb_config_as_dict(conf: Any) -> dict[str, Any]:
    d = conf.as_dict() if hasattr(conf, "as_dict") else dict(conf)
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _clear_stale_env_dimensions(
    merged: dict[str, Any],
    flat_overrides: dict[str, Any],
    override_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    env_cfg = merged.get("env")
    if not isinstance(env_cfg, dict):
        return merged

    updated = dict(merged)
    updated_env = dict(env_cfg)
    keys = set(flat_overrides if override_keys is None else override_keys)
    if "env.size" in keys and "env.width" not in keys and "env.height" not in keys:
        updated_env["width"] = None
        updated_env["height"] = None
    if not {"env.size", "env.width", "env.height"}.isdisjoint(keys) and (
        "env.depth" not in keys
    ):
        updated_env["depth"] = None
    updated["env"] = updated_env
    return updated


def apply_cli_overrides(
    cfg: Config, flat: dict[str, Any], *, resolve: bool = True
) -> Config:
    base = cfg.model_dump(mode="json")
    merged = deep_merge(base, unflatten(flat))
    merged = _clear_stale_env_dimensions(merged, flat)
    cfg = Config.model_validate(merged)
    return cfg.resolve_dependent_defaults() if resolve else cfg


def finalize_with_wandb_overrides(
    cfg_base: Config,
    run: Any,
    *,
    override_keys: Iterable[str] | None = None,
    resolve: bool = True,
) -> Config:
    base = cfg_base.model_dump(mode="json")
    flat = _ensure_external_env_defaults(wandb_config_as_dict(run.config), flat=True)
    merged = deep_merge(base, unflatten(flat))
    merged = _clear_stale_env_dimensions(merged, flat, override_keys)
    cfg = Config.model_validate(merged)
    return cfg.resolve_dependent_defaults() if resolve else cfg


def init_config_with_wandb_init(cfg: Config) -> tuple[Config, wandb.Run]:
    is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    if not is_sweep:
        cfg = cfg.resolve_dependent_defaults()

    # Flatten config for wandb
    nested_config = cfg.model_dump(mode="json")
    flat_config = flatten(nested_config)

    run = wandb.init(
        project=cfg.project if not is_sweep else None,
        name=cfg.run_name,
        group=cfg.group,
        notes=cfg.notes,
        mode=cfg.wandb_mode,
        dir=cfg.wandb_dir,
        config=None if is_sweep else flat_config,
        settings=wandb.Settings(console="off"),
    )
    sweep_locked_keys = frozenset(dict(run.config).keys()) if is_sweep else frozenset()

    if is_sweep:
        # Backfill missing keys without overriding sweep-locked values.
        run.config.update(
            {k: v for k, v in flat_config.items() if k not in run.config},
            allow_val_change=False,
        )

    cfg = finalize_with_wandb_overrides(
        cfg, run, override_keys=sweep_locked_keys if is_sweep else None
    )

    if is_sweep:
        # Resolved defaults for logging; skip keys wandb sweep owns (avoids locked-update warnings).
        resolved_flat = flatten(cfg.model_dump(mode="json"))
        run.config.update(
            {k: v for k, v in resolved_flat.items() if k not in sweep_locked_keys},
            allow_val_change=True,
        )
    return cfg, run


def load_wandb_config(run_path: str, *, resolve: bool = True) -> Config:
    api = wandb.Api()  # WANDB_API_KEY must be set in your env
    run = api.run(run_path)  # fetch the run object
    return finalize_with_wandb_overrides(Config(), run, resolve=resolve)


def load_yaml_config(path: str, *, resolve: bool = True) -> Config:
    with open(path) as f:
        config_dict = _ensure_external_env_defaults(yaml.safe_load(f), flat=False)
    base = Config().model_dump(mode="json")
    merged = deep_merge(base, config_dict)
    merged = _clear_stale_env_dimensions(merged, flatten(config_dict))
    cfg = Config.model_validate(merged)
    return cfg.resolve_dependent_defaults() if resolve else cfg
