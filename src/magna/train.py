import math
import pprint
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pufferlib
import torch
import torch.nn.functional as F
import wandb
from pufferlib.pufferl import PuffeRL as PufferTrainer
from pufferlib.pufferl import compute_puff_advantage
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm.auto import tqdm

from magna.config import (
    Config,
    EnvConfig,
    deep_merge,
    flatten,
    init_config_with_wandb_init,
    unflatten,
)
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
from magna.utils import (
    configure_torch_determinism,
    resolve_cell_colors,
    seed_everything,
)


def _cosine_power_scheduler(optimizer, T_max, *, min_lr_ratio=0.0, power=1.0):
    def lr_lambda(step):
        t = (min(step, T_max) / T_max) ** power if T_max > 0 else 1.0
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _emit_wandb_run_metadata(run: wandb.Run) -> None:
    print(f"WANDB_RUN_ID={getattr(run, 'id', None)}")
    print(f"WANDB_RUN_NAME={getattr(run, 'name', None)}")
    print(f"WANDB_RUN_DIR={getattr(run, 'dir', None)}")


class WandbRunLogger:
    def __init__(
        self,
        wandb_run: wandb.Run,
        save_model: bool = False,
        *,
        step_offset: int = 0,
        run_id: str | None = None,
    ):
        self.wandb_run = wandb_run
        self.run_id = run_id or wandb_run.id
        self.save_model = save_model
        self.step_offset = step_offset

    def log(self, logs: dict, step: int):
        self.wandb_run.log(logs, step=step + self.step_offset)

    def close(self, model_path: str | None = None):
        if self.save_model and model_path is not None:
            artifact = wandb.Artifact(self.run_id, type="model")
            artifact.add_file(model_path)
            self.wandb_run.log_artifact(artifact)


class _EmbeddingLRLogger:
    def __init__(self, logger, optimizer):
        self.logger = logger
        self.optimizer = optimizer

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def log(self, logs, step):
        logs["embedding_learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.logger.log(logs, step)


class PuffeRL(PufferTrainer):
    def __init__(self, config, vecenv, policy, logger=None):
        self.lstm_h0 = self.lstm_c0 = None
        super().__init__(config, vecenv, policy, logger=logger)
        self.scheduler = _cosine_power_scheduler(
            self.optimizer,
            self.total_epochs,
            min_lr_ratio=config["min_lr_ratio"],
            power=config["lr_cosine_power"],
        )
        if (
            config["embedding_pretrain_steps"]
            and self.uncompiled_policy.policy.learned_id is not None
        ):
            self._pretrain_embeddings()
        self._init_embedding_optimizer()
        if self.embedding_optimizer is not None:
            self.logger = _EmbeddingLRLogger(self.logger, self.embedding_optimizer)
        if self._comm_enabled:
            self._validate_comm_layout()
        if self._replay_segment_state_enabled:
            self._init_recurrent_buffers()

    def _init_embedding_auxiliary(self):
        neighbors = self.vecenv.driver_env._target_neighbors[0]
        distances = self.vecenv.driver_env._target_neighbor_distances[0]
        rows = np.broadcast_to(np.arange(len(neighbors))[:, None], neighbors.shape)
        valid = neighbors >= 0
        positive = np.column_stack((rows[valid], neighbors[valid]))
        order = np.argsort(distances)
        ordered = np.take_along_axis(neighbors, order, axis=1)
        ordered_distances = np.take_along_axis(distances, order, axis=1)
        ordered_valid = (
            (ordered[:, :-1] >= 0)
            & (ordered[:, 1:] >= 0)
            & ~np.isclose(ordered_distances[:, :-1], ordered_distances[:, 1:])
        )
        ordered_pairs = np.column_stack(
            (
                rows[:, :-1][ordered_valid],
                ordered[:, :-1][ordered_valid],
                ordered[:, 1:][ordered_valid],
            )
        )
        adjacency = np.eye(len(neighbors), dtype=bool)
        adjacency[positive[:, 0], positive[:, 1]] = True
        if not len(positive) or adjacency.all(axis=1).any():
            raise ValueError(
                "Embedding auxiliary loss requires positive and negative pairs"
            )
        weight = self.uncompiled_policy.policy.learned_id.weight
        self.embedding_aux_pairs = (
            torch.as_tensor(positive, device=weight.device),
            torch.as_tensor(adjacency, device=weight.device),
            torch.as_tensor(ordered_pairs, device=weight.device),
            torch.as_tensor(distances[valid], device=weight.device, dtype=weight.dtype),
        )

    def _embedding_aux_loss(self):
        positive, adjacency, ordered, target_distance = self.embedding_aux_pairs
        negative = torch.randint(
            len(adjacency), (len(positive),), device=positive.device
        )
        invalid = adjacency[positive[:, 0], negative]
        while invalid.any():
            negative[invalid] = torch.randint(
                len(adjacency), (int(invalid.sum()),), device=positive.device
            )
            invalid = adjacency[positive[:, 0], negative]
        embedding = self.uncompiled_policy.policy.learned_id.weight[1:]

        positive_distance = F.pairwise_distance(*embedding[positive].unbind(1))
        negative_distance = F.pairwise_distance(
            embedding[positive[:, 0]], embedding[negative]
        ).square()
        order_violation = F.relu(
            (embedding[ordered[:, 0]] - embedding[ordered[:, 1]]).square().sum(-1)
            - (embedding[ordered[:, 0]] - embedding[ordered[:, 2]]).square().sum(-1)
        )

        def active_mean(violation):
            return violation.sum() / violation.gt(0).sum().clamp_min(1)

        eps = torch.finfo(positive_distance.dtype).eps
        edge_loss = F.mse_loss(
            positive_distance / positive_distance.mean().clamp_min(eps),
            target_distance / target_distance.mean().clamp_min(eps),
        )
        return (
            active_mean(
                F.relu(
                    self.config["embedding_pretrain_margin"]
                    + positive_distance.square()
                    - negative_distance
                )
            )
            + self.config["embedding_pretrain_edge_coef"] * edge_loss
            + self.config["embedding_pretrain_order_coef"]
            * active_mean(order_violation)
        )

    def _pretrain_embeddings(self):
        self._init_embedding_auxiliary()
        embedding = self.uncompiled_policy.policy.learned_id.weight
        optimizer = torch.optim.AdamW(
            [embedding],
            lr=self.config["embedding_pretrain_learning_rate"],
            weight_decay=self.config["embedding_pretrain_weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["embedding_pretrain_steps"],
            eta_min=self.config["embedding_pretrain_min_learning_rate"],
        )
        for _ in tqdm(
            range(self.config["embedding_pretrain_steps"]),
            desc="Pretraining embeddings",
            disable=not self.config["embedding_pretrain_progbar"],
        ):
            optimizer.zero_grad()
            loss = self._embedding_aux_loss()
            loss.backward()
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            learned = embedding[1:]
            learned.sub_(learned.mean(0))
            learned.div_(
                learned.norm(dim=-1).max().clamp_min(torch.finfo(learned.dtype).eps)
            )
        del self.embedding_aux_pairs
        print(f"Embedding pretraining loss: {loss.item():.6f}")

    def _init_embedding_optimizer(self):
        self.embedding_optimizer = self.embedding_scheduler = None
        learned_id = getattr(self.uncompiled_policy.policy, "learned_id", None)
        if self.config["optimizer"] != "muon" or learned_id is None:
            return

        embedding = learned_id.weight
        occurrences = sum(
            param is embedding
            for group in self.optimizer.param_groups
            for param in group["params"]
        )
        if occurrences != 1:
            raise ValueError("Learned ID embedding must occur once in Muon parameters")
        for group in self.optimizer.param_groups:
            params = group["params"]
            group["params"] = [param for param in params if param is not embedding]

        self.embedding_optimizer = torch.optim.AdamW(
            [embedding],
            lr=self.config["embedding_learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config["embedding_weight_decay"],
        )
        self.embedding_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.embedding_optimizer,
            T_max=self.total_epochs,
            eta_min=self.config["embedding_learning_rate"]
            * self.config["min_lr_ratio"],
        )

    def _step_optimizers(self):
        if self.embedding_optimizer is not None:
            self.embedding_optimizer.step()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.embedding_optimizer is not None:
            self.embedding_optimizer.zero_grad()

    @property
    def _comm_enabled(self) -> bool:
        return bool(
            getattr(
                getattr(self.uncompiled_policy, "policy", None), "comm_enabled", False
            )
        )

    @property
    def _replay_segment_state_enabled(self) -> bool:
        return self.config["replay_segment_state"] and self.config["use_rnn"]

    def _init_recurrent_buffers(self):
        device = self.config["device"]
        h = self.uncompiled_policy.hidden_size
        self.lstm_h0 = torch.zeros(self.segments, h, device=device)
        self.lstm_c0 = torch.zeros(self.segments, h, device=device)

    def _validate_comm_layout(self):
        group_size = self._comm_group_size()
        if self.vecenv.agents_per_batch % group_size:
            raise ValueError(
                "Communication requires vecenv.agents_per_batch to contain whole local-agent groups"
            )
        if self.segments % group_size:
            raise ValueError(
                "Communication requires PPO segments to align to whole local-agent groups"
            )
        if self.minibatch_segments % group_size:
            raise ValueError(
                "Communication requires PPO minibatches to contain whole local-agent groups"
            )

    def _snapshot_segment_init(self, state: dict):
        env_id = state.get("env_id")
        if env_id is None:
            return
        seg_step = self.ep_lengths[env_id.start].item()
        if seg_step != 0:
            return
        key = int(env_id.start)
        batch_rows = slice(
            self.ep_indices[key].item(),
            1 + self.ep_indices[env_id.stop - 1].item(),
        )
        if self.config["use_rnn"]:
            self.lstm_h0[batch_rows] = state["lstm_h"]
            self.lstm_c0[batch_rows] = state["lstm_c"]

    def _reset_done_state(self, env_id: slice, done: torch.Tensor):
        if not done.any():
            return
        key = int(env_id.start)
        if self.config["use_rnn"]:
            self.lstm_h[key][done] = 0
            self.lstm_c[key][done] = 0

    def evaluate(self):
        if not self._replay_segment_state_enabled:
            return super().evaluate()

        recv = self.vecenv.recv
        forward_eval = self.policy.forward_eval
        device = self.config["device"]

        def recv_with_done_reset():
            out = recv()
            _, _, d, t, _, env_id, _ = out
            env_id = slice(env_id[0], env_id[-1] + 1)
            done = (
                torch.as_tensor(d, device=device).bool()
                | torch.as_tensor(t, device=device).bool()
            )
            self._reset_done_state(env_id, done)
            return out

        def forward_eval_with_snapshot(observations, state):
            self._snapshot_segment_init(state)
            return forward_eval(observations, state)

        try:
            self.vecenv.recv = recv_with_done_reset
            self.policy.forward_eval = forward_eval_with_snapshot
            return super().evaluate()
        finally:
            self.vecenv.recv = recv
            self.policy.forward_eval = forward_eval

    def _comm_group_size(self) -> int:
        if not self._comm_enabled:
            return 1
        return int(self.uncompiled_policy.policy.local_num_agents)

    def _sample_minibatch_indices(
        self, prio_weights: torch.Tensor, anneal_beta: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_size = self._comm_group_size()
        if group_size == 1:
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            return idx, (self.segments * prio_probs[idx, None]) ** -anneal_beta

        group_weights = prio_weights.reshape(-1, group_size).mean(dim=1)
        group_probs = (group_weights + 1e-6) / (group_weights.sum() + 1e-6)
        group_idx = torch.multinomial(
            group_probs, self.minibatch_segments // group_size
        )
        offsets = torch.arange(group_size, device=prio_weights.device)
        idx = (group_idx[:, None] * group_size + offsets).reshape(-1)
        row_probs = group_probs[group_idx].repeat_interleave(group_size).unsqueeze(1)
        return idx, (group_probs.shape[0] * row_probs) ** -anneal_beta

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        config = self.config
        device = config["device"]

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1
        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            idx, mb_prio = self._sample_minibatch_indices(prio_weights, anneal_beta)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = {"action": mb_actions, "lstm_h": None, "lstm_c": None}
            if self.lstm_h0 is not None:
                state["lstm_h"] = self.lstm_h0[idx].unsqueeze(0)
                state["lstm_c"] = self.lstm_c0[idx].unsqueeze(0)
            logits, newvalue = self.policy(mb_obs, state)
            _, newlogprob, entropy = pufferlib.pytorch.sample_logits(
                logits, action=mb_actions
            )

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

            adv = advantages[idx]
            adv = compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_terminals,
                ratio,
                adv,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = (
                pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss
            )
            self.amp_context.__enter__()
            self.values[idx] = newvalue.detach().float()

            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += entropy_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                emb = getattr(self, "embedding_optimizer", None)
                emb = emb.param_groups[0]["params"] if emb else []
                emb_ids = {id(p) for p in emb}
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if id(p) not in emb_ids],
                    config["max_grad_norm"],
                )
                if emb:
                    torch.nn.utils.clip_grad_norm_(emb, config["max_grad_norm"])
                self._step_optimizers()

        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()
            if self.embedding_scheduler is not None:
                self.embedding_scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses["explained_variance"] = explained_var.item()

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if (
            done_training
            or self.global_step == 0
            or time.time() > self.last_log_time + 0.25
        ):
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        return logs


class QuietPuffeRL(PuffeRL):
    def print_dashboard(self, clear=False):
        return None


def _resolve_vector_backend(name: str):
    if name == "multiprocessing":
        return pufferlib.vector.Multiprocessing
    raise ValueError(f"Unsupported vector backend '{name}'")


def _evaluate_frozen_policy(
    policy, state_dict: dict, cfg: Config, env_kwargs: dict
) -> tuple[float, float, float]:
    from magna.eval import DEFAULT_INFERENCE_STEPS, run_policy_inference

    policy.eval()
    policy.load_state_dict(state_dict)
    eval_env_kwargs = {
        **env_kwargs,
        "reset_mode": 0,
        "track_error_range": True,
        "cfg": EnvConfig.model_validate(
            {
                **env_kwargs["cfg"].model_dump(),
                "num_envs": cfg.vector.num_envs,
                "log_interval": 1,
                "time_horizon": (
                    cfg.env.time_horizon
                    if cfg.env.time_horizon > 0
                    else DEFAULT_INFERENCE_STEPS
                ),
            }
        ),
    }
    metrics = run_policy_inference(
        env_kwargs=eval_env_kwargs,
        policy=policy,
        device=cfg.train.device,
        seed=cfg.train.seed,
        n_rollouts=1,
        video_path=None,
        show_progress=False,
    )
    errors = (metrics["error_mean"], metrics["error_min"], metrics["error_max"])
    if any(error is None for error in errors):
        raise RuntimeError("Frozen evaluation did not produce an error")
    return tuple(map(float, errors))


def _sync_resolved_env_geometry_to_wandb(run: wandb.Run, cfg: Config) -> None:
    run.config.update(
        flatten(
            {
                "env": {
                    "width": cfg.env.width,
                    "height": cfg.env.height,
                    "depth": cfg.env.depth,
                }
            }
        ),
        allow_val_change=True,
    )


def train(
    cfg: Config | None = None,
    *,
    max_updates: int | None = None,
    load_weights_from: str | None = None,
    **kwargs,
):
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    cfg = cfg or Config()
    if kwargs:
        # we expect kwargs to be a flat dictionary of config overrides
        base = cfg.model_dump(mode="json")
        merged = deep_merge(base, unflatten(kwargs))
        cfg = Config.model_validate(merged)
    cfg, run = init_config_with_wandb_init(cfg)
    seed_everything(int(cfg.train.seed))
    configure_torch_determinism(bool(cfg.train.torch_deterministic))
    _emit_wandb_run_metadata(run)

    adata = load_spatial_data(
        cfg.data.path, target_obsm_key=cfg.data.target_obsm_key or cfg.data.obsm_key
    )
    raw_targets, _, color_values, _ = load_adata_cell_positions_and_colors(
        adata,
        obsm_key=cfg.data.obsm_key,
        target_obsm_key=cfg.data.target_obsm_key,
        color_key=cfg.data.color_key,
    )
    warn_if_targets_too_close(
        raw_targets,
        coordinate_scale=cfg.data.coordinate_scale,
        agent_radius=cfg.env.agent_radius,
        collision_iters=cfg.env.collision_iters,
    )
    targets, resolved_width, resolved_height = resolve_env_geometry(
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
    cfg.env.width = resolved_width
    cfg.env.height = resolved_height
    cfg.env.depth = cfg.env.depth or resolved_width
    _sync_resolved_env_geometry_to_wandb(run, cfg)

    if cfg.train.print_config:
        print("==== Training Config ====")
        pprint.pprint(cfg.model_dump())
        print("=========================")

    num_agents = int(targets.shape[0])
    target_neighbors, target_neighbor_distances, _ = build_target_neighbors(
        targets,
        k=cfg.env.k,
        neighbor_graph=cfg.env.neighbor_graph,
        perception_radius=cfg.env.perception_radius,
    )
    target_neighbor_pair_distances = (
        build_target_neighbor_pair_distances(targets, target_neighbors)
        if cfg.env.reward_mode == "interval_split_matched"
        else None
    )
    loaded_id_features = load_adata_id_features(
        adata,
        id_features=cfg.data.id_features,
    )
    id_features = resolve_id_features(
        loaded_id_features=loaded_id_features,
        targets=targets,
        learned_id_dim=cfg.model.learned_id_dim,
    )
    learned_id_dim = cfg.model.learned_id_dim
    if cfg.env.reward_mode == "interval_split_matched":
        id_features = np.ones((num_agents, 1), dtype=np.float32)
        learned_id_dim = 0
    perception_radius = resolve_perception_radius(
        configured_perception_radius=cfg.env.perception_radius,
        target_neighbor_distances=target_neighbor_distances,
        target_neighbors=target_neighbors,
    )

    env_cfg = EnvConfig.model_validate(
        {
            **cfg.env.model_dump(),
            "perception_radius": perception_radius,
        }
    )
    vector_cfg = cfg.vector
    n_init, n_rot, n_interp = cfg.reset_mix_counts()

    env_kwargs = {
        "cfg": env_cfg,
        "num_agents": num_agents,
        "coordinate_scale": cfg.data.coordinate_scale,
        "spatial_dim": targets.shape[1],
        "target_neighbors": target_neighbors,
        "target_neighbor_distances": target_neighbor_distances,
        "target_neighbor_pair_distances": target_neighbor_pair_distances,
        "target_positions": targets,
        "initial_positions": None,
        "reset_mode": 0,
        "comm_select_action": cfg.model.comm_select_action,
        "track_error_init": cfg.train.track_error_init,
    }
    modes = [0] * n_init + [1] * n_rot + [2] * n_interp
    train_env_kwargs = [{**env_kwargs, "reset_mode": mode} for mode in modes]
    render_env_kwargs = {
        **env_kwargs,
        "cell_colors": resolve_cell_colors(targets, color_values),
    }

    model_path = None
    logger = None
    step_offset = 0
    errors = []
    eval_errors = []
    metrics = {}
    min_error = np.inf
    updates_remaining = max_updates
    if load_weights_from is not None:
        from magna.eval import load_policy_weights
    for i in range(cfg.train.n_repeat):
        if updates_remaining == 0:
            break
        repeat_best_error = np.inf
        repeat_best_state_dict = None
        n_envs = vector_cfg.num_envs
        vecenv = pufferlib.vector.make(
            [MultiCellEnv] * n_envs,
            env_args=[[]] * n_envs,
            env_kwargs=train_env_kwargs,
            num_envs=n_envs,
            num_workers=vector_cfg.num_workers,
            batch_size=vector_cfg.batch_size,
            zero_copy=vector_cfg.zero_copy,
            seed=vector_cfg.seed,
            backend=_resolve_vector_backend(vector_cfg.backend),
        )

        id_dim = int(id_features.shape[-1])
        policy = Policy(
            vecenv.driver_env,
            n_neighbors=cfg.env.k,
            id_dim=id_dim,
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
        if load_weights_from is not None:
            print(f"Loading weights from: {load_weights_from}")
            load_policy_weights(policy, load_weights_from, device=cfg.train.device)
            policy.train()

        print(f"Using minibatch size: {cfg.train.minibatch_size}")

        logger = WandbRunLogger(
            run,
            cfg.save_model,
            step_offset=step_offset,
            run_id=run.id if cfg.train.n_repeat == 1 else f"{run.id}_{i + 1}",
        )
        trainer_cls = PuffeRL if cfg.train.print_dashboard else QuietPuffeRL
        trainer = trainer_cls(cfg.train.model_dump(), vecenv, policy, logger=logger)

        last_error = None
        repeat_updates = min(
            trainer.total_epochs, updates_remaining or trainer.total_epochs
        )
        for _ in range(repeat_updates):
            trainer.evaluate()
            evaluated_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in trainer.uncompiled_policy.state_dict().items()
            }
            logs = trainer.train()
            if not isinstance(logs, Mapping):
                continue
            metrics = dict(logs)
            if "environment/error" in logs:
                last_error = float(logs["environment/error"])
                min_error = min(min_error, last_error)
                if last_error < repeat_best_error:
                    repeat_best_error = last_error
                    repeat_best_state_dict = evaluated_state_dict
        errors += [last_error] if last_error is not None else []

        if repeat_best_state_dict is not None:
            path = (
                Path(trainer.config["data_dir"])
                / trainer.logger.run_id
                / "model_best.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(repeat_best_state_dict, path)

        model_path = trainer.close()
        if repeat_best_state_dict is not None:
            eval_errors.append(
                _evaluate_frozen_policy(
                    trainer.uncompiled_policy,
                    repeat_best_state_dict,
                    cfg,
                    env_kwargs,
                )
            )
        step_offset += int(trainer.global_step)
        if updates_remaining is not None:
            updates_remaining -= repeat_updates

    if errors:
        print(f"Final environment error: {errors[-1]:.6f}")
        print(f"Minimum environment error: {min_error:.6f}")
        metrics = {"environment/error_min": float(min_error)}
        if cfg.train.n_repeat > 1:
            metrics["environment/error_mean"] = float(np.mean(errors))
            metrics["environment/error_std"] = float(np.std(errors))
    if eval_errors:
        means, mins, maxes = zip(*eval_errors, strict=True)
        metrics["environment/error_eval"] = float(np.mean(means))
        metrics["environment/error_eval_min"] = float(min(mins))
        metrics["environment/error_eval_max"] = float(max(maxes))
        print(f"Frozen evaluation error: {metrics['environment/error_eval']:.6f}")

    if errors or eval_errors:
        run.log(metrics, step=step_offset)

    if logger is not None:
        logger.close(model_path)

    if cfg.train.generate_rollout_video:
        from magna.eval import run_policy_inference

        run_policy_inference(
            env_kwargs=render_env_kwargs,
            policy=policy,
            device=cfg.train.device,
            seed=200,
            n_rollouts=2,
            video_path="outputs/videos/rollout_last_training.mp4",
            collect_errors=False,
        )
    return {
        "run_id": run.id,
        "run_name": run.name,
        "run_dir": run.dir,
        "metrics": metrics,
    }
