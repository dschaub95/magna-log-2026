from typing import Literal, NamedTuple, cast

import numpy as np
import pufferlib.pytorch
import pufferlib.spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from pufferlib.models import LSTMWrapper

from magna.env import MultiCellEnv


class ParsedObs(NamedTuple):
    self_id_embed: torch.Tensor
    self_cell_obs: torch.Tensor
    self_context: torch.Tensor
    neighbor_id_embed: torch.Tensor
    neighbor_obs: torch.Tensor
    neighbor_ids: torch.Tensor


class PufferMLP(nn.Sequential):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        block = lambda x: [
            pufferlib.pytorch.layer_init(nn.Linear(x, hidden_size)),
            nn.GELU(),
        ]
        blocks = block(input_size) + [
            layer for _ in range(num_layers - 1) for layer in block(hidden_size)
        ]
        super().__init__(*blocks)


class Policy(LSTMWrapper):
    def __init__(
        self,
        env: MultiCellEnv,
        n_neighbors: int,
        id_dim: int,
        id_vectors: np.ndarray | torch.Tensor,
        id_feature_encoder: bool = False,
        learned_id_dim: int = 0,
        learned_id_init: Literal["zeros", "gaussian", "orthogonal"] = "zeros",
        mask_nontarget_id_features: bool = False,
        null_msg_learnable: bool = False,
        message_agg: Literal["mean", "sum", "max", "concat", "attn"] = "mean",
        cell_encoder: Literal["mlp", "linear", "raw"] = "mlp",
        hidden_size: int = 128,
        encoder_layers: int = 1,
        comm_enabled: bool = False,
        comm_msg_dim: int = 32,
        action_stay_bias: float = 0.0,
    ):
        backbone: PolicyBackbone = PolicyBackbone(
            env,
            n_neighbors,
            id_dim,
            id_vectors=id_vectors,
            id_feature_encoder=id_feature_encoder,
            learned_id_dim=learned_id_dim,
            mask_nontarget_id_features=mask_nontarget_id_features,
            null_msg_learnable=null_msg_learnable,
            message_agg=message_agg,
            cell_encoder=cell_encoder,
            hidden_size=hidden_size,
            encoder_layers=encoder_layers,
            comm_enabled=comm_enabled,
            comm_msg_dim=comm_msg_dim,
        )
        super().__init__(
            env,
            policy=backbone,
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.policy = cast(PolicyBackbone, self.policy)
        if self.policy.learned_id is not None:
            with torch.no_grad():
                if learned_id_init == "zeros":
                    nn.init.zeros_(self.policy.learned_id.weight)
                elif learned_id_init == "gaussian":
                    self.policy.learned_id.reset_parameters()
                self.policy.learned_id.weight[0].zero_()
        if action_stay_bias and self.policy.is_multidiscrete:
            with torch.no_grad():
                self.policy.decoder.bias[4] += action_stay_bias
                self.policy.decoder.bias[self.policy.action_nvec[0] + 5] += (
                    action_stay_bias
                )

    def _coerce_lstm_state(
        self, state: dict, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        h = state.get("lstm_h")
        c = state.get("lstm_c")
        if h is None or c is None:
            return None
        if h.shape != c.shape:
            raise ValueError("LSTM hidden and cell states must have matching shapes")
        if h.ndim == 3:
            assert h.shape[0] == 1 and h.shape[1] == batch_size, (
                "LSTM state must be (h, c)"
            )
            return h.squeeze(0), c.squeeze(0)
        assert h.ndim == 2 and h.shape[0] == batch_size, "LSTM state must be (h, c)"
        return h, c

    def forward(self, observations, state):
        if not self.policy.comm_enabled:
            return super().forward(observations, state)
        x = observations
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        x = x.reshape(B, TT, *space_shape)  # (B, TT, *obs_shape)
        lstm_state = self._coerce_lstm_state(state, B)
        parsed = self.policy._parse_observations(x.reshape(B * TT, *space_shape))
        parsed = ParsedObs(*(y.reshape(B, TT, *y.shape[1:]) for y in parsed))
        last_hidden = (
            x.new_zeros(B, self.hidden_size) if lstm_state is None else lstm_state[0]
        )

        hidden_seq = []
        for t in range(TT):
            fused = self.policy._build_lstm_input(
                ParsedObs(*(y[:, t] for y in parsed)), last_hidden
            )
            hidden, cell = self.cell(fused, lstm_state)
            hidden_seq.append(hidden)
            lstm_state = (hidden, cell)
            last_hidden = hidden

        hidden = torch.stack(hidden_seq, dim=1)  # (B, TT, H)
        flat_hidden = hidden.reshape(B * TT, self.hidden_size)  # (B * TT, H)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values


class PolicyBackbone(nn.Module):
    def __init__(
        self,
        env: MultiCellEnv,
        n_neighbors: int,
        id_dim: int,
        id_vectors: np.ndarray | torch.Tensor,
        id_feature_encoder: bool = False,
        learned_id_dim: int = 0,
        mask_nontarget_id_features: bool = False,
        null_msg_learnable: bool = False,
        message_agg: Literal["mean", "sum", "max", "concat", "attn"] = "mean",
        cell_encoder: Literal["mlp", "linear", "raw"] = "mlp",
        hidden_size: int = 128,
        encoder_layers: int = 1,
        comm_enabled: bool = False,
        comm_msg_dim: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.n_neighbors = n_neighbors
        self.comm_enabled = comm_enabled
        self.comm_msg_dim = comm_msg_dim
        self.id_input_dim = id_dim
        static_id_dim = min(id_dim, hidden_size) if id_feature_encoder else id_dim
        self.id_dim = static_id_dim + learned_id_dim
        self.message_agg = message_agg
        self.cell_encoder_mode = cell_encoder
        self.perceived_ratio_obs = bool(getattr(env, "perceived_ratio_obs", True))
        self.density_grad_obs = bool(getattr(env, "density_grad_obs", False))
        self.self_pos_delta_obs = bool(getattr(env, "self_pos_delta_obs", True))
        self.boundary_dist_obs = bool(getattr(env, "boundary_dist_obs", False))
        self.comm_select_action = bool(getattr(env, "comm_select_action", False))
        self.self_obs_payload_dim = env.self_obs_payload_dim
        self.neighbor_obs_payload_dim = env.neighbor_obs_payload_dim
        self.neighbor_cell_start = env.neighbor_cell_start
        self.cell_obs_dim = env.cell_obs_dim
        self.vec_obs_dim = env.vec_obs_dim
        self.id_obs_dim = 1  # one compact id index for neighbor/self
        self.neighbor_obs_dim = env.neighbor_obs_dim
        self.self_obs_dim = env.self_obs_dim
        self.cell_input_dim = (
            self.id_dim + self.cell_obs_dim + (comm_msg_dim if comm_enabled else 0)
        )
        self.cell_dim = self.cell_input_dim if cell_encoder == "raw" else hidden_size
        self.edge_dim = self.neighbor_obs_payload_dim - self.cell_obs_dim
        self.context_dim = self.self_obs_payload_dim - self.cell_obs_dim
        if self.message_agg not in {"mean", "sum", "max", "concat", "attn"}:
            raise ValueError(f"Unsupported message_agg='{self.message_agg}'")
        if self.cell_encoder_mode not in {"mlp", "linear", "raw"}:
            raise ValueError(f"Unsupported cell_encoder='{self.cell_encoder_mode}'")
        id_vectors_t = torch.as_tensor(id_vectors, dtype=torch.float32)
        if id_vectors_t.ndim != 2:
            raise ValueError(
                "id_vectors must be a 2D tensor/array with shape (num_agents, id_dim)"
            )
        if int(id_vectors_t.shape[1]) != self.id_input_dim:
            raise ValueError(
                f"id_vectors second dimension ({int(id_vectors_t.shape[1])}) "
                f"must match id_dim ({self.id_input_dim})"
            )
        self.local_num_agents = int(id_vectors_t.shape[0])
        if getattr(env, "agents_per_env", None) not in (None, self.local_num_agents):
            raise ValueError("id_vectors rows must equal env.agents_per_env")
        if id_feature_encoder and id_dim:
            self.id_encoder = PufferMLP(
                self.id_input_dim, static_id_dim, num_layers=self.encoder_layers
            )
        else:
            self.id_encoder = None
        self.learned_id = (
            nn.Embedding(self.local_num_agents + 1, learned_id_dim, padding_idx=0)
            if learned_id_dim
            else None
        )
        zero_id = torch.zeros((1, self.id_input_dim), dtype=id_vectors_t.dtype)
        id_table = torch.cat([zero_id, id_vectors_t], dim=0)
        self.register_buffer("id_table", id_table, persistent=False)
        self.max_id_index = int(id_table.shape[0] - 1)
        self.target_neighbor_bit_index = int(env.target_neighbor_bit_index)
        self.mask_nontarget_id_features = (
            bool(mask_nontarget_id_features) and self.target_neighbor_bit_index >= 0
        )
        self.null_msg = (
            nn.Parameter(torch.zeros(1, 1, hidden_size)) if null_msg_learnable else None
        )

        self.expected_obs_dim = (
            self.n_neighbors * self.neighbor_obs_dim + self.self_obs_dim
        )
        self.update_input_dim = (
            self.cell_dim
            + (n_neighbors * hidden_size if message_agg == "concat" else hidden_size)
            + self.context_dim
        )
        self.neighbor_obs_end = self.n_neighbors * self.neighbor_obs_dim
        self.self_id_end = self.neighbor_obs_end + self.id_obs_dim
        self.is_multidiscrete = isinstance(
            env.single_action_space, pufferlib.spaces.MultiDiscrete
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        obs_dim = int(np.prod(env.single_observation_space.shape))
        if obs_dim != self.expected_obs_dim:
            raise ValueError(
                "Observation dim mismatch for PolicyBackbone: "
                f"expected {self.expected_obs_dim}, got {obs_dim}. "
                f"(n_neighbors={self.n_neighbors}, id_dim={self.id_dim})"
            )

        if cell_encoder == "raw":
            self.cell_encoder = nn.Identity()
        elif cell_encoder == "linear":
            self.cell_encoder = pufferlib.pytorch.layer_init(
                nn.Linear(self.cell_input_dim, hidden_size)
            )
        else:
            self.cell_encoder = PufferMLP(
                self.cell_input_dim, hidden_size, num_layers=self.encoder_layers
            )
        self.comm_proj = (
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, comm_msg_dim))
            if comm_enabled
            else None
        )
        self.message_encoder = PufferMLP(
            2 * self.cell_dim + self.edge_dim,
            hidden_size,
            num_layers=self.encoder_layers,
        )
        self.attn_query = (
            pufferlib.pytorch.layer_init(nn.Linear(self.cell_dim, hidden_size))
            if cell_encoder == "raw" and message_agg == "attn"
            else nn.Identity()
        )
        self.update_encoder = PufferMLP(
            self.update_input_dim, hidden_size, num_layers=self.encoder_layers
        )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01
            )
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01
            )
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(
                torch.zeros(1, env.single_action_space.shape[0])
            )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def _gather_neighbor_messages(
        self, messages: torch.Tensor, neighbor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if messages.shape[0] % self.local_num_agents:
            raise ValueError(
                "Communication batch size must be a multiple of local num_agents"
            )
        rows = torch.arange(
            messages.shape[0], device=neighbor_ids.device, dtype=neighbor_ids.dtype
        )
        row_offsets = (rows // self.local_num_agents * self.local_num_agents).unsqueeze(
            1
        )
        neighbor_ids = torch.where(
            neighbor_ids == 0, neighbor_ids, neighbor_ids + row_offsets
        )
        padded = torch.cat([messages.new_zeros(1, messages.shape[1]), messages])
        return padded[neighbor_ids], neighbor_ids != 0

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        last_hidden = observations.new_zeros(batch_size, self.hidden_size)
        if self.comm_enabled and state is not None:
            h = state.get("lstm_h")
            if h is not None and h.ndim == 2 and h.shape[0] == batch_size:
                last_hidden = h
        return self._build_lstm_input(
            self._parse_observations(observations), last_hidden
        )

    def _parse_observations(self, observations: torch.Tensor) -> ParsedObs:
        obs = observations.view(observations.shape[0], -1).float()  # (B, obs_dim)
        if obs.shape[1] != self.expected_obs_dim:
            raise ValueError(
                "Unexpected observation width in PolicyBackbone observation parsing: "
                f"expected {self.expected_obs_dim}, got {obs.shape[1]}"
            )

        neighbors_obs = obs[:, : self.neighbor_obs_end].reshape(
            -1, self.n_neighbors, self.neighbor_obs_dim
        )
        self_id_obs = obs[:, self.neighbor_obs_end : self.self_id_end]
        self_end = self.self_id_end + self.self_obs_payload_dim
        self_obs = obs[:, self.self_id_end : self_end]

        neighbor_obs = neighbors_obs[..., : self.neighbor_obs_payload_dim]
        neighbor_id_obs = neighbors_obs[..., self.neighbor_obs_payload_dim :]
        neighbor_ids = (
            neighbor_id_obs.squeeze(-1).round().long().clamp_(0, self.max_id_index)
        )
        self_id = self_id_obs.round().long().clamp_(0, self.max_id_index)
        id_lookup = neighbor_ids
        if self.mask_nontarget_id_features:
            bit = neighbor_obs[..., self.target_neighbor_bit_index]
            id_lookup = torch.where(
                bit > 0.5, neighbor_ids, torch.zeros_like(neighbor_ids)
            )
        neighbor_id_embed = F.embedding(id_lookup, self.id_table)
        self_id_embed = F.embedding(self_id, self.id_table).squeeze(1)
        if self.id_encoder is not None:
            neighbor_id_embed = self.id_encoder(neighbor_id_embed)
            self_id_embed = self.id_encoder(self_id_embed)
        if self.learned_id is not None:
            neighbor_id_embed = torch.cat(
                [neighbor_id_embed, self.learned_id(id_lookup)], dim=-1
            )
            self_id_embed = torch.cat(
                [self_id_embed, self.learned_id(self_id).squeeze(1)], dim=-1
            )

        self_cell_obs, self_context = self._split_self_obs(self_obs)
        return ParsedObs(
            self_id_embed,
            self_cell_obs,
            self_context,
            neighbor_id_embed,
            neighbor_obs,
            neighbor_ids,
        )

    def _build_lstm_input(
        self, parsed: ParsedObs, last_hidden: torch.Tensor
    ) -> torch.Tensor:
        last_msg = self.comm_proj(last_hidden) if self.comm_enabled else None
        self_cell = self.cell_encoder(
            self._cell_input(parsed.self_id_embed, parsed.self_cell_obs, last_msg)
        )

        neighbor_msg = None
        if self.comm_enabled:
            neighbor_msg, neighbor_mask = self._gather_neighbor_messages(
                last_msg, parsed.neighbor_ids
            )
            if self.comm_select_action:
                neighbor_mask = neighbor_mask & (parsed.neighbor_obs[..., -1] > 0.5)
            neighbor_msg = neighbor_msg * neighbor_mask.unsqueeze(-1)
        return self._encode_view(
            self_cell,
            parsed.self_context,
            parsed.neighbor_id_embed,
            parsed.neighbor_obs,
            parsed.neighbor_ids,
            neighbor_msg,
        )

    def _encode_view(
        self,
        self_cell: torch.Tensor,
        self_context: torch.Tensor,
        neighbor_id_embed: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_ids: torch.Tensor,
        neighbor_msg: torch.Tensor | None,
    ) -> torch.Tensor:
        start = self.neighbor_cell_start
        end = start + self.cell_obs_dim
        cell_obs = neighbor_obs[..., start:end]
        neighbor_cell = self.cell_encoder(
            self._cell_input(neighbor_id_embed, cell_obs, neighbor_msg)
        )
        valid_mask = neighbor_ids != 0
        edge = torch.cat([neighbor_obs[..., :start], neighbor_obs[..., end:]], dim=-1)
        message_input = torch.cat(
            [self_cell[:, None].expand(-1, self.n_neighbors, -1), neighbor_cell, edge],
            dim=-1,
        )
        messages = self.message_encoder(message_input)
        mask = valid_mask.unsqueeze(-1).to(messages.dtype)
        null_msg = getattr(self, "null_msg", None)
        if self.message_agg == "attn":
            attn_mask = (~valid_mask).unsqueeze(1).to(messages.dtype) * torch.finfo(
                messages.dtype
            ).min
            pooled = F.scaled_dot_product_attention(
                self.attn_query(self_cell).unsqueeze(1),
                messages,
                messages,
                attn_mask=attn_mask,
            ).squeeze(1)
            fallback = (
                torch.zeros_like(pooled) if null_msg is None else null_msg.squeeze(1)
            )
            pooled = torch.where(valid_mask.any(dim=1, keepdim=True), pooled, fallback)
        elif self.message_agg == "concat":
            pooled = (
                messages * mask
                if null_msg is None
                else torch.where(mask.bool(), messages, null_msg)
            ).flatten(1)
        elif self.message_agg == "mean":
            pooled = (messages * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        elif self.message_agg == "sum":
            pooled = (messages * mask).sum(dim=1)
        else:
            neg_inf = torch.finfo(messages.dtype).min
            invalid_mask = ~valid_mask.unsqueeze(-1)
            masked_embed = messages.masked_fill(invalid_mask, neg_inf)
            maxed = masked_embed.max(dim=1).values
            has_valid = valid_mask.any(dim=1, keepdim=True)
            fallback = (
                torch.zeros_like(maxed) if null_msg is None else null_msg.squeeze(1)
            )
            pooled = torch.where(has_valid, maxed, fallback)
        if null_msg is not None and self.message_agg in {"mean", "sum"}:
            has_valid = valid_mask.any(dim=1, keepdim=True)
            pooled = torch.where(has_valid, pooled, null_msg.squeeze(1))
        return self.update_encoder(torch.cat([self_cell, pooled, self_context], dim=1))

    def _cell_input(
        self,
        id_embed: torch.Tensor,
        cell_obs: torch.Tensor,
        comm_msg: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [id_embed, cell_obs]
        parts += [comm_msg] if self.comm_enabled else []
        return torch.cat(parts, dim=-1)

    def _split_self_obs(
        self, self_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.perceived_ratio_obs)
        self_cell_obs = self_obs[:, i : i + self.cell_obs_dim]
        context = []
        if self.perceived_ratio_obs:
            context.append(self_obs[:, :1])
        i += self.cell_obs_dim
        if self.self_pos_delta_obs:
            context.append(self_obs[:, i : i + self.vec_obs_dim])
            i += self.vec_obs_dim
        if self.boundary_dist_obs:
            context.append(self_obs[:, i : i + 1])
            i += 1
        if self.density_grad_obs:
            context.append(self_obs[:, i : i + self.vec_obs_dim])
        return self_cell_obs, (
            torch.cat(context, dim=1)
            if context
            else self_obs.new_zeros(self_obs.shape[0], 0)
        )

    def decode_actions(self, hidden):
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values
