from types import SimpleNamespace

import numpy as np
import pytest
import torch

from magna.config import Config, TrainConfig


def test_puffer_trainer_initializes_replay_segment_buffers():
    from magna.train import PuffeRL

    class DummyPolicy:
        hidden_size = 3

    trainer = object.__new__(PuffeRL)
    trainer.config = {"device": "cpu", "use_rnn": True}
    trainer.segments = 4
    trainer.uncompiled_policy = DummyPolicy()

    trainer._init_recurrent_buffers()

    assert trainer.lstm_h0.shape == (4, 3)
    assert trainer.lstm_c0.shape == (4, 3)
    assert not trainer.lstm_h0.any()


def test_frozen_evaluation_reuses_vectorized_inference(monkeypatch):
    from magna.train import _evaluate_frozen_policy

    calls = {}
    policy = torch.nn.Linear(1, 1, bias=False)
    cfg = Config.model_validate(
        {
            "vector": {"num_envs": 4, "num_workers": 2},
            "env": {"time_horizon": -1},
            "train": {"device": "cpu", "seed": 7},
        }
    )
    monkeypatch.setattr(
        "magna.eval.run_policy_inference",
        lambda **kwargs: (
            calls.update(kwargs)
            or {"error_mean": 4.0, "error_min": 3.0, "error_max": 5.0}
        ),
    )

    error = _evaluate_frozen_policy(
        policy,
        {"weight": torch.tensor([[3.0]])},
        cfg,
        {"cfg": cfg.env},
    )

    assert error == (4, 3, 5)
    assert calls["env_kwargs"]["cfg"].num_envs == 4
    assert calls["env_kwargs"]["track_error_range"] is True
    assert calls["env_kwargs"]["cfg"].time_horizon == 1024
    assert calls["seed"] == 7
    assert not policy.training
    assert policy.weight == 3


def test_puffer_trainer_resets_done_recurrent_rows_only():
    from magna.train import PuffeRL

    trainer = object.__new__(PuffeRL)
    trainer.config = {"use_rnn": True}
    trainer.lstm_h = {0: torch.ones(3, 2)}
    trainer.lstm_c = {0: 2 * torch.ones(3, 2)}

    trainer._reset_done_state(slice(0, 3), torch.tensor([False, True, False]))

    torch.testing.assert_close(trainer.lstm_h[0][0], torch.ones(2))
    torch.testing.assert_close(trainer.lstm_h[0][1], torch.zeros(2))
    torch.testing.assert_close(trainer.lstm_c[0][1], torch.zeros(2))


def test_puffer_trainer_snapshots_segment_initial_state_only_at_segment_start():
    from magna.train import PuffeRL

    trainer = object.__new__(PuffeRL)
    trainer.config = {"use_rnn": True}
    trainer.ep_lengths = torch.tensor([0, 0, 1], dtype=torch.int32)
    trainer.ep_indices = torch.arange(3, dtype=torch.int32)
    trainer.lstm_h0 = torch.zeros(3, 2)
    trainer.lstm_c0 = torch.zeros(3, 2)

    state = {
        "env_id": slice(0, 2),
        "lstm_h": torch.ones(2, 2),
        "lstm_c": 2 * torch.ones(2, 2),
    }
    trainer._snapshot_segment_init(state)

    torch.testing.assert_close(trainer.lstm_h0[:2], torch.ones(2, 2))
    torch.testing.assert_close(trainer.lstm_c0[:2], 2 * torch.ones(2, 2))

    trainer._snapshot_segment_init(
        {
            "env_id": slice(2, 3),
            "lstm_h": 9 * torch.ones(1, 2),
            "lstm_c": 9 * torch.ones(1, 2),
        }
    )

    torch.testing.assert_close(trainer.lstm_h0[2], torch.zeros(2))


def test_puffer_trainer_replay_evaluate_wraps_and_restores_hooks(monkeypatch):
    import magna.train as train_mod
    from magna.train import PuffeRL

    class DummyVecEnv:
        def __init__(self):
            self.recv_calls = 0

        def recv(self):
            self.recv_calls += 1
            return (
                torch.zeros(2, 1),
                torch.zeros(2),
                torch.tensor([False, True]),
                torch.tensor([True, False]),
                [],
                torch.tensor([0, 1]),
                torch.ones(2),
            )

    class DummyPolicy:
        def __init__(self):
            self.forward_calls = 0

        def forward_eval(self, observations, state):
            self.forward_calls += 1
            return observations, torch.zeros(observations.shape[0])

    def fake_evaluate(self):
        _, _, _, _, _, env_id, _ = self.vecenv.recv()
        state = {
            "env_id": slice(env_id[0].item(), env_id[-1].item() + 1),
            "lstm_h": 4 * torch.ones(2, 2),
            "lstm_c": 6 * torch.ones(2, 2),
        }
        self.policy.forward_eval(torch.zeros(2, 1), state)
        return "stats"

    trainer = object.__new__(PuffeRL)
    trainer.config = {"device": "cpu", "replay_segment_state": True, "use_rnn": True}
    trainer.vecenv = DummyVecEnv()
    trainer.policy = DummyPolicy()
    trainer.lstm_h = {0: torch.ones(2, 2)}
    trainer.lstm_c = {0: 2 * torch.ones(2, 2)}
    trainer.ep_lengths = torch.zeros(2, dtype=torch.int32)
    trainer.ep_indices = torch.arange(2, dtype=torch.int32)
    trainer.lstm_h0 = torch.zeros(2, 2)
    trainer.lstm_c0 = torch.zeros(2, 2)
    monkeypatch.setattr(train_mod.PufferTrainer, "evaluate", fake_evaluate)

    assert trainer.evaluate() == "stats"

    assert trainer.vecenv.__dict__["recv"].__func__ is DummyVecEnv.recv
    assert trainer.vecenv.__dict__["recv"].__self__ is trainer.vecenv
    assert trainer.policy.__dict__["forward_eval"].__func__ is DummyPolicy.forward_eval
    assert trainer.policy.__dict__["forward_eval"].__self__ is trainer.policy
    torch.testing.assert_close(trainer.lstm_h[0], torch.zeros(2, 2))
    torch.testing.assert_close(trainer.lstm_c[0], torch.zeros(2, 2))
    torch.testing.assert_close(trainer.lstm_h0, 4 * torch.ones(2, 2))
    torch.testing.assert_close(trainer.lstm_c0, 6 * torch.ones(2, 2))


def test_puffer_trainer_train_injects_replay_segment_state(monkeypatch):
    import magna.train as train_mod
    from magna.train import PuffeRL

    class DummyProfile:
        def __call__(self, *_args, **_kwargs):
            pass

        def end(self):
            pass

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3))
            self.seen_state = None

        def forward(self, observations, state):
            self.seen_state = {k: v.clone() for k, v in state.items() if v is not None}
            steps = observations.shape[0] * observations.shape[1]
            return self.logits.expand(steps, -1), torch.zeros(
                observations.shape[:2], device=observations.device
            )

    trainer = object.__new__(PuffeRL)
    trainer.config = {
        "device": "cpu",
        "use_rnn": True,
        "prio_beta0": 0.5,
        "prio_alpha": 1.0,
        "clip_coef": 0.2,
        "vf_clip_coef": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 1.0,
        "anneal_lr": False,
        "total_timesteps": 999,
        "checkpoint_interval": 999,
    }
    trainer.policy = DummyPolicy()
    trainer.profile = DummyProfile()
    trainer.amp_context = torch.amp.autocast("cpu", enabled=False)
    trainer.total_epochs = 10
    trainer.epoch = 1
    trainer.total_minibatches = 1
    trainer.minibatch_segments = 2
    trainer.accumulate_minibatches = 2
    trainer.segments = 2
    trainer.global_step = 1
    trainer.last_log_time = float("inf")
    trainer.ratio = torch.ones(2, 2)
    trainer.observations = torch.zeros(2, 2, 1)
    trainer.actions = torch.zeros(2, 2, dtype=torch.long)
    trainer.logprobs = torch.zeros(2, 2)
    trainer.rewards = torch.zeros(2, 2)
    trainer.terminals = torch.zeros(2, 2)
    trainer.values = torch.zeros(2, 2)
    trainer.lstm_h0 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    trainer.lstm_c0 = 10 + trainer.lstm_h0
    monkeypatch.setattr(
        trainer,
        "_sample_minibatch_indices",
        lambda *_args: (torch.tensor([1, 0]), torch.ones(2, 1)),
    )
    monkeypatch.setattr(
        train_mod,
        "compute_puff_advantage",
        lambda *args: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    trainer.train()

    state = trainer.policy.seen_state
    torch.testing.assert_close(state["lstm_h"], trainer.lstm_h0[[1, 0]].unsqueeze(0))
    torch.testing.assert_close(state["lstm_c"], trainer.lstm_c0[[1, 0]].unsqueeze(0))


def test_puffer_trainer_validates_comm_layout_group_alignment():
    from magna.train import PuffeRL

    class DummyBackbone:
        local_num_agents = 3
        comm_enabled = True

    class DummyPolicy:
        policy = DummyBackbone()

    class DummyVecEnv:
        agents_per_batch = 4

    trainer = object.__new__(PuffeRL)
    trainer.uncompiled_policy = DummyPolicy()
    trainer.vecenv = DummyVecEnv()
    trainer.segments = 6
    trainer.minibatch_segments = 3

    with pytest.raises(ValueError, match="agents_per_batch"):
        trainer._validate_comm_layout()


def test_puffer_trainer_samples_whole_comm_groups_for_ppo():
    from magna.train import PuffeRL

    class DummyBackbone:
        local_num_agents = 3
        comm_enabled = True

    class DummyPolicy:
        policy = DummyBackbone()

    trainer = object.__new__(PuffeRL)
    trainer.uncompiled_policy = DummyPolicy()
    trainer.segments = 12
    trainer.minibatch_segments = 6

    idx, weights = trainer._sample_minibatch_indices(torch.ones(12), 0.5)

    assert idx.shape == (6,)
    assert weights.shape == (6, 1)
    torch.testing.assert_close(
        idx.reshape(-1, 3) % 3, torch.tensor([[0, 1, 2], [0, 1, 2]])
    )


def test_puffer_trainer_noncomm_sampler_matches_puffer_formula():
    from magna.train import PuffeRL

    class DummyPolicy:
        pass

    trainer = object.__new__(PuffeRL)
    trainer.uncompiled_policy = DummyPolicy()
    trainer.segments = 8
    trainer.minibatch_segments = 5
    prio_weights = torch.arange(1, 9, dtype=torch.float32)
    anneal_beta = 0.35

    torch.manual_seed(7)
    idx, weights = trainer._sample_minibatch_indices(prio_weights, anneal_beta)

    torch.manual_seed(7)
    prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
    expected_idx = torch.multinomial(prio_probs, trainer.minibatch_segments)
    expected_weights = (trainer.segments * prio_probs[expected_idx, None]) ** (
        -anneal_beta
    )

    torch.testing.assert_close(idx, expected_idx)
    torch.testing.assert_close(weights, expected_weights)


def test_muon_uses_adamw_only_for_learned_id_embedding():
    from heavyball import ForeachMuon

    from magna.train import PuffeRL

    embedding = torch.nn.Embedding(4, 3, padding_idx=0)
    linear = torch.nn.Linear(3, 2)
    trainer = object.__new__(PuffeRL)
    trainer.uncompiled_policy = SimpleNamespace(
        policy=SimpleNamespace(learned_id=embedding)
    )
    trainer.optimizer = ForeachMuon([*linear.parameters(), embedding.weight], lr=0.1)
    trainer.config = {
        "optimizer": "muon",
        "embedding_learning_rate": 3e-4,
        "embedding_weight_decay": 0.01,
        "min_lr_ratio": 0.25,
    }
    trainer.total_epochs = 10

    trainer._init_embedding_optimizer()

    muon_params = [
        param for group in trainer.optimizer.param_groups for param in group["params"]
    ]
    adamw_params = [
        param
        for group in trainer.embedding_optimizer.param_groups
        for param in group["params"]
    ]
    assert all(param is not embedding.weight for param in muon_params)
    assert adamw_params == [embedding.weight]
    assert trainer.embedding_optimizer.defaults["betas"] == (0.9, 0.999)
    assert trainer.embedding_optimizer.defaults["eps"] == 1e-8
    assert trainer.embedding_scheduler.eta_min == pytest.approx(7.5e-5)
    assert {id(param) for param in muon_params + adamw_params} == {
        id(param) for param in [*linear.parameters(), embedding.weight]
    }

    before = embedding.weight.detach().clone()
    embedding.weight.grad = torch.ones_like(embedding.weight)
    for parameter in linear.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer._step_optimizers()
    assert not torch.equal(embedding.weight, before)


def test_muon_without_learned_ids_is_unchanged():
    from magna.train import PuffeRL

    parameter = torch.nn.Parameter(torch.ones(1))
    trainer = object.__new__(PuffeRL)
    trainer.uncompiled_policy = SimpleNamespace(policy=SimpleNamespace(learned_id=None))
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer.config = {"optimizer": "muon"}
    original_groups = trainer.optimizer.param_groups

    trainer._init_embedding_optimizer()

    assert trainer.optimizer.param_groups is original_groups
    assert trainer.optimizer.param_groups[0]["params"] == [parameter]
    assert trainer.embedding_optimizer is None
    assert trainer.embedding_scheduler is None


def test_embedding_pretraining_updates_learned_ids():
    from magna.train import PuffeRL

    trainer = object.__new__(PuffeRL)
    embedding = torch.nn.Embedding(5, 2, padding_idx=0)
    trainer.config = TrainConfig(embedding_pretrain_steps=2).model_dump()
    trainer.vecenv = SimpleNamespace(
        driver_env=SimpleNamespace(
            _target_neighbors=np.array([[[1, 2], [0, 2], [0, 1], [-1, -1]]]),
            _target_neighbor_distances=np.array(
                [[[1.0, 1.0], [1.0, 2.0], [1.0, 2.0], [np.inf, np.inf]]]
            ),
        )
    )
    trainer.uncompiled_policy = SimpleNamespace(
        policy=SimpleNamespace(learned_id=embedding)
    )
    with torch.no_grad():
        embedding.weight[1:].copy_(
            torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
        )
    trainer._init_embedding_auxiliary()
    assert not (trainer.embedding_aux_pairs[2][:, 0] == 0).any()
    del trainer.embedding_aux_pairs
    before = embedding.weight.detach().clone()
    trainer._pretrain_embeddings()
    assert not torch.equal(embedding.weight, before)
    torch.testing.assert_close(embedding.weight[0], torch.zeros(2))
    torch.testing.assert_close(
        embedding.weight[1:].mean(0), torch.zeros(2), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        embedding.weight[1:].norm(dim=-1).max(), torch.tensor(1.0)
    )
    assert not hasattr(trainer, "embedding_aux_pairs")


def test_embedding_learning_rate_is_logged():
    from magna.train import _EmbeddingLRLogger

    logged = []
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=3e-4)
    logger = _EmbeddingLRLogger(
        SimpleNamespace(log=lambda values, step: logged.append((values, step))),
        optimizer,
    )

    logs = {"agent_steps": 7}
    logger.log(logs, 7)

    assert logs["embedding_learning_rate"] == pytest.approx(3e-4)
    assert logged == [({"agent_steps": 7, "embedding_learning_rate": 3e-4}, 7)]
