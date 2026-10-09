from contextlib import contextmanager
from sys import maxsize
from types import SimpleNamespace

import pytest

import magna.cli.puffer_sweep as mod
from magna.config import Config

SWEEP = {
    "method": "FakeMethod",
    "metric": "error_mean",
    "goal": "minimize",
    "train": {
        "learning_rate": {
            "distribution": "log_normal",
            "min": 1e-4,
            "max": 3e-2,
            "scale": "auto",
            "mean": 1e-2,
        }
    },
    "model": {
        "hidden_size": {
            "distribution": "uniform_pow2",
            "min": 64,
            "max": 512,
            "scale": "auto",
            "mean": 128,
        }
    },
}
MULTI = {
    **SWEEP,
    "datasets": {
        "Grid2d": "/data/Grid2d.pt",
        "Torus": {"data.path": "/data/Torus.pt", "env.k": 4},
    },
}


class FakeSweeper:
    last = None

    def __init__(self, cfg, **kwargs):
        self.cfg, self.kwargs, self.obs = cfg, kwargs, []
        FakeSweeper.last = self

    def suggest(self, fill=None):
        fill["train"]["learning_rate"] = self.cfg["train"]["learning_rate"]["mean"]
        fill["model"]["hidden_size"] = self.cfg["model"]["hidden_size"]["mean"]
        return fill, {}

    def observe(self, hypers, score, cost, is_failure=False):
        self.obs.append((hypers, score, cost, is_failure))


def _run(name, score=None, runtime=30.0, config=None):
    return SimpleNamespace(
        name=name,
        summary={} if score is None else {"error_mean": score, "_runtime": runtime},
        config=config or {"train.learning_rate": 0.02, "model.hidden_size": 256},
    )


@pytest.fixture
def sweep(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    st = {"score": 3.5, "calls": [], "cfg": SWEEP, "args": {}, "ref": None}

    def args(**kw):
        return SimpleNamespace(
            **(
                {
                    "sweep_config": None,
                    "id": "ref-run",
                    "cfg": None,
                    "commit": "HEAD",
                    "iterations": 1,
                    "sweep_id": "test-sweep",
                    "resume": None,
                    "ref_first": False,
                    "no_train_output": True,
                    "overrides": {},
                }
                | kw
            )
        )

    def resume(runs, group="old-sweep"):
        monkeypatch.setenv("WANDB_ENTITY", "test-entity")
        monkeypatch.setattr(mod, "_parse_args", lambda argv=None: args(resume=group))

        class Api:
            def runs(self, path, filters=None):
                return runs

        monkeypatch.setattr(mod.wandb, "Api", Api)

    def train(worktree, overrides, metric_name, cfg_path=None, stream=True):
        st["calls"].append(overrides)
        score = st["score"]
        if isinstance(score, list):
            score = score[len(st["calls"]) - 1]
        return score, 1.0

    @contextmanager
    def pinned(repo, commit="HEAD"):
        yield wt

    monkeypatch.setattr(mod.sweep_cli, "pinned_worktree", pinned)
    monkeypatch.setattr(mod.sweep_cli, "repo_root", lambda: tmp_path)

    def reference_base(repo, run_id=None, cfg=None):
        st["ref"] = run_id, cfg
        return Config().model_dump(mode="json"), None

    monkeypatch.setattr(mod, "_reference_base", reference_base)
    monkeypatch.setattr(mod, "_parse_args", lambda argv=None: args(**st["args"]))
    monkeypatch.setattr(mod, "_load_sweep_config", lambda path: st["cfg"])
    monkeypatch.setattr(mod.pufferlib.sweep, "FakeMethod", FakeSweeper, raising=False)
    monkeypatch.setattr(mod.sweep_cli, "run_train", train)
    st["resume"] = resume
    return st


def test_trial_observe_and_failure(sweep, capsys):
    mod.puffer_sweep([])
    assert FakeSweeper.last.obs[0][1:] == (3.5, 1.0, False)
    assert sweep["calls"][0]["run_name"] == "test-sweep-trial-001"

    sweep["score"], sweep["calls"] = None, []
    mod.puffer_sweep([])
    assert FakeSweeper.last.obs == []
    assert "trial failed or metric missing" in capsys.readouterr().out


def test_multi_dataset_mean_and_partial_fail(sweep):
    sweep["cfg"], sweep["score"] = MULTI, [2.0, 4.0]
    mod.puffer_sweep([])
    assert [c["run_name"] for c in sweep["calls"]] == [
        "test-sweep-trial-001-Grid2d",
        "test-sweep-trial-001-Torus",
    ]
    assert sweep["calls"][0]["data.path"] == "/data/Grid2d.pt"
    assert sweep["calls"][1]["data.path"] == "/data/Torus.pt"
    assert sweep["calls"][1]["env.k"] == 4
    assert FakeSweeper.last.obs[0][1:] == (3.0, 2.0, False)

    sweep["score"], sweep["calls"] = [2.0, None], []
    FakeSweeper.last.obs.clear()
    mod.puffer_sweep([])
    assert FakeSweeper.last.obs == [] and len(sweep["calls"]) == 2


def test_ref_first_and_cli_overrides(sweep):
    sweep["args"] = {"ref_first": True}
    mod.puffer_sweep([])
    assert set(sweep["calls"][0]) == {
        "run_name",
        "group",
        "train.generate_rollout_video",
    }

    sweep["args"] = {"overrides": {"seed": 42}}
    sweep["calls"].clear()
    mod.puffer_sweep([])
    assert sweep["calls"][0]["seed"] == 42
    assert sweep["calls"][0]["train.learning_rate"] == 1e-2


def test_parse_args():
    args = mod._parse_args(
        ["--cfg", "configs/foo.yaml", "--env.lam_nbr_reward=0.7", "--seed", "42"]
    )
    assert args.cfg == "configs/foo.yaml"
    assert args.overrides == {"env.lam_nbr_reward": 0.7, "seed": 42}
    assert mod._parse_args(["--id", "ref-run"]).iterations == maxsize
    assert mod._parse_args([]).cfg is None


def test_sweep_cfg_reference_and_cli_override(sweep):
    sweep["args"] = {"id": None}
    with pytest.raises(ValueError, match="provide --id, --cfg"):
        mod.puffer_sweep([])

    sweep["cfg"] = {**SWEEP, "cfg": "base.yaml"}
    mod.puffer_sweep([])
    assert sweep["ref"] == (None, "base.yaml")
    assert "cfg" not in FakeSweeper.last.cfg

    sweep["args"]["cfg"] = "cli.yaml"
    mod.puffer_sweep([])
    assert sweep["ref"] == (None, "cli.yaml")


def test_sweeper_kwargs_forwarded(sweep):
    sweep["cfg"] = {
        **SWEEP,
        "num_random_samples": 100,
        "seed_with_search_center": False,
        "max_suggestion_cost": 1800,
    }
    mod.puffer_sweep([])
    assert FakeSweeper.last.kwargs == {
        "num_random_samples": 100,
        "seed_with_search_center": False,
        "max_suggestion_cost": 1800,
    }
    assert "num_random_samples" not in FakeSweeper.last.cfg


def test_resume_single_and_multi(sweep):
    sweep["resume"]([_run("old-sweep-trial-003", 2.5), _run("old-sweep-trial-002")])
    mod.puffer_sweep([])
    assert FakeSweeper.last.obs[0][1:] == (2.5, 30.0, False)
    assert sweep["calls"][0]["run_name"] == "old-sweep-trial-004"

    sweep["cfg"], sweep["score"], sweep["calls"] = MULTI, [1.0, 3.0], []
    FakeSweeper.last.obs.clear()
    sweep["resume"](
        [
            _run("old-sweep-trial-001-Grid2d", 2.0, runtime=10.0),
            _run("old-sweep-trial-001-Torus", 4.0, runtime=20.0),
        ]
    )
    mod.puffer_sweep([])
    assert FakeSweeper.last.obs[0][1:] == (3.0, 30.0, False)
    assert FakeSweeper.last.obs[1][1:] == (2.0, 2.0, False)
    assert [c["run_name"] for c in sweep["calls"]] == [
        "old-sweep-trial-002-Grid2d",
        "old-sweep-trial-002-Torus",
    ]


def test_resume_rejects_inconsistent_hypers(sweep):
    sweep["cfg"] = MULTI
    sweep["resume"](
        [
            _run("old-sweep-trial-001-Grid2d", 2.0),
            _run(
                "old-sweep-trial-001-Torus",
                4.0,
                config={"train.learning_rate": 0.03, "model.hidden_size": 256},
            ),
        ]
    )
    with pytest.raises(ValueError, match="trial 1 has inconsistent hyperparameters"):
        mod.puffer_sweep([])


@pytest.mark.parametrize(
    ("run", "match"),
    [
        (_run("old-sweep-trial-001", 2.5), "_runtime"),
        (_run("old-sweep", 2.5), "old-sweep-trial-NNN"),
        (
            _run("old-sweep-trial-001", 2.5, config={"train.learning_rate": 0.02}),
            "config key",
        ),
    ],
)
def test_resume_failures(sweep, run, match):
    if match == "_runtime":
        run.summary.pop("_runtime")
    sweep["resume"]([run])
    with pytest.raises(ValueError, match=match):
        mod.puffer_sweep([])
