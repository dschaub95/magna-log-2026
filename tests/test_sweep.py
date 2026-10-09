import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import magna.cli.sweep as sweep_mod

_real_prepare_worktree = sweep_mod._prepare_worktree


@pytest.fixture(autouse=True)
def _mock_worktree(monkeypatch, tmp_path):
    wt = tmp_path / "worktree"
    wt.mkdir()
    monkeypatch.setenv("WANDB_DIR", os.environ.get("WANDB_DIR", ""))

    @contextmanager
    def fake_pinned(repo, commit="HEAD"):
        os.environ["WANDB_DIR"] = str(repo / "outputs")
        yield wt

    monkeypatch.setattr(sweep_mod, "pinned_worktree", fake_pinned)
    monkeypatch.setattr(sweep_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(sweep_mod, "_git", lambda cwd, *args: "a" * 40)
    return wt


def test_sweep_forces_rollout_video_disabled(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        sweep_mod,
        "parse_dot_overrides",
        lambda argv: ({}, []),
    )
    monkeypatch.setattr(
        sweep_mod.wandb,
        "sweep",
        lambda sweep_config, project=None: (
            captured.update({"sweep_config": sweep_config, "project": project})
            or "sweep-id-123"
        ),
    )
    monkeypatch.setattr(
        sweep_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(sweep_mod.wandb, "finish", lambda: None)

    sweep_mod.main(cfg={"method": "random", "parameters": {}})

    params = captured["sweep_config"]["parameters"]
    assert params["train.generate_rollout_video"] == {"value": False}


def test_sweep_requires_parameters_mapping(monkeypatch):
    monkeypatch.setattr(
        sweep_mod,
        "parse_dot_overrides",
        lambda argv: ({}, []),
    )

    with pytest.raises(AssertionError, match="cfg\\['parameters'\\] must be a mapping"):
        sweep_mod.main(cfg={"method": "random", "parameters": None})


def test_sweep_injects_base_cfg_into_command(tmp_path, monkeypatch):
    captured: dict = {}
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text("project: magna\n")

    monkeypatch.setattr(
        sweep_mod,
        "parse_dot_overrides",
        lambda argv: ({"cfg": str(base_cfg)}, []),
    )
    monkeypatch.setattr(
        sweep_mod.wandb,
        "sweep",
        lambda sweep_config, project=None: (
            captured.update({"sweep_config": sweep_config}) or "sweep-id-123"
        ),
    )

    def run(*args, **kwargs):
        captured["base_cfg"] = Path(
            captured["sweep_config"]["command"][-1].removeprefix("--cfg=")
        ).read_text()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep_mod.subprocess, "run", run)
    monkeypatch.setattr(sweep_mod.wandb, "finish", lambda: None)

    sweep_mod.main(
        cfg={
            "method": "random",
            "command": ["uv", "run", "train"],
            "parameters": {},
        }
    )

    assert captured["sweep_config"]["command"][:3] == [
        sys.executable,
        "-m",
        "magna.cli.train",
    ]
    assert captured["base_cfg"] == "project: magna\n"
    snapshot = Path(captured["sweep_config"]["command"][-1].removeprefix("--cfg="))
    assert snapshot.parent == tmp_path / "outputs" / "wandb" / "sweep-configs"
    assert os.environ["WANDB_DIR"] == str(tmp_path / "outputs")


def test_sweep_runs_agent_in_worktree(monkeypatch, _mock_worktree):
    captured: dict = {}

    monkeypatch.setattr(
        sweep_mod,
        "parse_dot_overrides",
        lambda argv: ({}, []),
    )
    monkeypatch.setattr(
        sweep_mod.wandb,
        "sweep",
        lambda sweep_config, project=None: "sweep-id-123",
    )
    monkeypatch.setattr(
        sweep_mod.subprocess,
        "run",
        lambda *args, **kwargs: (
            captured.update(kwargs) or SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.setattr(sweep_mod.wandb, "finish", lambda: None)

    sweep_mod.main(cfg={"method": "random", "parameters": {}})

    assert captured["cwd"] == _mock_worktree
    assert captured["env"]["PYTHONPATH"] == str(_mock_worktree / "src")


def test_sweep_warns_on_dirty_repo(monkeypatch, tmp_path, capsys):
    sha = "c" * 40
    wt = tmp_path.parent / f"{tmp_path.name}-sweep-{sha[:7]}"
    wt.mkdir()

    def fake_git(cwd, *args):
        if args[0] == "rev-parse":
            return sha
        if args[:2] == ("status", "--porcelain"):
            return " M src/foo.py" if cwd == tmp_path else ""
        raise AssertionError((cwd, args))

    monkeypatch.setattr(sweep_mod, "_git", fake_git)
    _real_prepare_worktree(tmp_path, sha)

    out = capsys.readouterr().out
    assert "uncommitted changes" in out


def test_prepare_worktree_replaces_tracked_data_dir(monkeypatch, tmp_path):
    repo = tmp_path / "magna"
    repo.mkdir()
    sha = "d" * 40
    wt = repo.parent / f"magna-sweep-{sha[:7]}"
    wt.mkdir()
    (wt / "data").mkdir()
    (wt / "data" / ".gitkeep").write_text("")
    (repo / "data").mkdir()
    (repo / "data" / "cells.h5ad").write_text("x")

    def fake_git(cwd, *args):
        if args[0] == "rev-parse":
            return sha
        if args[:2] == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(sweep_mod, "_git", fake_git)
    out_wt = _real_prepare_worktree(repo, sha)
    assert out_wt == wt
    assert (wt / "data").is_symlink()
    assert (wt / "data" / "cells.h5ad").read_text() == "x"


def test_prepare_worktree_rejects_unrelated_repo(monkeypatch, tmp_path):
    repo = tmp_path / "magna"
    wt = tmp_path / "magna-sweep-ddddddd"
    repo.mkdir()
    wt.mkdir()
    data = wt / "data"
    data.mkdir()

    def fake_git(cwd, *args):
        if args == ("status", "--porcelain"):
            return ""
        if args[-1] == "--git-common-dir":
            return str(cwd / ".git")
        return "d" * 40

    monkeypatch.setattr(sweep_mod, "_git", fake_git)
    with pytest.raises(RuntimeError, match="not a clean worktree"):
        _real_prepare_worktree(repo, "d" * 40)
    assert data.is_dir()


def test_prepare_worktree_rejects_modified_code(monkeypatch, tmp_path):
    repo = tmp_path / "magna"
    wt = tmp_path / "magna-sweep-ddddddd"
    repo.mkdir()
    wt.mkdir()

    def fake_git(cwd, *args):
        if args[:2] == ("status", "--porcelain"):
            return " M src/train.py" if cwd == wt else ""
        return str(repo / ".git") if args[-1] == "--git-common-dir" else "d" * 40

    monkeypatch.setattr(sweep_mod, "_git", fake_git)
    with pytest.raises(RuntimeError, match="not a clean worktree"):
        _real_prepare_worktree(repo, "d" * 40)


def test_sweep_lock_tracks_running_process(tmp_path):
    with sweep_mod._sweep_lock(tmp_path, "a" * 40):
        assert sweep_mod._sweep_running(tmp_path, "a" * 40)
    assert not sweep_mod._sweep_running(tmp_path, "a" * 40)


def test_clean_worktrees_keeps_active_sweep(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "magna"
    active = tmp_path / "magna-sweep-bbbbbbb"
    stale = tmp_path / "magna-sweep-ccccccc"
    dirty = tmp_path / "magna-sweep-ddddddd"
    for path in (repo, active, stale, dirty):
        path.mkdir()

    def fake_git(cwd, *args):
        if args[-1] == "--git-common-dir":
            return str(repo / ".git")
        if args[-1] == "HEAD":
            return {active: "b", stale: "c", dirty: "d"}[cwd] * 40
        if args[:2] == ("status", "--porcelain"):
            return " M src/train.py" if cwd == dirty else ""
        raise AssertionError(args)

    removed = []
    monkeypatch.setattr(sweep_mod, "_git", fake_git)

    def remove(command, **kwargs):
        assert sweep_mod._sweep_running(repo, "c" * 40)
        removed.append(command)

    monkeypatch.setattr(sweep_mod.subprocess, "run", remove)

    with sweep_mod._sweep_lock(repo, "b" * 40):
        sweep_mod._clean_worktrees(repo, "a" * 40)

    assert removed == [["git", "worktree", "remove", "--force", str(stale)]]
    assert f"Keeping {dirty}; has local changes" in capsys.readouterr().out
