import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import wandb
import yaml

from magna.config import Config, parse_dot_overrides

DEFAULT_SWEEP_CONFIG = {
    "command": [sys.executable, "-m", "magna.cli.train"],
    "method": "random",
    "parameters": {
        "train.learning_rate": {"values": [1.5e-2, 1.0e-2, 5.0e-3]},
        "model.hidden_size": {"values": [128, 256, 512]},
        "train.bptt_horizon": {"values": [32, 64, 96]},
    },
    "metric": {"name": "environment/error", "goal": "minimize"},
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def repo_root() -> Path:
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))


def _abs(repo: Path, path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _lock_path(repo: Path, sha: str) -> Path:
    path = repo / "outputs" / "wandb" / "sweep-locks" / f"{sha}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _sweep_lock(repo: Path, sha: str):
    with open(_lock_path(repo, sha), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        yield


@contextmanager
def _removal_lock(repo: Path, sha: str):
    with open(_lock_path(repo, sha), "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True


def _sweep_running(repo: Path, sha: str) -> bool:
    with _removal_lock(repo, sha) as removable:
        return not removable


def _inspect_worktree(wt: Path, common_dir: str) -> tuple[str, bool, bool]:
    sha = _git(wt, "rev-parse", "HEAD")
    valid = (
        _git(wt, "rev-parse", "--path-format=absolute", "--git-common-dir")
        == common_dir
    )
    clean = not _git(
        wt,
        "status",
        "--porcelain",
        "--",
        ".",
        ":(exclude)data",
        ":(exclude)outputs",
    )
    return sha, valid, clean


def _clean_worktrees(repo: Path, keep_sha: str) -> None:
    common_dir = _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    for wt in repo.parent.glob(f"{repo.name}-sweep-*"):
        try:
            sha, valid, clean = _inspect_worktree(wt, common_dir)
        except subprocess.CalledProcessError:
            print(f"⚠️  Warning: Keeping {wt}; could not inspect it.")
            continue
        if not valid or not clean:
            reason = (
                "not this repository's worktree" if not valid else "has local changes"
            )
            print(f"⚠️  Warning: Keeping {wt}; {reason}.")
            continue
        if sha == keep_sha:
            continue
        with _removal_lock(repo, sha) as removable:
            if removable:
                try:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(wt)],
                        cwd=repo,
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    print(f"⚠️  Warning: Keeping {wt}; remove failed.")


def _prepare_worktree(repo: Path, sha: str) -> Path:
    if _git(repo, "status", "--porcelain"):
        print(
            f"⚠️  Warning: main repo has uncommitted changes. "
            f"Sweep runs pinned code at {sha[:7]}, not local edits."
        )
    wt = repo.parent / f"{repo.name}-sweep-{sha[:7]}"
    common_dir = _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if wt.exists():
        try:
            actual_sha, valid, clean = _inspect_worktree(wt, common_dir)
        except subprocess.CalledProcessError:
            valid = False
            actual_sha, clean = "", False
        if not valid or not clean or actual_sha != sha:
            raise RuntimeError(f"{wt} is not a clean worktree at {sha[:7]}")
    else:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), sha],
            cwd=repo,
            check=True,
        )
    for name in ("data", "outputs"):
        target = (repo / name).resolve()
        if name == "outputs":
            target.mkdir(parents=True, exist_ok=True)
        link = wt / name
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink(missing_ok=True)
        link.symlink_to(target)
    return wt


@contextmanager
def pinned_worktree(repo: Path, commit: str = "HEAD"):
    os.environ["WANDB_DIR"] = str(repo / "outputs")
    sha = _git(repo, "rev-parse", commit)
    with _sweep_lock(repo, sha):
        _clean_worktrees(repo, sha)
        wt = _prepare_worktree(repo, sha)
        print(f"Sweep commit: {sha[:7]} (worktree: {wt})")
        yield wt


def snapshot_cfg(repo: Path, cfg_path: str | Path) -> Path:
    path = _abs(repo, cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    snapshot_dir = repo / "outputs" / "wandb" / "sweep-configs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = (
        snapshot_dir / f"{hashlib.sha256(path.read_bytes()).hexdigest()}{path.suffix}"
    )
    if not snapshot.exists():
        shutil.copy2(path, snapshot)
    return snapshot


def worktree_env(worktree: Path) -> dict[str, str]:
    return os.environ | {"PYTHONPATH": str(worktree / "src")}


def run_train(
    worktree: Path,
    overrides: dict[str, Any],
    metric_name: str,
    *,
    cfg_path: Path,
    stream: bool = True,
) -> tuple[float | None, float]:
    started_at = time.time()
    fd, metrics_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    path = Path(metrics_path)
    try:
        cmd = [
            sys.executable,
            "-m",
            "magna.cli.train",
            f"--metrics-out={path}",
            f"--cfg={cfg_path}",
        ]
        cmd.extend(f"--{k}={json.dumps(v)}" for k, v in overrides.items())
        subprocess.run(
            cmd,
            cwd=worktree,
            env=worktree_env(worktree),
            check=True,
            capture_output=not stream,
            text=not stream,
        )
        metrics = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None, max(0.0, time.time() - started_at)
    finally:
        path.unlink(missing_ok=True)
    return metrics.get(metric_name), max(0.0, time.time() - started_at)


def main(
    cfg: dict | str | None = None,
    name: str | None = None,
    count: int | None = None,
):
    argv = sys.argv[1:]
    flat_cli, _ = parse_dot_overrides(argv)
    project = flat_cli.get("project", Config().project)
    base_cfg = flat_cli.get("cfg")
    cfg = flat_cli.get("sweep-cfg", cfg)
    name = flat_cli.get("name", name)
    count = flat_cli.get("count", count)
    commit = flat_cli.get("commit", "HEAD")
    wandb_mode = flat_cli.get("wandb_mode")
    if wandb_mode is not None:
        os.environ["WANDB_MODE"] = str(wandb_mode)

    repo = repo_root()
    if isinstance(cfg, str):
        cfg = str(_abs(repo, cfg))
        assert Path(cfg).exists(), "Sweep config file does not exist."
        with open(cfg) as f:
            cfg = yaml.safe_load(f)

    if cfg is None:
        print("⚠️  Warning: No sweep-cfg specified. Using default sweep configuration.")
        cfg = DEFAULT_SWEEP_CONFIG
    if name is not None:
        cfg["name"] = name
    command = list(cfg.get("command", DEFAULT_SWEEP_CONFIG["command"]))
    if command == ["uv", "run", "train"]:
        command = list(DEFAULT_SWEEP_CONFIG["command"])
    if base_cfg is not None:
        command.append(f"--cfg={snapshot_cfg(repo, base_cfg)}")
    cfg["command"] = command

    parameters = cfg.get("parameters")
    assert isinstance(parameters, dict), "cfg['parameters'] must be a mapping"
    parameters["train.generate_rollout_video"] = {"value": False}

    with pinned_worktree(repo, str(commit)) as worktree:
        sweep_id = wandb.sweep(cfg, project=project)

        print(f"Sweep created with ID: {sweep_id}")
        print(f"Sweep config: {cfg}")
        print(f"Project: {project}")

        command = ["wandb", "agent", sweep_id]
        if count is not None:
            command += ["--count", str(count)]

        subprocess.run(command, check=True, cwd=worktree, env=worktree_env(worktree))
        wandb.finish()


if __name__ == "__main__":
    main()
