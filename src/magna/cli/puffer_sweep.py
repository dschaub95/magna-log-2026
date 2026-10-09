import argparse
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pufferlib.sweep
import wandb
import yaml

from magna.cli import sweep as sweep_cli
from magna.config import (
    Config,
    apply_cli_overrides,
    flatten,
    load_yaml_config,
    parse_dot_overrides,
    unflatten,
)
from magna.utils import save_wandb_run_config_to_yaml

_SWEEPER_KWARGS = frozenset(
    {
        "max_suggestion_cost",
        "resample_frequency",
        "num_random_samples",
        "global_search_scale",
        "random_suggestions",
        "suggestions_per_pareto",
        "seed_with_search_center",
        "expansion_rate",
    }
)
SWEEP_META_KEYS = {
    "method",
    "metric",
    "goal",
    "downsample",
    "metric_distribution",
    "use_gpu",
    "prune_pareto",
    "sweep_only",
    "early_stop_quantile",
    "datasets",
    "cfg",
} | _SWEEPER_KWARGS
_PROTEIN_META = frozenset({"method", "metric", "goal", "downsample"})

DEFAULT_SWEEP_CONFIG: dict[str, Any] = {
    "method": "Protein",
    "metric": "environment/error_eval",
    "goal": "minimize",
    "downsample": 8,
    "train": {
        "learning_rate": {
            "distribution": "log_normal",
            "min": 0.001,
            "max": 0.1,
            "scale": 0.5,
            "mean": 1e-2,
        },
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", default=None)
    ref = parser.add_mutually_exclusive_group()
    ref.add_argument("--id", "--run-id")
    ref.add_argument("--cfg")
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--iterations", type=int, default=sys.maxsize)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ref-first", action="store_true")
    parser.add_argument("--no-train-output", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    overrides, rest = parse_dot_overrides(remaining)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    args.overrides = overrides
    return args


def _load_sweep_config(path: str | None) -> dict[str, Any]:
    if path is None:
        print("No --sweep_config provided. Using DEFAULT_SWEEP_CONFIG.")
        data = deepcopy(DEFAULT_SWEEP_CONFIG)
    else:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("sweep config must be a mapping")
    for key in ("method", "metric", "goal"):
        if key not in data:
            raise ValueError(f"missing required sweep key: {key}")
    return data


def _parse_datasets(raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, value in (raw or {}).items():
        ov = {"data.path": value} if isinstance(value, str) else flatten(value)
        if "data.path" not in ov:
            raise ValueError(f"dataset '{name}' missing data.path")
        out[str(name)] = ov
    return out


def _extract_sweep_hypers(
    base_cfg: dict[str, Any], sweep_cfg: dict[str, Any]
) -> dict[str, Any]:
    hypers: dict[str, Any] = {}
    for key, value in sweep_cfg.items():
        if key in SWEEP_META_KEYS:
            continue
        if key not in base_cfg:
            raise ValueError(f"sweep key '{key}' not found in base config")
        if isinstance(value, dict) and "distribution" in value:
            hypers[key] = base_cfg[key]
        elif isinstance(value, dict):
            if not isinstance(base_cfg[key], dict):
                raise ValueError(
                    f"sweep subtree '{key}' expects a mapping in base config"
                )
            hypers[key] = _extract_sweep_hypers(base_cfg[key], value)
    return hypers


def _reference_base(
    repo: Path, *, run_id: str | None, cfg: str | None
) -> tuple[dict[str, Any], Path]:
    if run_id:
        cfg = save_wandb_run_config_to_yaml(
            run_id, repo / "outputs" / "wandb" / "sweep-configs" / f"{run_id}.yaml"
        )
    return (
        load_yaml_config(str(cfg), resolve=False).model_dump(mode="json"),
        sweep_cli.snapshot_cfg(repo, cfg),
    )


def _trial_info(group: str, run: Any) -> tuple[int, str | None]:
    match = re.fullmatch(rf"{re.escape(group)}-trial-(\d+)(?:-(.+))?", str(run.name))
    return (int(match.group(1)), match.group(2)) if match else (0, None)


def _mean_or_none(scores: list[float | None]) -> float | None:
    vals = [s for s in scores if s is not None]
    return None if not scores or len(vals) != len(scores) else sum(vals) / len(vals)


def _is_better(score: float, best: float | None, goal: str) -> bool:
    if best is None:
        return True
    return (score > best) if goal == "maximize" else (score < best)


def _resume_sweep(
    sweeper: Any,
    group: str,
    *,
    project: str,
    metric: str,
    goal: str,
    base_hypers: dict[str, Any],
    dataset_names: list[str],
) -> tuple[int, float | None]:
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or getattr(api, "default_entity", None)
    if not entity:
        raise ValueError("Could not infer W&B entity from WANDB_ENTITY or wandb.Api()")
    runs = [
        run
        for run in api.runs(f"{entity}/{project}", filters={"group": group})
        if _trial_info(group, run)[0]
    ]
    if not runs:
        raise ValueError(f"No W&B runs named '{group}-trial-NNN' found")

    by_trial: dict[int, list[Any]] = {}
    for run in runs:
        by_trial.setdefault(_trial_info(group, run)[0], []).append(run)

    flat_keys = flatten(base_hypers).keys()
    best_score = None
    for trial, trial_runs in sorted(by_trial.items()):
        if dataset_names:
            found = {ds: r for r in trial_runs if (ds := _trial_info(group, r)[1])}
            if any(n not in found for n in dataset_names):
                continue
            trial_runs = [found[n] for n in dataset_names]

        scores, costs, hypers = [], [], None
        for run in trial_runs:
            score = run.summary.get(metric)
            scores.append(None if score is None else float(score))
            if score is None:
                continue
            if "_runtime" not in run.summary:
                raise ValueError(f"W&B run '{run.name}' missing summary key '_runtime'")
            missing = [k for k in flat_keys if k not in run.config]
            if missing:
                raise ValueError(
                    f"W&B run '{run.name}' missing config key '{missing[0]}'"
                )
            run_hypers = unflatten({k: run.config[k] for k in flat_keys})
            if hypers is not None and run_hypers != hypers:
                raise ValueError(f"W&B trial {trial} has inconsistent hyperparameters")
            hypers = run_hypers
            costs.append(float(run.summary["_runtime"]))

        score = _mean_or_none(scores)
        if score is None or hypers is None:
            continue
        sweeper.observe(hypers, score, sum(costs))
        if _is_better(score, best_score, goal):
            best_score = score
    return max(by_trial, default=0) + 1, best_score


def puffer_sweep(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sweep_cfg = _load_sweep_config(args.sweep_config)
    metric = str(sweep_cfg["metric"])
    goal = str(sweep_cfg["goal"]).lower()
    method = str(sweep_cfg["method"])
    datasets = _parse_datasets(sweep_cfg.get("datasets"))
    cfg = args.cfg or sweep_cfg.get("cfg")
    if not args.id and not cfg:
        raise ValueError("provide --id, --cfg, or 'cfg' in the sweep config")

    repo = sweep_cli.repo_root()
    base_cfg, cfg_path = _reference_base(repo, run_id=args.id, cfg=cfg)
    base_cfg = apply_cli_overrides(
        Config.model_validate(base_cfg), args.overrides
    ).model_dump(mode="json")
    base_hypers = _extract_sweep_hypers(base_cfg, sweep_cfg)
    try:
        cls = getattr(pufferlib.sweep, method)
    except AttributeError as exc:
        raise ValueError(f"unknown pufferlib.sweep method: {method}") from exc
    config = {
        k: v
        for k, v in sweep_cfg.items()
        if k in _PROTEIN_META or k not in SWEEP_META_KEYS
    }
    kwargs = {k: sweep_cfg[k] for k in _SWEEPER_KWARGS if k in sweep_cfg}
    sweeper = cls(config, **kwargs)

    sweep_id = (
        args.resume or args.sweep_id or f"puffer-{datetime.now():%Y-%m-%d-%H-%M-%S}"
    )
    start_trial, best_score = 1, None
    if args.resume:
        start_trial, best_score = _resume_sweep(
            sweeper,
            args.resume,
            project=str(base_cfg["project"]),
            metric=metric,
            goal=goal,
            base_hypers=base_hypers,
            dataset_names=list(datasets),
        )
    print(
        f"Sweep ID: {sweep_id}\nMethod: {method}\nMetric: {metric}\n"
        f"Reference config: {args.id or cfg}"
        + (f"\nDatasets: {', '.join(datasets)}" if datasets else "")
    )

    with sweep_cli.pinned_worktree(repo, args.commit) as worktree:
        for trial in range(start_trial, start_trial + args.iterations):
            trial_hypers = deepcopy(base_hypers)
            apply_suggest = not args.ref_first or trial > 1
            if apply_suggest:
                trial_hypers, _ = sweeper.suggest(fill=trial_hypers)

            shared = {
                **args.overrides,
                **(flatten(trial_hypers) if apply_suggest else {}),
                "group": sweep_id,
                "train.generate_rollout_video": False,
            }
            scores, costs, run_names = [], [], []
            for name, ds_ov in datasets.items() or [(None, {})]:
                run_name = f"{sweep_id}-trial-{trial:03d}" + (
                    f"-{name}" if name else ""
                )
                run_names.append(run_name)
                score, cost = sweep_cli.run_train(
                    worktree,
                    {**shared, **ds_ov, "run_name": run_name},
                    metric,
                    cfg_path=cfg_path,
                    stream=not args.no_train_output,
                )
                scores.append(score)
                costs.append(cost)

            score, cost = _mean_or_none(scores), sum(costs)
            if score is None:
                print(
                    "[warning] trial failed or metric missing; "
                    f"trial={trial} run_names={run_names} "
                    f"params={json.dumps(flatten(trial_hypers), sort_keys=True)}"
                )
            else:
                sweeper.observe(trial_hypers, float(score), cost)
                if _is_better(score, best_score, goal):
                    best_score = score
            print(
                f"trial={trial} score={score} cost_seconds={cost:.2f} "
                f"best_score={best_score} failure={score is None}"
            )


if __name__ == "__main__":
    puffer_sweep()
