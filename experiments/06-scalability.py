"""Bunny agent-count scalability: final MSE and optional W&B episode-return curves."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import (
    UNBOUNDED_FLAT,
    checkpoint,
    experiment_dirs,
    load_run_cfg,
    mean_std,
    metrics_line,
    plot_scalability,
    stale_checkpoints,
)
from magna.eval import run_eval

EXPERIMENT = "06-scalability"
RESULT_DIR, FIG_DIR = experiment_dirs(EXPERIMENT, "results", "figures")
CSV_PATH, RET_DIR = RESULT_DIR / "metrics.csv", RESULT_DIR / "returns"
BUNNIES = ("Bunny512", "Bunny1024", "Bunny1500", "Bunny2000")
FIELDS = ("dataset", "seed", "mse", "gw", "checkpoint")
# W&B runs of the shipped checkpoints, only used by --include-returns.
WANDB_RUNS = {
    "Bunny512": "dlf6aqyc",
    "Bunny1024": "4t4cod5j",
    "Bunny1500": "e0rkumwf",
    "Bunny2000": "nuq3cbun",
}
RETURN_KEY = "environment/episode_return"
RETURN_SAMPLES = 100


def _n(name: str) -> int:
    return int(name.removeprefix("Bunny"))


def _run_eval():
    rows = []
    for name in BUNNIES:
        ckpt = checkpoint(name)
        for seed in range(5):
            m = run_eval(
                load_run_cfg(name),
                ckpt,
                video_path=None,
                return_positions="last",
                autoreset=False,
                flat_overrides={
                    "train.seed": seed,
                    **UNBOUNDED_FLAT,
                },
            )["metrics"]
            row = {
                "dataset": name,
                "seed": seed,
                "mse": float(m["mse"]),
                "gw": float(m["gw"]),
                "checkpoint": str(ckpt),
            }
            rows.append(row)
            print(f"{name:12s} seed={seed} {metrics_line(row)} ckpt={ckpt}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _returns():
    RET_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in BUNNIES:
        path, rid = RET_DIR / f"{name}.npz", WANDB_RUNS[name]
        if path.exists():
            out[name] = ((d := np.load(path))["steps"], d["returns"])
            continue
        import wandb

        print(f"{name:12s} fetching returns ({rid})...", flush=True)
        df = (
            wandb.Api()
            .run(f"dschaub/magna/{rid}")
            .history(keys=[RETURN_KEY, "agent_steps", "_step"], samples=RETURN_SAMPLES)
        )
        y = df[RETURN_KEY]
        s = (
            df["agent_steps"]
            if "agent_steps" in df and df["agent_steps"].notna().any()
            else df["_step"]
        )
        m = y.notna() & s.notna()
        if not m.any():
            raise SystemExit(f"no {RETURN_KEY} history for {name} ({rid})")
        steps, rets = s[m].to_numpy(float), y[m].to_numpy(float)
        np.savez_compressed(path, steps=steps, returns=rets)
        out[name] = (steps, rets)
        print(f"{name:12s} returns={len(rets)} run={rid}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--rerun-eval", action="store_true")
    p.add_argument(
        "--include-returns", action="store_true", help="Add episode-return subplot."
    )
    args = p.parse_known_args()[0]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(CSV_PATH.open())) if CSV_PATH.exists() else []
    if args.rerun_eval or stale_checkpoints(rows, BUNNIES):
        rows = _run_eval()
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append({"mse": float(r["mse"])})
    mse = sorted((_n(ds), *mean_std(ms)["mse"]) for ds, ms in by_ds.items())
    curves = None
    if args.include_returns:
        curves = [(_n(n), s / s.max(), r) for n, (s, r) in _returns().items()]

    fig_path = FIG_DIR / f"06-scalability-mse{'-returns' if curves else ''}.png"
    plt.close(plot_scalability(mse, curves, fig_path))
    print(fig_path.resolve())


if __name__ == "__main__":
    main()
