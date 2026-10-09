"""Long-horizon (10k-step) pattern-formation eval on MAIN targets."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import (
    MAIN,
    UNBOUNDED_FLAT,
    checkpoint,
    even_snap_times,
    experiment_dirs,
    format_metrics_latex_table,
    load_run_cfg,
    mean_std,
    metrics_line,
    plot_snapshots,
    stale_checkpoints,
)
from magna.eval import load_eval_data, run_eval

EXPERIMENT = "05-long-horizon"
RESULT_DIR, TABLE_DIR, FIG_DIR = experiment_dirs(
    EXPERIMENT, "results", "tables", "figures"
)
CSV_PATH, SNAP_DIR = RESULT_DIR / "metrics.csv", RESULT_DIR / "snapshots"
FIELDS = ("dataset", "seed", "mse", "gw", "checkpoint")
SNAP_PLOT = tuple((n, 1 if n == "Cube" else 0) for n in MAIN)


def _snap_rows():
    rows = []
    for name, seed in SNAP_PLOT:
        d = np.load(SNAP_DIR / f"{name}_seed{seed}.npz")
        times = tuple(map(int, d["times"]))
        targets, *_ = load_eval_data(load_run_cfg(name, resolve=True))
        rows.append(
            (name, targets, dict(zip(times, d["positions"], strict=True)), times)
        )
    return rows


def _run_eval():
    rows = []
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    for name in MAIN:
        ckpt = checkpoint(name)
        for seed in range(5):
            out = run_eval(
                load_run_cfg(name),
                ckpt,
                video_path=None,
                return_positions="all",
                autoreset=False,
                flat_overrides={
                    "train.seed": seed,
                    "env.time_horizon": 10_000,
                    **UNBOUNDED_FLAT,
                },
            )
            m, traj = out["metrics"], out["positions"][0]
            row = {
                "dataset": name,
                "seed": seed,
                "mse": float(m["mse"]),
                "gw": float(m["gw"]),
                "checkpoint": str(ckpt),
            }
            rows.append(row)
            times = even_snap_times(traj.shape[0] - 1)
            np.savez_compressed(
                SNAP_DIR / f"{name}_seed{seed}.npz",
                times=np.asarray(times, dtype=np.int32),
                positions=np.stack([traj[t] for t in times]),
            )
            print(f"{name:12s} seed={seed} {metrics_line(row)} ckpt={ckpt}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--rerun-eval", action="store_true")
    args = p.parse_known_args()[0]
    have_snaps = all(
        (SNAP_DIR / f"{ds}_seed{s}.npz").exists() for ds in MAIN for s in range(5)
    )
    rows = list(csv.DictReader(CSV_PATH.open())) if CSV_PATH.exists() else []
    if args.rerun_eval or not have_snaps or stale_checkpoints(rows, MAIN):
        rows = _run_eval()

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append({"mse": float(r["mse"]), "gw": float(r["gw"])})
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tex, fig = TABLE_DIR / "eval_metrics.txt", FIG_DIR / "05-long-horizon-snapshots.png"
    format_metrics_latex_table(
        {"MAGNA (ours)": {ds: mean_std(ms) for ds, ms in by_ds.items()}},
        highlight_best=False,
        caption=(
            r"Final normalized MSE and GW distance after $10{,}000$-step unbounded "
            r"pattern-formation rollouts. Values report mean \(\pm\) standard "
            r"deviation over five evaluation seeds. Lower is better."
        ),
        label="tab:long-horizon",
        out_path=tex,
    )
    print(tex.resolve())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.close(plot_snapshots(_snap_rows(), fig))
    print(fig.resolve())


if __name__ == "__main__":
    main()
