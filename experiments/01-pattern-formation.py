"""Eval pattern-formation targets (shipped checkpoints, or ``--runs``/``--ckpt``)."""

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

from experiments.utils import (
    MAIN,
    UNBOUNDED_FLAT,
    checkpoint,
    even_snap_times,
    experiment_dirs,
    format_metrics_latex_table,
    load_baseline_table,
    load_npa_baselines,
    load_run_cfg,
    mean_std,
    metrics_line,
    plot_metric_bars,
    plot_snapshots,
)
from magna.eval import METRIC_KEYS, load_eval_data, run_eval

EXPERIMENT = "01-pattern-formation"
FIG_DIR, TABLE_DIR = experiment_dirs(EXPERIMENT, "figures", "tables")
SEEDS, SNAPSHOT_SEED = range(5), 0


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--gnca",
        action="store_true",
        help="Write GNCA/E(n)-GNCA table only.",
    )
    args = parser.parse_known_args()[0]
    gnca_table = {}
    for method, label in (("GNCA", "GNCA"), ("EGNCA", r"E($n$)-GNCA")):
        try:
            gnca_table[label] = load_baseline_table(
                list(MAIN), {method: "standard_rollout"}
            )[method]
        except FileNotFoundError as e:
            print(f"skip {method}: {e}")
    if gnca_table:
        gnca_path = TABLE_DIR / "eval_metrics_gnca.txt"
        format_metrics_latex_table(
            gnca_table,
            highlight_best=False,
            caption=(
                r"Dynamic-graph adaptations of GNCA and E($n$)-GNCA: final "
                r"normalized MSE and GW distance on standard pattern-formation "
                r"rollouts."
            ),
            label="tab:pattern-gnca",
            out_path=gnca_path,
        )
        print(gnca_path.resolve())
    if args.gnca:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results, snap_rows = [], []
    seed_metrics: dict[str, list] = defaultdict(list)

    for name in MAIN:
        ckpt = checkpoint(name)
        for seed in SEEDS:
            snap = seed == SNAPSHOT_SEED
            out = run_eval(
                load_run_cfg(name),
                ckpt,
                video_path=None,
                return_positions="all" if snap else "last",
                autoreset=False,
                flat_overrides={
                    "train.seed": seed,
                    **UNBOUNDED_FLAT,
                },
            )
            metrics = {
                k: float(out["metrics"][k])
                for k in METRIC_KEYS
                if out["metrics"].get(k) is not None
            }
            seed_metrics[name].append(metrics)
            results.append((name, metrics))
            print(f"{name:12s} seed={seed} {metrics_line(metrics)} ckpt={ckpt}")

            if snap:
                traj = out["positions"][0]
                times = even_snap_times(traj.shape[0] - 1)
                snaps = {t: traj[t] for t in times}
                targets, *_ = load_eval_data(out["cfg"])
                snap_rows.append((name, targets, snaps, times))

    for fname, plot, rows in (
        ("eval_errors.png", plot_metric_bars, results),
        ("01-pattern-formation-snapshots.png", plot_snapshots, snap_rows),
    ):
        path = FIG_DIR / fname
        plt.close(plot(rows, path))
        print(path.resolve())

    table = {
        **load_npa_baselines(list(MAIN), "native"),
        "MAGNA (ours)": {ds: mean_std(mets) for ds, mets in seed_metrics.items()},
    }
    tex_path = TABLE_DIR / "eval_metrics.txt"
    format_metrics_latex_table(
        table,
        caption=r"Final normalized MSE and GW distance for standard (same as during training but unbounded) pattern-formation rollouts. MAGNA values report mean \(\pm\) standard deviation over five evaluation seeds. NPA is our retrain of the original method (GW only); NPA-Chamfer is an adapted Chamfer-trained variant. Lower is better.",
        label="tab:pattern",
        out_path=tex_path,
    )
    print(tex_path.resolve())


if __name__ == "__main__":
    main()
