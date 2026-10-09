"""Generate sunflower/gaussian/uniform init AnnData, then eval robustness to init."""

import sys

import matplotlib.pyplot as plt

from experiments.utils import (
    MAIN,
    UNBOUNDED_FLAT,
    checkpoint,
    even_snap_times,
    experiment_dirs,
    format_metrics_latex_table,
    generate_init_datasets,
    load_npa_baselines,
    load_run_cfg,
    metrics_line,
    nested_mean_std,
    plot_metric_bars,
    plot_snapshots,
)
from magna.eval import METRIC_KEYS, load_eval_data, run_eval

EXPERIMENT = "02-robustness-to-initialization"
OUT_DIR, FIG_DIR, VIDEO_DIR, TABLE_DIR = experiment_dirs(
    EXPERIMENT, "data", "figures", "videos", "tables"
)
INITS, SEEDS, SNAPSHOT_SEED = ("gaussian", "uniform"), range(5), 0


def run_inference(
    *,
    videos: bool = False,
    name_suffix: str = "",
    collect_snaps: bool = True,
):
    results, snap_rows = [], []
    seed_metrics: dict[tuple[str, str], list] = {}
    for name in MAIN:
        ckpt = checkpoint(name)
        for init in INITS:
            path = OUT_DIR / name / f"{init}{name_suffix}.h5ad"
            for seed in SEEDS:
                video = (
                    VIDEO_DIR / name / f"{name}_{init}{name_suffix}_seed{seed}.mp4"
                    if videos
                    else None
                )
                out = run_eval(
                    load_run_cfg(name),
                    ckpt,
                    video_path=video,
                    autoreset=False,
                    return_positions=(
                        "all" if collect_snaps and seed == SNAPSHOT_SEED else "last"
                    ),
                    flat_overrides={
                        "data.path": str(path),
                        "data.init_obsm_key": f"init_seed_{seed}",
                        **UNBOUNDED_FLAT,
                    },
                )
                metrics = {
                    k: float(out["metrics"][k])
                    for k in METRIC_KEYS
                    if out["metrics"].get(k) is not None
                }
                if collect_snaps and seed == SNAPSHOT_SEED:
                    traj = out["positions"][0]
                    times = even_snap_times(traj.shape[0] - 1)
                    snaps = {t: traj[t] for t in times}
                    targets, *_ = load_eval_data(out["cfg"])
                    snap_rows.append((name, targets, snaps, times))
                seed_metrics.setdefault((name, init), []).append(metrics)
                results.append((init, metrics))
                tag = "video" if video else "no-video"
                print(
                    f"{name:12s} {init + name_suffix:10s} seed={seed} "
                    f"{metrics_line(metrics)} [{tag}]"
                )
    return results, snap_rows, nested_mean_std(seed_metrics)


def main() -> None:
    videos = "--videos" in sys.argv
    generate_init_datasets(MAIN, OUT_DIR, INITS, SEEDS)
    generate_init_datasets(
        MAIN, OUT_DIR, INITS, SEEDS, bbox_scale=0.5, name_suffix="_half"
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results, snap_rows, table = run_inference(videos=videos)
    fig_path = FIG_DIR / "eval_errors.png"
    plt.close(plot_metric_bars(results, fig_path))
    print(fig_path.resolve())
    for name in MAIN:
        rows = [r for r in snap_rows if r[0] == name]
        path = FIG_DIR / f"02-robustness-snapshots-{name}.png"
        plt.close(plot_snapshots(rows, path))
        print(path.resolve())
    tex_path = TABLE_DIR / "eval_metrics.txt"
    format_metrics_latex_table(
        {
            **load_npa_baselines(list(MAIN), {init: f"init_{init}" for init in INITS}),
            **table,
        },
        variant_header="Init",
        variants=INITS,
        caption=r"Final normalized MSE and GW distance under unseen Gaussian and uniform initializations. Policies are evaluated without fine-tuning. MAGNA values report mean \(\pm\) standard deviation over five evaluation seeds. NPA is our retrain of the original method (GW only); NPA-Chamfer is an adapted Chamfer-trained variant. Lower is better.",
        label="tab:init",
        out_path=tex_path,
    )
    print(tex_path.resolve())
    _, _, table_half = run_inference(
        videos=videos, name_suffix="_half", collect_snaps=False
    )
    half_path = TABLE_DIR / "eval_metrics_half_bbox.txt"
    format_metrics_latex_table(
        table_half,
        variant_header="Init",
        variants=INITS,
        highlight_best=False,
        caption=r"Final normalized MSE and GW distance under Gaussian and uniform initializations drawn in a centered half-extent target bounding box. Policies are evaluated without fine-tuning. Values report mean \(\pm\) standard deviation over five evaluation seeds; lower is better.",
        label="tab:init-half",
        out_path=half_path,
    )
    print(half_path.resolve())


if __name__ == "__main__":
    main()
