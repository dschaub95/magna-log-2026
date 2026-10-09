"""Mid-rollout gaussian position noise; continue with preserved LSTM state."""

import argparse

import matplotlib.pyplot as plt

from experiments.utils import (
    MAIN,
    UNBOUNDED,
    checkpoint,
    experiment_dirs,
    format_metrics_latex_table,
    load_npa_baselines,
    load_run_cfg,
    metrics_line,
    nested_mean_std,
    plot_metric_bars,
    plot_snapshots,
    rollout,
)
from magna.eval import (
    DEFAULT_INFERENCE_STEPS,
    build_eval_env_and_policy,
    load_eval_data,
    load_policy_weights,
)
from magna.utils import resolve_cell_colors

EXPERIMENT = "03-regeneration-after-perturbation"
FIG_DIR, VIDEO_DIR, TABLE_DIR = experiment_dirs(
    EXPERIMENT, "figures", "videos", "tables"
)
MODES, SEEDS, SNAPSHOT_SEED = ("local", "global"), range(5), 0


def run_inference(*, videos=False):
    results, snap_rows = [], []
    seed_metrics: dict[tuple[str, str], list] = {}
    for name in MAIN:
        cfg = load_run_cfg(name, resolve=True)
        targets, init_pos, _, _, colors, id_feats, init_head = load_eval_data(cfg)
        n_half = (
            int(cfg.env.time_horizon)
            if cfg.env.time_horizon > 0
            else DEFAULT_INFERENCE_STEPS
        )
        probe, policy, base_seed, env_kwargs = build_eval_env_and_policy(
            cfg,
            targets=targets,
            initial_positions=init_pos,
            initial_headings=init_head,
            loaded_id_features=id_feats,
            time_horizon=-1,
            **UNBOUNDED,
            reuse_initial_positions=init_pos is not None or init_head is not None,
            cell_colors=resolve_cell_colors(targets, colors),
        )
        probe.close()
        load_policy_weights(policy, checkpoint(name), device=cfg.train.device)
        for mode in MODES:
            for seed in SEEDS:
                video = (
                    VIDEO_DIR / name / f"{name}_{mode}_seed{seed}.mp4"
                    if videos
                    else None
                )
                metrics, snap, times = rollout(
                    env_kwargs,
                    policy,
                    cfg.train.device,
                    int(base_seed + seed),
                    targets,
                    2 * n_half,
                    (mode, seed),
                    video,
                )
                if seed == SNAPSHOT_SEED:
                    snap_rows.append((name, targets, snap, times))
                seed_metrics.setdefault((name, mode), []).append(metrics)
                results.append((mode, metrics))
                tag = "video" if video else "no-video"
                print(
                    f"{name:12s} {mode:8s} seed={seed} {metrics_line(metrics)} [{tag}]"
                )
    return results, snap_rows, nested_mean_std(seed_metrics)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_known_args()[0]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results, snap_rows, table = run_inference(videos=args.videos)
    path = FIG_DIR / "eval_errors.png"
    plt.close(plot_metric_bars(results, path))
    print(path.resolve())
    for name in MAIN:
        rows = [r for r in snap_rows if r[0] == name]
        path = FIG_DIR / f"03-regeneration-snapshots-{name}.png"
        plt.close(plot_snapshots(rows, path))
        print(path.resolve())
    tex_path = TABLE_DIR / "regen_metrics.txt"
    format_metrics_latex_table(
        {
            **load_npa_baselines(
                list(MAIN),
                {mode: f"perturb_{mode}_trainmatch_n64" for mode in MODES},
            ),
            **table,
        },
        variant_header="Noise",
        variants=MODES,
        caption=r"Final normalized MSE and GW distance after local or global perturbations. Policies are evaluated without fine-tuning. MAGNA values report mean \(\pm\) standard deviation over five evaluation seeds. NPA/NPA-Chamfer use their training-matched horizon ($N_{\mathrm{half}}=64$); MAGNA uses its native regeneration horizon. NPA is our retrain of the original method (GW only); NPA-Chamfer is an adapted Chamfer-trained variant. Lower is better.",
        label="tab:regen",
        out_path=tex_path,
    )
    print(tex_path.resolve())


if __name__ == "__main__":
    main()
