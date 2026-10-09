"""Evaluate the kidney policy on pattern formation, initialization, and repair."""

import matplotlib.pyplot as plt

from experiments.utils import (
    UNBOUNDED,
    checkpoint,
    experiment_dirs,
    format_metrics_latex_table,
    init_in_bbox,
    load_run_cfg,
    mean_std,
    metrics_line,
    plot_snapshots,
    rollout,
)
from magna.eval import (
    build_eval_env_and_policy,
    load_eval_data,
    load_policy_weights,
)
from magna.utils import resolve_cell_colors

EXPERIMENT = "07-kidney-tasks"
FIG_DIR, TABLE_DIR = experiment_dirs(EXPERIMENT, "figures", "tables")
TASKS, SEEDS = ("Pattern formation", "Robustness", "Regeneration"), range(5)


def main():
    cfg = load_run_cfg("Kidney512", resolve=True)
    targets, init_pos, _, _, labels, id_feats, init_head = load_eval_data(cfg)
    colors = resolve_cell_colors(targets, labels)
    probe, policy, base_seed, standard_kwargs = build_eval_env_and_policy(
        cfg,
        targets,
        initial_positions=init_pos,
        initial_headings=init_head,
        loaded_id_features=id_feats,
        time_horizon=-1,
        **UNBOUNDED,
    )
    probe.close()
    load_policy_weights(policy, checkpoint("Kidney512"), cfg.train.device)

    horizon = int(cfg.env.time_horizon)
    metrics = {task: [] for task in TASKS}
    trajectory_rows = []
    for seed in SEEDS:
        robust_kwargs = {
            **standard_kwargs,
            "initial_positions": init_in_bbox("gaussian", targets, seed),
        }
        specs = (
            (standard_kwargs, seed, horizon, None),
            (robust_kwargs, base_seed, horizon, None),
            (standard_kwargs, base_seed + seed, 2 * horizon, ("local", seed)),
        )
        for task, (env_kwargs, run_seed, n_steps, perturb) in zip(
            TASKS, specs, strict=True
        ):
            task_metrics, snapshots, times = rollout(
                env_kwargs,
                policy,
                cfg.train.device,
                run_seed,
                targets,
                n_steps,
                perturb,
            )
            metrics[task].append(task_metrics)
            print(f"{task:18s} seed={seed} {metrics_line(task_metrics)}")
            if seed == 0:
                trajectory_rows.append((task, targets, snapshots, times))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIG_DIR / "07-kidney-tasks-snapshots.png"
    plt.close(
        plot_snapshots(
            trajectory_rows,
            figure_path,
            colors=colors,
            color_labels=labels,
            row_labels=True,
        )
    )
    print(figure_path.resolve())

    table_path = TABLE_DIR / "kidney_task_metrics.txt"
    format_metrics_latex_table(
        {"MAGNA (ours)": {task: mean_std(metrics[task]) for task in TASKS}},
        highlight_best=False,
        caption=(
            r"Final normalized MSE and GW distance on the Kidney512 pattern-formation, "
            r"Gaussian-initialization, and local-regeneration tasks. Values report mean "
            r"$\pm$ standard deviation over five evaluation seeds; lower is better."
        ),
        label="tab:kidney-tasks",
        out_path=table_path,
    )
    print(table_path.resolve())


if __name__ == "__main__":
    main()
