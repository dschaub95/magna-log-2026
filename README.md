# Learning Graph Neural Cellular Automata via Multi-Agent Reinforcement Learning

Code, configs, data, and trained weights for MAGNA (Multi-Agent Graph Neural Cellular Automata): a shared decentralized policy with which moving agents, each using only local observations and neighbor messages on a dynamic geometric graph, self-organize into a target structure. The Python package is `magna`.

## Installation

Requires Linux, Python 3.11+, a CUDA GPU, and [uv](https://github.com/astral-sh/uv).

```bash
bash setup.sh   # uv env (all extras), pre-commit hook, raylib/qhull/xvfb
```

The native environment binding is built automatically on the first `uv run train` (or explicitly with `uv run build`).

## Repository layout

```text
configs/<target>.yaml        # training configs of the paper models
configs/sweeps/<target>.yaml # W&B hyperparameter sweeps (appendix "Hyperparameter Search")
checkpoints/<target>.pt      # trained policies used for all paper results
data/pattern_formation/      # synthetic targets (Grid2d, Torus, Cube, Stanford bunny subsamples)
data/real/                   # kidney tissue target (see Data)
data/baselines/              # GNCA / E(n)-GNCA / NPA baseline results
experiments/                 # one script per paper experiment
src/magna/                   # environment (C binding), policy, training, evaluation
outputs/                     # generated runs, figures, tables (not tracked)
```

## Reproducing the paper

Every experiment evaluates the shipped checkpoints by default:

```bash
uv run experiment --list
uv run experiment 1 2 3          # main-text results
uv run experiment                # everything
uv run experiment 2 3 --videos   # also render rollout videos
```

| Script | Paper |
|---|---|
| `01-pattern-formation` | Fig. pattern formation, Tab. pattern formation, App. Tab. GNCA / E(n)-GNCA |
| `02-robustness-to-initialization` | Tab. unseen initializations, App. half-extent table and snapshots |
| `03-regeneration-after-perturbation` | Fig. + Tab. regeneration, App. snapshots |
| `04-compute-scaling` | App. compute scaling (probes peak GPU memory with short training runs) |
| `05-long-horizon` | App. long-horizon persistence |
| `06-scalability` | App. Bunny agent-count scalability |
| `07-kidney-tasks` | App. tissue-derived kidney target |

Figures and tables are written to `outputs/{figures,tables,results}/<experiment>/`.

All paper results were produced on one NVIDIA A100-SXM4 GPU (40 GB), 16 cores of an AMD EPYC 7742, and 128 GB RAM, with Python 3.13, PyTorch 2.11 and CUDA 12.8 (`uv.lock`). Rollouts are seeded, but GPU kernels and floating-point reductions are not bit-identical across GPU models, drivers, and library versions, so numbers obtained on other hardware may differ from the paper, and retrained models more so.

## Training your own models

```bash
uv run train --cfg=configs/grid2d.yaml
uv run train --cfg=configs/grid2d.yaml --train.total_timesteps=1000000 --wandb_mode=disabled  # smoke test (too short for a checkpoint)
```

Any config field can be overridden with dot notation (e.g. `--train.learning_rate=0.01`). Logging goes to Weights & Biases (`project: magna`) unless `--wandb_mode=disabled`. Each run writes its best policy (by training error, tracked once the first episodes finish) to `outputs/runs/<target>/<run-id>/model_best.pt` (one `<run-id>_<i>` directory per repeat with `--train.n_repeat>1`); runs never overwrite each other.

Point the experiments at your runs instead of the shipped checkpoints:

```bash
uv run experiment 1 2 3 --runs outputs/runs                                 # newest run (or repeat) per target
uv run experiment 1 --ckpt grid2d=outputs/runs/grid2d/<run-id>/model_best.pt  # pin a specific run
```

Evaluate a single policy:

```bash
uv run eval --cfg=configs/grid2d.yaml                                       # uses checkpoints/grid2d.pt
uv run eval --cfg=configs/grid2d.yaml --checkpoint=path/to/model_best.pt --n_rollouts=3 --greedy
```

## Data

Synthetic targets are stored as coordinate tensors (`target_coord.pt`); AnnData (`.h5ad`) targets are supported as well. The kidney target (`data/real/`) is a 512-cell region of a Xenium spatial profile of a human kidney biopsy from

> C. Ly, D. P. Schaub, R. Khatri, Z. Sultana, A. Boxnick, Z. Song, A. Hube, T. Huber, T. Wiech, E. Tolosa, U. Panzer, S. Bonn, C. Krebs, I. Prinz. *A spatial in situ hybridization approach to T cell clonotype analysis using T cell receptor variable gene probes.* Cell Reports 45(6), 117483 (2026). [doi:10.1016/j.celrep.2026.117483](https://doi.org/10.1016/j.celrep.2026.117483)

Cell-type annotations in that file are used only for visualization.

## Citation

TBD

## License

MIT, see [LICENSE](LICENSE).
