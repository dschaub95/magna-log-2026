# MAGNA

## Overview

MAGNA (Multi-Agent Graph Neural Cellular Automata) trains a shared decentralized policy so moving agents, each using only local observations and neighbor messages on a dynamic geometric graph, self-organize into a target structure.

## Agent instructions

- For Python commands, always prefix with `UV_CACHE_DIR=/tmp/uv-cache`. Check if the workspace is on NFS with `findmnt -T . -o FSTYPE -n`; if the result is `nfs` or `nfs4`, set `VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT:-${UV_BASE:-$HOME/.uv-local}/venvs/magna-log-2026}` and `UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV`, use `uv run --active` (and the same env for `uv sync` / `uv lock`), and dont create a project-local `.venv`. Otherwise use plain `uv run`.
- After substantive code changes, run `pytest`, `ruff format .`, and `ruff check . --fix` with that same Python prefix before finishing. Running tests requires network access.
- keep `__init__.py` files empty, if not explicitly asked to add something.
- keep the number of lines of code (taking auto-formatting into account) as minimal as possible while still retaining maintainability and readability.
- generally do not try to preserve backwards compatibility, unless explicitly asked to do so.
- we dont need extensive validation inside each function, one or maximal two different error cases are enough if any.
- only write tests that are actually useful

## Project structure

- `src/magna/`: Core implementation.
  - `env/`: Simulation environment and native C/C++ binding.
  - `data/`: Dataset loading, preprocessing, lattice construction, and initial positions.
  - `cli/`: Implementations of the commands declared in `pyproject.toml`.
  - `config.py`, `train.py`, `eval.py`, and `policy.py`: Main configuration, training, evaluation, and model logic.
- `experiments/`: Paper experiment scripts. Numbered scripts correspond to the paper experiments; shared plotting and result utilities live in `experiments/utils.py`. Run them through `uv run experiment`.
- `configs/`: Per-target training configurations (`<target>.yaml`) used by the experiments; `sweeps/` holds the sweep configurations.
- `checkpoints/`: Trained policies (`<target>.pt`) behind the paper results.
- `tests/`: Tests for the core package and command-line interfaces.
- `data/`: Pattern-formation targets (`pattern_formation/`), the kidney target (`real/`), and paper baselines (`baselines/`).
- `outputs/`: Generated training runs (`runs/<target>/<run-id>/`), figures, videos, and tables; do not treat these as source files.
