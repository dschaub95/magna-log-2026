"""Shared experiment helpers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pufferlib
import seaborn as sns
import torch
from matplotlib.patches import Patch
from scipy import sparse
from tqdm.auto import tqdm

from magna.data.constants import DEFAULT_OBSM_KEY
from magna.data.io import load_pt_coordinates
from magna.env import MultiCellEnv
from magna.eval import METRIC_KEYS, _downsample_frame_max_side, formation_metrics
from magna.utils import seed_everything

BASELINE_CSV_ROOT = Path("data/baselines/split_csv")
UNBOUNDED = {"bounded": False, "boundary_penalty": 0.0}
UNBOUNDED_FLAT = {f"env.{k}": v for k, v in UNBOUNDED.items()}
TEXTWIDTH_IN = 5.5  # documents/magna-paper/log_2026.sty
_CSV_DATASET = {"Bunny512": "Bunny-512", "Bunny1024": "Bunny-1024"}
_CSV_METRIC = {"mse": "mse_matched", "gw": "gw"}
MAIN = ("Grid2d", "Torus", "Cube", "Bunny512")
_CFG_DIR, CHECKPOINT_DIR = Path("configs"), Path("checkpoints")
RUN_CFGS = {
    name: _CFG_DIR / f"{name.lower()}.yaml"
    for name in (*MAIN, "Bunny1024", "Bunny1500", "Bunny2000", "Kidney512")
}


def checkpoint(name: str) -> Path:
    """Shipped weights, or the newest run under ``--runs DIR`` / a ``--ckpt KEY=PATH`` pin."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--ckpt", action="append", default=[])
    args = parser.parse_known_args()[0]
    key = RUN_CFGS[name].stem
    if pinned := dict(pin.split("=", 1) for pin in args.ckpt).get(key):
        return Path(pinned)
    if args.runs is None:
        return CHECKPOINT_DIR / f"{key}.pt"
    if not (runs := list((args.runs / key).glob("*/model_best.pt"))):
        raise FileNotFoundError(f"No */model_best.pt under {args.runs / key}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def stale_checkpoints(rows: list[dict[str, str]], names) -> bool:
    """Whether cached ``rows`` were not evaluated with the currently selected checkpoints."""
    return {r.get("checkpoint") for r in rows} != {str(checkpoint(n)) for n in names}


def load_run_cfg(name: str, *, resolve: bool = False):
    from magna.config import load_yaml_config

    return load_yaml_config(str(RUN_CFGS[name]), resolve=resolve)


def _parse_pm(s: str) -> tuple[float, float | None] | None:
    s = (s or "").strip()
    if not s:
        return None
    if "±" in s:
        mean, std = s.split("±", 1)
        return float(mean), float(std)
    return float(s), None


def _metrics_from_row(
    row: dict[str, str], metrics: tuple[str, ...]
) -> dict[str, tuple[float, float | None]]:
    out = {
        met: parsed
        for met in metrics
        if (parsed := _parse_pm(row.get(_CSV_METRIC.get(met, met), "")))
    }
    if missing := [met for met in metrics if met not in out]:
        raise ValueError(f"missing baseline metrics: {missing}")
    return out


def load_baseline_table(
    datasets: list[str],
    condition_by_method: dict[str, str | dict[str, str]],
    *,
    root: Path = BASELINE_CSV_ROOT,
    metrics: tuple[str, ...] = ("mse", "gw"),
) -> dict:
    out: dict[str, dict] = {m: {} for m in condition_by_method}
    for ds in datasets:
        csv_ds = _CSV_DATASET.get(ds, ds)
        for method, conds in condition_by_method.items():
            path = root / csv_ds / f"{method}.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing baseline CSV: {path}")
            with path.open(newline="", encoding="utf-8") as f:
                rows = {r["condition"]: r for r in csv.DictReader(f)}
            pairs = [(None, conds)] if isinstance(conds, str) else conds.items()
            for var, cond in pairs:
                if not (row := rows.get(cond)):
                    raise ValueError(f"missing condition {cond!r} in {path}")
                block = _metrics_from_row(row, metrics)
                if var is None:
                    out[method][ds] = block
                else:
                    out[method].setdefault(ds, {})[var] = block
    return out


def load_npa_baselines(
    datasets: list[str],
    conditions: str | dict[str, str],
    **kwargs,
) -> dict:
    """Load NPA (retrain of original, GW only) and adapted NPA-Chamfer baselines."""
    return {
        "NPA": load_baseline_table(
            datasets, {"NPA-full": conditions}, metrics=("gw",), **kwargs
        )["NPA-full"],
        "NPA-Chamfer": load_baseline_table(
            datasets, {"NPA-Chamfer": conditions}, **kwargs
        )["NPA-Chamfer"],
    }


def experiment_dirs(name: str, *kinds: str) -> tuple[Path, ...]:
    roots = {
        "data": "data",
        "figures": "outputs/figures",
        "videos": "outputs/videos",
        "tables": "outputs/tables",
        "results": "outputs/results",
    }
    return tuple(Path(roots[k]) / name for k in kinds)


def metrics_line(metrics: dict) -> str:
    return " ".join(
        f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS if metrics.get(k) is not None
    )


def nested_mean_std(seed_metrics: dict[tuple[str, str], list]) -> dict:
    out: dict = {}
    for (ds, var), mets in seed_metrics.items():
        out.setdefault(ds, {})[var] = mean_std(mets)
    return {"MAGNA (ours)": out}


def paper_style() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}\renewcommand{\rmdefault}{ptm}"
            ),
        }
    )


def save_fig(fig, out_path: Path, *, bbox_inches: str | None = "tight"):
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    kw = {} if bbox_inches is None else {"bbox_inches": bbox_inches}
    fig.savefig(out_path, dpi=300, **kw)
    fig.savefig(out_path.with_suffix(".pdf"), **kw)
    return fig


def even_snap_times(n_steps: int, n: int = 5) -> tuple[int, ...]:
    """N equally spaced timesteps in [0, n_steps]."""
    return tuple(int(round(i * n_steps / (n - 1))) for i in range(n))


def plot_snapshots(
    rows, out_path: Path, *, colors=None, color_labels=None, row_labels=False
):
    """rows: (label, targets, snaps_dict, times). 3D positions use Axes3D."""
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype == np.uint8:
            colors = colors.astype(float) / 255
    paper_style()
    n_rows, n_cols = len(rows), len(rows[0][3])
    fig = plt.figure(
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * n_rows / n_cols + 0.18 * n_rows),
        layout="constrained",
    )
    point_color = sns.color_palette("deep")[0] if colors is None else colors
    for r, (name, targets, snaps, times) in enumerate(rows):
        pts = np.concatenate([snaps[t] for t in times])
        dim = pts.shape[1]
        lo, hi = pts.min(0), pts.max(0)
        side = 1.1 * float(np.max(np.maximum(hi - lo, 1e-6)))
        mid = 0.5 * (lo + hi)
        lims = [(mid[i] - 0.5 * side, mid[i] + 0.5 * side) for i in range(dim)]
        for c, t in enumerate(times):
            pos = snaps[t]
            kw = {"projection": "3d"} if dim == 3 else {}
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, **kw)
            if dim == 3:
                ax.scatter(*pos.T, s=3, color=point_color, linewidths=0)
                ax.set(xlim=lims[0], ylim=lims[1], zlim=lims[2])
                ax.set_box_aspect((1, 1, 1))
                ax.set_axis_off()
            else:
                ax.scatter(pos[:, 0], pos[:, 1], s=3, color=point_color, linewidths=0)
                ax.set(xlim=lims[0], ylim=lims[1], aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                sns.despine(ax=ax, left=True, bottom=True)
            m = formation_metrics(targets, pos)
            key = "mse" if "mse" in m else "gw"
            ax.set_title(rf"$t$ = {t}, {key.upper()} = {m[key]:.2e}", fontsize=6)
            if row_labels and c == 0:
                ax.set_ylabel(name)
    if color_labels is not None:
        color_map = dict(zip(map(str, color_labels), colors, strict=True))
        fig.legend(
            handles=[
                Patch(color=color_map[label], label=label)
                for label in sorted(color_map)
            ],
            loc="outside lower center",
            ncols=(len(color_map) + 1) // 2,
            mode="expand",
            frameon=False,
            fontsize=6,
            handlelength=0.8,
        )
    return save_fig(fig, out_path, bbox_inches=None)


def plot_metric_bars(results: list[tuple], out_path: Path | None = None):
    """Bar plot of METRIC_KEYS; results are (group, metrics_dict, *_)."""
    groups = list(dict.fromkeys(g for g, *_ in results))
    paper_style()
    n, m = len(groups), len(METRIC_KEYS)
    fig, axes = plt.subplots(1, m, figsize=(0.55 * n * m + 1.0, 2.6), squeeze=False)
    x, color = np.arange(n), sns.color_palette("deep")[0]
    for ax, key in zip(axes[0], METRIC_KEYS, strict=True):
        vals = [
            [
                metrics[key]
                for g, metrics, *_ in results
                if g == d and metrics.get(key) is not None
            ]
            for d in groups
        ]
        ax.bar(
            x,
            [float(np.mean(v)) if v else 0.0 for v in vals],
            yerr=[float(np.std(v)) if v else 0.0 for v in vals],
            width=0.6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
        )
        ax.set(ylabel=key, ylim=(0, None), xlim=(-0.75, n - 0.25))
        ax.set_xticks(x, groups)
        sns.despine(ax=ax)
    if out_path is not None:
        return save_fig(fig, out_path)
    fig.tight_layout()
    return fig


def plot_scalability(
    mse_points: list[tuple[int, float, float]],
    return_curves: list[tuple[int, np.ndarray, np.ndarray]] | None = None,
    out_path: Path | None = None,
):
    """MSE vs agents; optional right panel of episode return vs relative steps."""
    paper_style()
    ns, means, stds = map(np.asarray, zip(*mse_points, strict=True))
    colors = sns.color_palette("deep")
    if return_curves:
        fig, axes = plt.subplots(
            1, 2, figsize=(TEXTWIDTH_IN, 2.2), layout="constrained"
        )
        ax = axes[0]
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(TEXTWIDTH_IN, 2.2),
            width_ratios=[1, 2.4, 1],
            layout="constrained",
        )
        axes[0].set_axis_off()
        axes[2].set_axis_off()
        ax = axes[1]
    ax.errorbar(
        ns,
        means,
        yerr=stds,
        fmt="-o",
        color=colors[0],
        ecolor="black",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        markersize=3.5,
        linewidth=0.9,
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax.set(xlabel="Number of agents", ylabel="MSE", yscale="log")
    ax.set_xticks(ns, [str(int(n)) for n in ns])
    e0 = np.floor(np.log10(ax.get_ylim()[0]))
    e1 = np.ceil(np.log10(ax.get_ylim()[1]))
    ax.set_yticks(10.0 ** np.arange(e0, e1 + 0.5))
    ax.set_yticks(10.0 ** np.arange(e0, e1 + 0.05, 0.1), minor=True)
    ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="y", which="major", length=3.5)
    ax.tick_params(axis="y", which="minor", length=1.5)
    ax.grid(True, axis="y", which="major", ls=":", lw=0.5, alpha=0.6)
    ax.grid(True, axis="y", which="minor", ls=":", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    if return_curves:
        for i, (n, x, y) in enumerate(return_curves):
            axes[1].plot(
                x, y, "-", color=colors[i % len(colors)], label=str(n), linewidth=0.9
            )
        axes[1].set(xlabel="Relative steps", ylabel="Episode return", xlim=(0.0, 1.0))
        axes[1].legend(frameon=False, title=r"$N$")
        axes[1].yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        axes[1].set_axisbelow(True)
        sns.despine(ax=axes[1])
    return save_fig(fig, out_path, bbox_inches=None) if out_path is not None else fig


def fit_to_bbox(
    pts: np.ndarray, targets: np.ndarray, *, bbox_scale: float = 1.0
) -> np.ndarray:
    lo, hi = targets.min(0), targets.max(0)
    mid = 0.5 * (lo + hi)
    span = np.maximum(float(bbox_scale) * (hi - lo), 1e-6)
    extent = np.maximum(pts.max(0) - pts.min(0), 1e-6)
    scale = float(np.min(span / extent))
    return (mid + (pts - 0.5 * (pts.min(0) + pts.max(0))) * scale).astype(np.float32)


def init_in_bbox(
    name: str, targets: np.ndarray, seed: int, *, bbox_scale: float = 1.0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "gaussian":
        return fit_to_bbox(
            rng.normal(0.0, 1.0, targets.shape).astype(np.float32),
            targets,
            bbox_scale=bbox_scale,
        )
    if name == "uniform":
        lo, hi = targets.min(0), targets.max(0)
        if bbox_scale != 1.0:
            mid = 0.5 * (lo + hi)
            half = 0.5 * float(bbox_scale) * (hi - lo)
            lo, hi = mid - half, mid + half
        return rng.uniform(lo, hi, targets.shape).astype(np.float32)
    if name != "sunflower":
        raise ValueError(f"unknown init distribution: {name!r}")
    n, dim = targets.shape
    i = np.arange(n, dtype=np.float32)
    theta = i * (np.pi * (3.0 - np.sqrt(5.0)))
    if dim == 3:
        z = 1.0 - 2.0 * (i + 0.5) / n
        xy, r = (
            np.sqrt(np.maximum(0.0, 1.0 - z * z)),
            np.cbrt(np.mod((i + 0.5) * np.sqrt(2.0), 1.0)),
        )
        pts = np.column_stack([r * xy * np.cos(theta), r * xy * np.sin(theta), r * z])
    else:
        r = np.sqrt(i / n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return fit_to_bbox(pts, targets, bbox_scale=bbox_scale)[rng.permutation(n)]


def perturb_positions(pos, mode, targets, seed, local_frac=0.25):
    if mode == "global":
        return init_in_bbox("gaussian", targets, seed)
    rng = np.random.default_rng(seed)
    out = np.asarray(pos, dtype=np.float32).copy()
    center = int(rng.integers(len(out)))
    count = max(2, round(local_frac * len(out)))
    distance2 = ((out - out[center]) ** 2).sum(1)
    neighbors = np.argsort(distance2)[:count]
    scale = max(0.5 * float(np.sqrt(distance2[neighbors[1]])), 1e-6)
    out[neighbors] = out[center] + rng.normal(0, scale, (count, out.shape[1]))
    return out.astype(np.float32)


def rollout(
    env_kwargs, policy, device, seed, targets, n_steps, perturb=None, video=None
):
    """Policy rollout; perturb=(mode, noise_seed) is applied at n_steps // 2."""
    seed_everything(seed)
    env = MultiCellEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    state, frames, snaps = {"lstm_h": None, "lstm_c": None}, [], {}
    times = even_snap_times(n_steps)
    try:
        for t in tqdm(range(n_steps), leave=False):
            if perturb is not None and t == n_steps // 2:
                pos = perturb_positions(
                    env.positions()[0], perturb[0], targets, perturb[1]
                )
                obs = env.set_positions(pos)
            if t in times:
                snaps[t] = env.positions()[0].copy()
            with torch.no_grad():
                logits, _ = policy.forward_eval(
                    torch.as_tensor(obs, device=device), state
                )
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
            obs, *_ = env.step(action.cpu().numpy())
            if video is not None:
                frame = np.clip(env.capture(), 0, 255).astype(np.uint8)
                frames.append(_downsample_frame_max_side(frame, 1024))
        snaps[n_steps] = env.positions()[0].copy()
    finally:
        env.close()
    if frames:
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=30, codec="libx264", macro_block_size=4)
    metrics = formation_metrics(targets, snaps[n_steps])
    return {k: float(v) for k, v in metrics.items()}, snaps, times


def generate_init_datasets(
    names: tuple[str, ...],
    out_dir: Path,
    inits: tuple[str, ...] | list[str],
    seeds,
    *,
    bbox_scale: float = 1.0,
    name_suffix: str = "",
) -> None:
    for name in names:
        targets = np.asarray(
            load_pt_coordinates(load_run_cfg(name).data.path).float().numpy(),
            dtype=np.float32,
        )
        for init in inits:
            adata = ad.AnnData(X=sparse.csc_matrix((len(targets), 0), dtype=np.float32))
            adata.obsm[DEFAULT_OBSM_KEY] = targets.copy()
            for seed in seeds:
                adata.obsm[f"init_seed_{seed}"] = init_in_bbox(
                    init, targets, seed, bbox_scale=bbox_scale
                )
            out = out_dir / name / f"{init}{name_suffix}.h5ad"
            out.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(out)
            print(out.resolve())


def mean_std(
    metrics_list: list[dict],
    keys: tuple[str, ...] = ("mse", "gw"),
) -> dict[str, tuple[float, float]]:
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _fmt_num(x: float) -> str:
    return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.4f}"


def _cell(mean: float, std: float | None, *, best: bool) -> str:
    m = _fmt_num(mean)
    if std is None:
        return rf"$\mathbf{{{m}}}$" if best else rf"${m}$"
    pm = rf"{{\scriptstyle\pm {_fmt_num(std)}}}"
    return rf"$\mathbf{{{m}}} {pm}$" if best else rf"${m} {pm}$"


def format_metrics_latex_table(
    results: dict,
    *,
    metrics: tuple[str, ...] = ("mse", "gw"),
    variant_header: str | None = None,
    variants: tuple[str, ...] | None = None,
    highlight_best: bool = True,
    caption: str = "",
    label: str = "tab:metrics",
    note: str = "",
    width: float | None = TEXTWIDTH_IN,
    out_path: str | Path | None = None,
) -> str:
    """Booktabs table from nested mean±std results.

    No variant: results[method][dataset][metric] = (mean, std)
    With variant: results[method][dataset][variant][metric] = (mean, std)
    """
    methods = list(results)
    datasets: list[str] = []
    for method in methods:
        for ds in results[method]:
            if ds not in datasets:
                datasets.append(ds)

    has_var = variant_header is not None
    if has_var and not variants:
        raise ValueError("variants required when variant_header is set")

    best: dict[tuple, float] = {}
    if highlight_best:
        for ds in datasets:
            for met in metrics:
                if has_var:
                    for var in variants:
                        vals = [
                            results[m][ds][var][met][0]
                            for m in methods
                            if ds in results[m]
                            and var in results[m][ds]
                            and met in results[m][ds][var]
                        ]
                        if vals:
                            best[(ds, met, var)] = min(vals)
                else:
                    vals = [
                        results[m][ds][met][0]
                        for m in methods
                        if ds in results[m] and met in results[m][ds]
                    ]
                    if vals:
                        best[(ds, met)] = min(vals)

    n_met = len(metrics)
    n_lead = 2 if has_var else 1
    col_spec = ("l" * n_lead) + "c" * (n_met * len(datasets))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
    ]
    # Shrink only when natural width exceeds `width` (never scale up).
    if width is not None:
        lines += [
            r"\makeatletter",
            r"\@ifundefined{magna@tbl}{\newsavebox\magna@tbl}{}",
            r"\begin{lrbox}{\magna@tbl}",
        ]
    lines += [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    lead_pad = "& " if has_var else ""
    ds_heads = " & ".join(rf"\multicolumn{{{n_met}}}{{c}}{{{ds}}}" for ds in datasets)
    lines.append(f"& {lead_pad}{ds_heads} \\\\")
    cmid, col = [], n_lead + 1
    for _ in datasets:
        cmid.append(rf"\cmidrule(lr){{{col}-{col + n_met - 1}}}")
        col += n_met
    lines.append("".join(cmid))
    met_heads = " & ".join(
        [rf"{m.upper()} $\downarrow$" for _ in datasets for m in metrics]
    )
    if has_var:
        lines.append(rf"Method & {variant_header} & {met_heads} \\")
    else:
        lines.append(rf"Method & {met_heads} \\")
    lines.append(r"\midrule")

    for mi, method in enumerate(methods):
        if has_var:
            present = [
                var
                for var in variants
                if any(var in results[method].get(ds, {}) for ds in datasets)
            ]
            row_vars = present
        else:
            row_vars = [None]

        for oi, var in enumerate(row_vars):
            cells = []
            for ds in datasets:
                block = results[method].get(ds, {})
                if has_var:
                    block = block.get(var) if block else None
                for met in metrics:
                    if block is None or met not in block:
                        cells.append("---")
                        continue
                    mean, std = block[met]
                    key = (ds, met, var) if has_var else (ds, met)
                    cells.append(
                        _cell(
                            mean,
                            std,
                            best=highlight_best and mean == best.get(key),
                        )
                    )
            method_cell = (
                rf"\multirow{{{len(row_vars)}}}{{*}}{{{method}}}"
                if has_var and oi == 0
                else (method if not has_var else "")
            )
            if has_var:
                lines.append(
                    f"{method_cell} & {var.capitalize()} & "
                    + " & ".join(cells)
                    + r" \\"
                )
            else:
                lines.append(f"{method_cell} & " + " & ".join(cells) + r" \\")
        if mi < len(methods) - 1:
            lines.append(r"\midrule")

    n_cols = n_lead + n_met * len(datasets)
    lines.append(r"\bottomrule")
    if note:
        lines.append(rf"\multicolumn{{{n_cols}}}{{@{{}}l@{{}}}}{{\tiny {note}}} \\")
    lines.append(r"\end{tabular}")
    if width is not None:
        lines += [
            r"\end{lrbox}",
            rf"\ifdim\wd\magna@tbl>{width}in",
            rf"  \resizebox{{{width}in}}{{!}}{{\usebox\magna@tbl}}%",
            r"\else",
            r"  \usebox\magna@tbl",
            r"\fi",
            r"\makeatother",
        ]
    lines += [r"\end{table}", ""]
    tex = "\n".join(lines)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex)
    return tex
