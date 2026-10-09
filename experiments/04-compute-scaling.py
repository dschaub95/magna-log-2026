"""A100 wall time and peak GPU memory vs node count."""

import atexit
import csv
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.utils import TEXTWIDTH_IN, experiment_dirs, paper_style, save_fig

FIG_DIR, RESULT_DIR = experiment_dirs("04-compute-scaling", "figures", "results")
MEM_CSV = RESULT_DIR / "peak_gpu_memory.csv"
BASELINE_CSV = Path("data/baselines/compute_scaling/progress.csv")
_CFG = Path("configs")
RUN_CFGS = {
    256: _CFG / "n256.yaml",
    512: _CFG / "n512.yaml",
    1024: _CFG / "bunny1024.yaml",
    1500: _CFG / "bunny1500.yaml",
    2000: _CFG / "bunny2000.yaml",
}
# Wall hours from reference training runs (offline; no W&B needed).
RUNTIME_HOURS = {
    256: 0.18534682264194446,
    512: 0.2692921657136111,
    1024: 0.7549498089827779,
    1500: 1.0996902394730557,
    2000: 1.732937294437222,
}
# (csv method key, legend label); uncomment egnca to include E($n$)-GNCA
BASELINES = (("npa_chamfer", "NPA-Chamfer"),)  # ("egnca_dynamic", r"E($n$)-GNCA"),
FORCE_PROBE = False
_KW = {
    "markersize": 3.5,
    "linewidth": 0.9,
    "markeredgecolor": "black",
    "markeredgewidth": 0.5,
}
_PROBE: subprocess.Popen | None = None
_SMI = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_gpu_memory",
    "--format=csv,noheader,nounits",
]


def _kill_probe() -> None:
    global _PROBE
    if _PROBE is not None and _PROBE.poll() is None:
        try:
            os.killpg(_PROBE.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _PROBE = None


def _pdeathsig() -> None:
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(
        1, signal.SIGKILL
    )  # PR_SET_PDEATHSIG
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def _on_signal(signum, _frame) -> None:
    _kill_probe()
    raise SystemExit(128 + signum)


atexit.register(_kill_probe)
for _s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(_s, _on_signal)


def _baseline_hours():
    rows = list(csv.DictReader(BASELINE_CSV.open()))
    return [
        (
            lab,
            sorted(
                (int(r["n"]), float(r["elapsed_hours"]))
                for r in rows
                if r["status"] == "finished" and r["method"] == key
            ),
        )
        for key, lab in BASELINES
    ]


def _probe(n: int, cfg: Path) -> float:
    """One-update wandb-disabled train; peak GPU mem via nvidia-smi."""
    global _PROBE
    needle = str(cfg).encode()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "magna.cli.train",
            f"--cfg={cfg}",
            "--no-build",
            "--wandb_mode=disabled",
            "--save_model=false",
            f"--train.total_timesteps={64 * n * 96}",
            "--train.generate_rollout_video=false",
            "--train.print_dashboard=false",
            "--train.print_config=false",
            f"--train.data_dir={RESULT_DIR / 'probe_runs'}",
        ],
        env={**os.environ, "WANDB_MODE": "disabled"},
        start_new_session=True,
        preexec_fn=_pdeathsig,
    )
    _PROBE, peak = proc, 0
    try:
        while proc.poll() is None:
            for line in subprocess.run(
                _SMI, capture_output=True, text=True, check=False
            ).stdout.splitlines():
                pid_s, _, mib = line.partition(",")
                if not pid_s.strip().isdigit():
                    continue
                try:
                    cmd = Path(f"/proc/{int(pid_s)}/cmdline").read_bytes()
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if needle in cmd and b"wandb_mode=disabled" in cmd:
                    peak = max(peak, int(float(mib)))
            time.sleep(0.05)
    finally:
        _kill_probe()
    if proc.returncode or peak <= 0:
        raise SystemExit(f"probe {cfg} failed (code={proc.returncode}, peak={peak})")
    return peak / 1024


def _peaks() -> dict[int, float]:
    cached = (
        {}
        if FORCE_PROBE or not MEM_CSV.exists()
        else {r["cfg"]: float(r["peak_gib"]) for r in csv.DictReader(MEM_CSV.open())}
    )
    out, rows = {}, []
    for n, cfg in RUN_CFGS.items():
        key = str(cfg)
        if key not in cached:
            print(f"N={n} {cfg.name} probing...", flush=True)
        out[n] = cached[key] if key in cached else _probe(n, cfg)
        print(
            f"N={n} {cfg.name} peak_gpu={out[n]:.2f} GiB"
            + (" (cached)" if key in cached else ""),
            flush=True,
        )
        rows.append({"n": n, "cfg": key, "peak_gib": f"{out[n]:.4f}"})
    MEM_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MEM_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "cfg", "peak_gib"])
        w.writeheader()
        w.writerows(rows)
    return out


def plot_compute_scaling(points, baselines=None, out_path=None):
    paper_style()
    ns, hours, gib = map(np.asarray, zip(*points, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.2), layout="constrained")
    colors = sns.color_palette("deep")
    for i, (lab, pts) in enumerate(
        [("MAGNA", list(zip(ns, hours, strict=True))), *(baselines or [])]
    ):
        x, y = zip(*pts, strict=True)
        axes[0].plot(x, y, "-o", color=colors[i], label=lab, **_KW)
    if baselines:
        axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(ns, gib, "-o", color=colors[0], **_KW)
    for ax, ylabel in zip(
        axes, ("Wall time (h)", "Peak GPU memory (GiB)"), strict=True
    ):
        ax.set(xlabel="Number of agents", ylabel=ylabel, ylim=(0, None))
        ax.set_xticks(ns, [str(int(n)) for n in ns])
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    return save_fig(fig, out_path, bbox_inches=None) if out_path else fig


def main() -> None:
    peaks = _peaks()
    points = [(n, RUNTIME_HOURS[n], peaks[n]) for n in RUN_CFGS]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.close(
        plot_compute_scaling(
            points, _baseline_hours(), FIG_DIR / "04-compute-scaling.png"
        )
    )


if __name__ == "__main__":
    main()
