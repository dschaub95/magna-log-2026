"""Run numbered experiment scripts: ``uv run experiment [N ...] [script-args...]``."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_RE = re.compile(r"^(\d+)-.+\.py$")
_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent


def discover() -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in _DIR.glob("*.py"):
        m = _SCRIPT_RE.match(path.name)
        if m:
            found[int(m.group(1))] = path
    return dict(sorted(found.items()))


def _parse(argv: list[str]) -> tuple[list[int] | None, list[str]]:
    """Leading digit tokens are experiment ids; the rest are forwarded."""
    if argv in (["-h"], ["--help"]):
        return None, argv
    if argv == ["--list"]:
        return None, argv
    ids: list[int] = []
    i = 0
    while i < len(argv) and argv[i].isdigit():
        ids.append(int(argv[i]))
        i += 1
    return ids, argv[i:]


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ids, rest = _parse(argv)
    scripts = discover()

    if rest in (["-h"], ["--help"]):
        print(
            "Usage: experiment [N ...] [script-args...]\n"
            "       experiment --list\n"
            "Run all numbered experiments, or only those listed (e.g. experiment 1 3).\n"
            "Non-number args are forwarded to each script."
        )
        return
    if rest == ["--list"]:
        for n, path in scripts.items():
            print(f"{n:02d}  {path.name}")
        return
    if not scripts:
        raise SystemExit(f"No experiment scripts found in {_DIR}")

    if ids is not None and ids:
        missing = [n for n in ids if n not in scripts]
        if missing:
            raise SystemExit(f"Unknown experiment(s): {missing}; have {list(scripts)}")
        selected = [(n, scripts[n]) for n in ids]
    else:
        selected = list(scripts.items())

    for n, path in selected:
        print(f"=== experiment {n:02d}: {path.name} ===", flush=True)
        proc = subprocess.run([sys.executable, str(path), *rest], cwd=_ROOT)
        if proc.returncode:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
