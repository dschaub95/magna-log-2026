import os
import sys
from pathlib import Path

from experiments.utils import checkpoint


def test_checkpoint_resolves_shipped_newest_run_and_pin(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["exp", "--videos"])
    assert checkpoint("Grid2d") == Path("checkpoints/grid2d.pt")

    old, new = (tmp_path / "grid2d" / r / "model_best.pt" for r in ("a", "b"))
    for i, path in enumerate((old, new)):
        path.parent.mkdir(parents=True)
        path.touch()
        os.utime(path, (i, i))
    monkeypatch.setattr(sys, "argv", ["exp", "--runs", str(tmp_path)])
    assert checkpoint("Grid2d") == new

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--ckpt", f"grid2d={old}"])
    assert checkpoint("Grid2d") == old
