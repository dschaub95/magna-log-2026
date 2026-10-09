import random
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import colormaps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_determinism(enabled: bool) -> None:
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    torch.use_deterministic_algorithms(enabled)


def coords_to_colors(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (n, 2 or 3)")

    colors = np.zeros((coords.shape[0], 3), dtype=np.uint8)
    lo = coords.min(axis=0)
    span = coords.max(axis=0) - lo
    colors[:, : coords.shape[1]] = (
        np.divide(coords - lo, span, out=np.zeros_like(coords), where=span > 0) * 255
    ).astype(np.uint8)
    return colors


def _normalize_color_label(value: object) -> str:
    if value is None:
        return "<missing>"
    try:
        if bool(np.isnan(value)):  # type: ignore[arg-type]
            return "<missing>"
    except TypeError:
        pass
    return str(value)


def _categorical_palette(name: str, size: int) -> np.ndarray:
    cmap = colormaps[name]
    colors = cmap(np.arange(size))[:, :3]
    return np.rint(colors * 255.0).astype(np.uint8)


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=object)
    if labels_array.ndim != 1:
        raise ValueError("labels must be 1D")

    normalized = np.asarray(
        [_normalize_color_label(value) for value in labels_array],
        dtype=object,
    )
    categories = sorted(set(normalized.tolist()))
    palette_name = "tab10" if len(categories) <= 10 else "tab20"
    palette_size = 10 if palette_name == "tab10" else 20
    palette = _categorical_palette(palette_name, palette_size)
    color_map = {
        category: palette[idx % len(palette)] for idx, category in enumerate(categories)
    }
    return np.ascontiguousarray(
        np.stack([color_map[category] for category in normalized], axis=0),
        dtype=np.uint8,
    )


def resolve_cell_colors(
    targets: np.ndarray, color_labels: np.ndarray | None = None
) -> np.ndarray:
    targets_array = np.asarray(targets, dtype=np.float32)
    if color_labels is None:
        return coords_to_colors(targets_array)

    labels_array = np.asarray(color_labels, dtype=object)
    if labels_array.ndim != 1:
        raise ValueError("color_labels must be 1D")
    if labels_array.shape[0] != targets_array.shape[0]:
        raise ValueError("color_labels must match the number of target positions")
    return labels_to_colors(labels_array)


def save_wandb_run_config_to_yaml(
    run_id: str,
    path: str | Path,
    *,
    project: str = "magna",
    entity: str | None = None,
) -> Path:
    from magna.eval import load_cfg_from_wandb_run

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"# Exported from W&B run {run_id}\n"
        + yaml.dump(
            load_cfg_from_wandb_run(run_id, project=project, entity=entity).model_dump(
                mode="json"
            ),
            default_flow_style=False,
            sort_keys=False,
        )
    )
    return output
