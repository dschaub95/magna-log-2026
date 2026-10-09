import argparse
import json
import sys
import sysconfig
from pathlib import Path

import wandb

from magna.cli.build import add_build_args, build_argv_from_namespace
from magna.cli.build import main as build_main
from magna.config import (
    Config,
    apply_cli_overrides,
    load_yaml_config,
    parse_dot_overrides,
)


def _is_binding_up_to_date(root: Path) -> bool:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    env_dir = root / "src" / "magna" / "env"
    target = env_dir / f"binding{ext_suffix}"
    if not target.exists():
        return False

    target_mtime = target.stat().st_mtime
    sources = [*env_dir.glob("*.c"), *env_dir.glob("*.cpp"), *env_dir.glob("*.h")]
    return bool(sources) and all(
        source.stat().st_mtime <= target_mtime for source in sources
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--cfg", default=None, help="Path to config YAML")
    parser.add_argument("--init-weights", help="Warm-start from this checkpoint.")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Write final training metrics JSON to this path.",
    )
    add_build_args(parser, prefix="build_")

    args, remaining = parser.parse_known_args(sys.argv[1:])
    flat_cli, _ = parse_dot_overrides(remaining)

    root = Path(__file__).resolve().parents[3]
    if not args.no_build:
        if not _is_binding_up_to_date(root):
            print("🔨 Building magna env C binding...")
            build_main(build_argv_from_namespace(args, prefix="build_"))
        else:
            print("✅ magna env C binding is up to date.")

    # import here to use latest binding after build
    from magna.train import train

    cfg_init = load_yaml_config(args.cfg, resolve=False) if args.cfg else Config()
    cfg = apply_cli_overrides(cfg_init, flat_cli, resolve=False)

    try:
        result = train(cfg, load_weights_from=args.init_weights)
        if args.metrics_out and result:
            Path(args.metrics_out).write_text(json.dumps(result.get("metrics", {})))
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
