import argparse
import pprint
import sys
from pathlib import Path

from magna.config import load_yaml_config, parse_dot_overrides
from magna.eval import run_eval


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg", required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", help="Policy weights (default: checkpoints/<cfg stem>.pt)"
    )
    parser.add_argument("--video_path", default="outputs/videos/rollout_eval.mp4")
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")

    args, remaining = parser.parse_known_args(sys.argv[1:])
    flat_cli, rest = parse_dot_overrides(remaining)
    if rest:
        raise SystemExit(f"Unrecognized arguments: {' '.join(rest)}")

    result = run_eval(
        load_yaml_config(args.cfg, resolve=False),
        args.checkpoint or Path("checkpoints") / f"{Path(args.cfg).stem}.pt",
        flat_overrides=flat_cli,
        video_path=args.video_path,
        n_rollouts=args.n_rollouts,
        greedy=args.greedy,
    )

    print(f"Target obsm key: {result['target_obsm_key']}")
    print(f"Initial positions key: {result['initial_obsm_key']}")
    print(f"Video path: {result['video_path']}")
    print("Metrics:")
    pprint.pprint({k: v for k, v in result["metrics"].items() if k != "video_path"})


if __name__ == "__main__":
    main()
