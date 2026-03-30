from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch training from YAML GPU settings.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved launch command without executing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    env = os.environ.copy()

    if config.runtime.distributed:
        if config.runtime.num_gpu < 2:
            raise ValueError("Distributed launch requires `runtime.num_gpu >= 2`.")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={config.runtime.num_gpu}",
            "train.py",
            "--config",
            args.config,
        ]
    else:
        command = [sys.executable, "train.py", "--config", args.config]

    if args.dry_run:
        print("CUDA_VISIBLE_DEVICES=" + env.get("CUDA_VISIBLE_DEVICES", ""))
        print(" ".join(command))
        return

    raise SystemExit(subprocess.run(command, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
