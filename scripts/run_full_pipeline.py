from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCN and then run the hybrid reasoning pipeline.")
    parser.add_argument("--tcn-config", type=str, required=True, help="Path to the TCN config YAML.")
    parser.add_argument("--hybrid-config", type=str, required=True, help="Path to the hybrid config YAML.")
    parser.add_argument("--skip-train", action="store_true", help="Skip TCN training and only run the hybrid stage.")
    parser.add_argument("--resume", action="store_true", help="Resume TCN training if latest checkpoint exists.")
    parser.add_argument("--engine-id", type=int, action="append", default=None, help="Specific engine ID to analyze in the hybrid stage.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of engines to process in the hybrid stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_tcn.py"),
            "--config",
            args.tcn_config,
        ]
        if args.resume:
            train_cmd.append("--resume")
        subprocess.run(train_cmd, check=True, cwd=project_root)

    hybrid_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_hybrid_demo.py"),
        "--config",
        args.hybrid_config,
    ]
    if args.limit is not None:
        hybrid_cmd.extend(["--limit", str(args.limit)])
    for engine_id in args.engine_id or []:
        hybrid_cmd.extend(["--engine-id", str(engine_id)])
    subprocess.run(hybrid_cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
