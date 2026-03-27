from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize outputs from the vendored TCN project.")
    parser.add_argument("--config", type=str, required=True, help="Path to the TCN config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from hybrid_rul.paths import materialize_resolved_yaml

    project_root = PROJECT_ROOT
    resolved_config_path = materialize_resolved_yaml(
        project_root / args.config,
        cache_root=project_root / ".resolved_configs",
    )
    cmd = [
        sys.executable,
        str(project_root / "tcn_core" / "scripts" / "visualize.py"),
        "--config",
        str(resolved_config_path),
    ]
    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
