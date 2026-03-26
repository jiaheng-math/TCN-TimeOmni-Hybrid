from __future__ import annotations

from pathlib import Path


def get_experiment_name(config: dict, config_path: str | Path) -> str:
    """返回实验名，默认使用配置文件名。"""
    output_cfg = config.get("output", {})
    explicit_name = output_cfg.get("experiment_name")
    if explicit_name:
        return str(explicit_name)
    return Path(config_path).stem
