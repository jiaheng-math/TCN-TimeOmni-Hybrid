from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        default_value = match.group(2)
        if variable_name in os.environ:
            return os.environ[variable_name]
        if default_value is not None:
            return default_value
        return ""

    return ENV_PATTERN.sub(replace, value)


def expand_env_tree(payload):
    if isinstance(payload, dict):
        return {key: expand_env_tree(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [expand_env_tree(item) for item in payload]
    if isinstance(payload, str):
        return expand_env_string(payload)
    return payload


def load_yaml(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as fp:
        return expand_env_tree(yaml.safe_load(fp))


def dump_yaml(payload: dict, path: str | Path) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return destination


def materialize_resolved_yaml(path: str | Path, cache_root: str | Path | None = None) -> Path:
    source_path = Path(path).resolve()
    payload = load_yaml(source_path)
    if cache_root is None:
        cache_root = source_path.parent / ".resolved_configs"
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / source_path.name
    with destination.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return destination


def resolve_path(base_dir: str | Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
