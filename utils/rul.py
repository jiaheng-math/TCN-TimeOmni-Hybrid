from __future__ import annotations

import numpy as np


def clip_rul_array(values, min_value: float = 0.0, max_value: float | None = None) -> np.ndarray:
    """将预测 RUL 裁剪到物理可行范围内。"""
    values = np.asarray(values, dtype=np.float32)
    upper = np.inf if max_value is None else max_value
    return np.clip(values, min_value, upper)
