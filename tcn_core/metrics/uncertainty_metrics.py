from __future__ import annotations

import numpy as np


def compute_picp(lower, upper, true) -> float:
    """预测区间覆盖概率（Prediction Interval Coverage Probability）。

    衡量真实值落在预测区间 [lower, upper] 内的比例。
    理想情况下，95% 置信区间的 PICP 应接近 0.95。
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.mean((true >= lower) & (true <= upper)))


def compute_mpiw(lower, upper) -> float:
    """平均预测区间宽度（Mean Prediction Interval Width）。

    衡量预测区间的平均宽度，越窄说明模型越确定。
    需要与 PICP 联合评价：窄区间但低覆盖率说明模型过于自信。
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return float(np.mean(upper - lower))
