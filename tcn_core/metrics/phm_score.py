from __future__ import annotations

import numpy as np


def compute_phm_score(pred, true) -> float:
    """CMAPSS 竞赛评价指标（PHM08 评分函数）。

    使用非对称惩罚：
        d = pred - true（d < 0 表示预测偏早/低估，d > 0 表示预测偏晚/高估）
        d < 0: score = exp(-d/13) - 1  （低估 RUL，惩罚较轻）
        d ≥ 0: score = exp(d/10) - 1   （高估 RUL，惩罚较重）

    高估 RUL 更危险（可能延误维护导致故障），因此惩罚系数更大（/10 vs /13）。
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    d = pred - true
    score = np.where(d < 0.0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))
