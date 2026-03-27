from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from metrics.uncertainty_metrics import compute_picp


def calibrate_sigma_scale(
    mu: np.ndarray,
    sigma: np.ndarray,
    true: np.ndarray,
    target_picp: float = 0.95,
    z: float = 1.96,
) -> float:
    """在验证集上搜索 σ 缩放系数 T，使 PICP(T × σ) ≈ target_picp。

    原理：模型输出的 σ 可能偏小（过度自信）或偏大（过度保守），
    通过后验温度缩放（temperature scaling）找到最优 T，
    使置信区间校准到目标覆盖率。不改变 μ，不影响 RMSE。

    Args:
        mu: 预测均值
        sigma: 预测标准差（原始，未缩放）
        true: 真实 RUL
        target_picp: 目标覆盖率，默认 0.95
        z: 置信区间的 z 值，默认 1.96（对应 95%）

    Returns:
        最优缩放系数 T，使用时：calibrated_sigma = T * sigma
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    def picp_at_scale(t: float) -> float:
        scaled_sigma = t * sigma
        lower = mu - z * scaled_sigma
        upper = mu + z * scaled_sigma
        return compute_picp(lower, upper, true) - target_picp

    # 搜索范围：T ∈ [0.1, 10]，覆盖从极度自信到极度保守
    return float(brentq(picp_at_scale, 0.1, 10.0))


def apply_sigma_scale(
    mu: np.ndarray,
    sigma: np.ndarray,
    scale: float,
    z: float = 1.96,
) -> dict:
    """应用缩放系数，返回校准后的置信区间。"""
    calibrated_sigma = scale * sigma
    lower = mu - z * calibrated_sigma
    upper = mu + z * calibrated_sigma
    return {
        "sigma": calibrated_sigma,
        "lower": lower,
        "upper": upper,
    }
