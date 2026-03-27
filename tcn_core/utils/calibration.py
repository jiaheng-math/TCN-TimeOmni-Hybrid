from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from metrics.uncertainty_metrics import compute_interval_score, compute_mpiw, compute_picp


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

    lower_bound, upper_bound = 0.1, 10.0
    lower_value = picp_at_scale(lower_bound)
    upper_value = picp_at_scale(upper_bound)

    # PICP 是阶梯函数，小样本下可能没有理想的符号变化；此时退化为网格搜索。
    if lower_value == 0.0:
        return float(lower_bound)
    if upper_value == 0.0:
        return float(upper_bound)
    if lower_value * upper_value < 0:
        return float(brentq(picp_at_scale, lower_bound, upper_bound))

    candidate_scales = np.geomspace(lower_bound, upper_bound, num=200)
    candidate_errors = [abs(picp_at_scale(scale)) for scale in candidate_scales]
    best_idx = int(np.argmin(candidate_errors))
    return float(candidate_scales[best_idx])


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


def summarize_calibrated_uncertainty(
    mu: np.ndarray,
    sigma: np.ndarray,
    true: np.ndarray,
    sigma_scale: float,
    alpha: float = 0.05,
    z: float = 1.96,
) -> dict:
    """返回校准后区间质量摘要。"""
    calibrated = apply_sigma_scale(mu, sigma, sigma_scale, z=z)
    return {
        "sigma_scale": float(sigma_scale),
        "picp": compute_picp(calibrated["lower"], calibrated["upper"], true),
        "mpiw": compute_mpiw(calibrated["lower"], calibrated["upper"]),
        "interval_score": compute_interval_score(calibrated["lower"], calibrated["upper"], true, alpha=alpha),
        "lower": calibrated["lower"],
        "upper": calibrated["upper"],
        "sigma": calibrated["sigma"],
    }
