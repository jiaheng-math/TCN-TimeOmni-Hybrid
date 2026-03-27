from __future__ import annotations

import math


# 四级维护预警等级：正常 → 关注 → 预警 → 危险
LEVELS = ["正常", "关注", "预警", "危险"]


def _resolve_warning_config(config: dict) -> dict:
    """兼容传入完整 config 或仅传入 warning 子配置的情况。"""
    if "warning" in config:
        return config["warning"]
    return config


def get_warning_level(mu: float, logvar: float, config: dict) -> dict:
    """根据预测的 RUL 均值和不确定性计算维护预警等级。

    决策逻辑基于 95% 置信区间的下界（lower = μ - 1.96σ），
    即在最坏情况下的 RUL 估计：
        lower > 80  → 正常（发动机健康）
        50 < lower ≤ 80 → 关注（需密切监测）
        20 < lower ≤ 50 → 预警（应安排维护）
        lower ≤ 20 → 危险（需立即维护）

    不确定性升级机制：当 σ 超过阈值时，即使均值看似正常，
    也将预警等级上调一级，防止高不确定性下的误判。
    """
    warning_cfg = _resolve_warning_config(config)
    thresholds = warning_cfg["thresholds"]
    sigma_threshold = warning_cfg["sigma_threshold"]
    sigma_escalation = warning_cfg.get("sigma_escalation", True)

    # 从 log(σ²) 恢复 σ: σ = exp(0.5 * log(σ²))
    sigma = math.exp(0.5 * float(logvar))
    # 95% 置信区间下界
    lower = float(mu) - 1.96 * sigma

    if lower > thresholds["normal"]:
        level_idx = 0
    elif lower > thresholds["watch"]:
        level_idx = 1
    elif lower > thresholds["alert"]:
        level_idx = 2
    else:
        level_idx = 3

    # 不确定性升级：σ 过大时提高一级预警（除非已是最高级）
    escalated = False
    if sigma_escalation and sigma > sigma_threshold and level_idx < len(LEVELS) - 1:
        level_idx += 1
        escalated = True

    return {
        "level": LEVELS[level_idx],
        "escalated": escalated,
        "lower": lower,
        "sigma": sigma,
    }
