from __future__ import annotations

import torch
import torch.nn as nn


class PointHead(nn.Module):
    """点预测输出头：将 TCN 编码特征映射为单一 RUL 预测值。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class GaussianHead(nn.Module):
    """高斯（不确定性）输出头：输出均值 μ 和对数方差 log(σ²)。

    异方差建模：每个样本有独立的预测不确定性（σ），
    而非整个数据集共享一个固定方差。
    log(σ²) 被裁剪到 [clamp_min, clamp_max] 以防止数值不稳定。
    """

    def __init__(self, hidden_dim: int, clamp_min: float = -10.0, clamp_max: float = 10.0) -> None:
        super().__init__()
        self.head_mu = nn.Linear(hidden_dim, 1)       # 预测均值 μ
        self.head_logvar = nn.Linear(hidden_dim, 1)    # 预测对数方差 log(σ²)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.head_mu(x).squeeze(-1)
        # 裁剪 log(σ²) 避免 σ 过大（爆炸）或过小（坍缩）
        logvar = self.head_logvar(x).squeeze(-1).clamp(self.clamp_min, self.clamp_max)
        return mu, logvar
