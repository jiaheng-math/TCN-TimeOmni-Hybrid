from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm as apply_weight_norm
except ImportError:  # pragma: no cover - fallback for older PyTorch
    from torch.nn.utils import weight_norm as apply_weight_norm


class CausalConv1d(nn.Module):
    """因果卷积：仅使用当前及过去时间步的信息，不泄漏未来信息。

    通过左填充（left padding）+ 无内置 padding 的 Conv1d 实现严格因果性。
    填充量 = (kernel_size - 1) * dilation，确保输出长度与输入相同。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        # 左填充量：保证卷积核只能看到当前及之前的时间步
        self.left_padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # 不使用内置padding，手动在forward中左填充
        )
        self.conv = apply_weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅在时间维度左侧填零，保持因果性
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """TCN 残差块：两层因果卷积 + 残差连接。

    当输入输出通道数不同时，使用 1x1 卷积对残差路径做维度对齐。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, use_weight_norm=True)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, use_weight_norm=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 通道数不匹配时，用 1x1 卷积对齐维度以实现残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class TCN(nn.Module):
    """时序卷积网络（Temporal Convolutional Network）。

    通过指数增长的膨胀率（dilation = 2^i）堆叠 TCNBlock，
    使感受野随层数指数增长，高效捕获长程依赖。

    Args:
        n_features: 输入特征维度（传感器数量）
        num_channels: 每层输出通道数列表，列表长度决定网络深度
        kernel_size: 卷积核大小
        dropout: Dropout 比率
    """

    def __init__(
        self,
        n_features: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = n_features
        for i, out_channels in enumerate(num_channels):
            # 膨胀率指数增长：1, 2, 4, 8...  感受野 = O(2^layers * kernel_size)
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.ModuleList(layers)
        self.hidden_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) 时序输入
        Returns:
            (batch, hidden_dim) 取最后一个时间步的特征作为序列表示
        """
        for layer in self.network:
            x = layer(x)
        # 取最后一个时间步的输出，作为整个序列的特征表示
        return x[:, :, -1]
