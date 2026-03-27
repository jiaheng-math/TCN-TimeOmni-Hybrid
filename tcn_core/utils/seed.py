import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置全局随机种子以确保实验可复现。

    同时固定 Python、NumPy、PyTorch（CPU/GPU）的随机数生成器，
    并关闭 cuDNN 的非确定性算法优化（benchmark=False, deterministic=True）。
    注意：deterministic 模式可能略微降低训练速度。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
