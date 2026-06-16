import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization（层归一化）。

    在最后一个维度上对输入做归一化，稳定训练、加速收敛。

    公式：
        1. 计算最后维度的均值 mean 和方差 var
        2. 归一化：norm_x = (x - mean) / sqrt(var + eps)
        3. 缩放和平移：y = scale * norm_x + shift
           scale (γ) 和 shift (β) 是可学习参数，让网络能恢复原始分布

    与 BatchNorm 的区别：
        - BatchNorm 跨 batch 做归一化，依赖 batch size
        - LayerNorm 跨特征维度做归一化，与 batch size 无关
        - Transformer / NLP 中 LayerNorm 是标配

    参数：
        emb_dim : int  归一化的维度大小（嵌入向量的维度）
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数

        # 可学习参数：缩放（gamma）和偏移（beta）
        # 初始：γ=1（不缩放），β=0（不偏移），让归一化后保留原始分布
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        前向传播。

        参数：
            x : Tensor  形状 (..., emb_dim)

        返回：
            y : Tensor  形状与输入相同，在最后一个维度上做了归一化
        """
        # Step 1: 沿最后一个维度（emb_dim）计算均值和方差
        # keepdim=True 保留维度，以便广播：(..., emb_dim) → (..., 1)
        # unbiased=False 使用有偏方差估计（除以 N 而非 N-1），
        #  这更适用于 LayerNorm 对已归一化输出的假设，也是 nn.LayerNorm 的默认行为
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Step 2: 归一化 — 减去均值，除以标准差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Step 3: 缩放和平移 — 可学习的仿射变换
        return self.scale * norm_x + self.shift
