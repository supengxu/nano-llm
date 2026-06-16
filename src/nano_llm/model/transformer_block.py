import torch.nn as nn

from nano_llm.model.attention import MultiHeadAttention
from nano_llm.model.feed_forward import FeedForward
from nano_llm.inference.generator import LayerNorm


class TransformerBlock(nn.Module):
    """
    Transformer 块（GPT-2 风格，Pre-LayerNorm）。

    每个块包含两个子层：
        1. 多头自注意力（Multi-Head Self-Attention）— token 之间交换信息
        2. 前馈网络（Feed-Forward Network）— 对每个 token 独立做非线性变换

    每个子层前有 Layer Normalization，后有残差连接和 Dropout。
    这种 "Pre-LN" 设计（先归一化再计算）比原始论文的 "Post-LN" 训练更稳定，
    是 GPT-2 / GPT-3 等现代 Transformer 的标配。

    数据流：
        x → LayerNorm → Attention → Dropout → + x (残差)
          → LayerNorm → FFN       → Dropout → + x (残差)

    参数：
        cfg : dict  包含模型超参数的字典，需要：
            - emb_dim         : int  嵌入向量维度
            - context_length  : int  最大序列长度
            - n_heads         : int  注意力头数
            - drop_rate       : float  Dropout 丢弃概率
            - qkv_bias        : bool  Q/K/V 注意力投影是否使用偏置
    """

    def __init__(self, cfg):
        super().__init__()
        # ---- 多头自注意力 ----
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        # ---- 前馈网络 ----
        self.ff = FeedForward(cfg)

        # ---- Layer Normalization（两个独立的 LN，参数不共享）----
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 注意力之前的 LN
        self.norm2 = LayerNorm(cfg["emb_dim"])  # FFN 之前的 LN

        # ---- Dropout ----
        # 残差连接之后对主路径做 dropout，防止过拟合
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        前向传播。

        参数：
            x : Tensor  形状 (batch_size, num_tokens, emb_dim)

        返回：
            x : Tensor  形状 (batch_size, num_tokens, emb_dim)，与输入形状相同
        """
        # ---- 子层 1: 多头自注意力 + 残差连接 ----
        shortcut = x                # 保存输入，用于残差连接
        x = self.norm1(x)          # Pre-LN: 先归一化
        x = self.att(x)            # 多头自注意力
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut           # 残差连接: 加上原始输入

        # ---- 子层 2: 前馈网络 + 残差连接 ----
        shortcut = x                # 保存第一子层的输出
        x = self.norm2(x)          # Pre-LN: 先归一化
        x = self.ff(x)             # 逐位置前馈网络
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut           # 残差连接: 加上第一子层的输出

        return x
