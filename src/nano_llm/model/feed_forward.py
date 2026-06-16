import torch.nn as nn


class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络（Feed-Forward Network / FFN）。

    这是 Transformer 块的"思考"组件 — 在自注意力让 token 之间交换信息之后，
    FFN 对每个位置的表示独立做 non-linear 变换，增强模型的表达容量。

    GPT-2 / GPT-3 风格的设计：
        Linear(emb_dim → 4 × emb_dim) → GELU → Linear(4 × emb_dim → emb_dim)

    第一层将隐藏维度扩展 4 倍（增加"思考空间"），经过 GELU 激活（引入非线性），
    第二层压缩回原始维度（恢复形状，以便接下一层或残差连接）。

    参数：
        cfg : dict  包含模型超参数的字典，需要：
            - emb_dim : int  嵌入向量的维度
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # 扩展层：emb_dim → 4×emb_dim，放大"思考空间"
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # GELU 激活：smooth ReLU 替代，处处可导，经验上比 ReLU 更适合 Transformer
            nn.GELU(),
            # 压缩层：4×emb_dim → emb_dim，恢复原始形状
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        前向传播。

        参数：
            x : Tensor  形状 (batch_size, seq_len, emb_dim)

        返回：
            out : Tensor  形状 (batch_size, seq_len, emb_dim)，与输入形状相同
        """
        return self.layers(x)
