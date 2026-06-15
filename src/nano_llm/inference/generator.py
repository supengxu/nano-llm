import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    """
    GPT 风格 Transformer 模型的简化骨架（占位版本）。

    这个类的目的是先搭好模型的整体结构，验证前向传播能跑通、张量形状正确，
    之后再逐步把 DummyTransformerBlock 和 DummyLayerNorm 替换为有实际计算逻辑的实现。

    数据流：
        Token IDs → Token Embedding → + Position Embedding → Dropout
        → N × TransformerBlock → LayerNorm → Linear Head → Logits

    参数：
        cfg : dict  包含模型超参数的字典，例如：
            - vocab_size      : 词表大小（token 的种类数）
            - emb_dim         : 嵌入向量的维度
            - context_length  : 最大序列长度
            - drop_rate       : Dropout 丢弃概率
            - n_layers        : Transformer 块的层数
    """

    def __init__(self, cfg):
        super().__init__()

        # ---- 词嵌入层（Token Embedding）----
        # 一个可学习的查找表 (vocab_size × emb_dim)，将每个 token ID（整数）映射为一个稠密向量。
        # token ID 的取值范围：0 到 vocab_size-1
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # ---- 位置嵌入层（Position Embedding）----
        # 也是一个可学习的查找表 (context_length × emb_dim)，给序列中每个位置
        # （0 到 context_length-1）学习一个 emb_dim 维的向量。
        # Transformer 本身不具备位置感知能力，通过位置嵌入告诉模型"第 i 个 token 在第 i 位"。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # ---- Dropout 层 ----
        # 训练时以 drop_rate 的概率随机将嵌入向量中的某些元素置零，防止过拟合。
        # 推理时（eval 模式）自动关闭，不做丢弃。
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # ---- Transformer 块序列（占位）----
        # 用 nn.Sequential 把 n_layers 个 DummyTransformerBlock 串起来，
        # 数据依次流过每个块。
        # *[...] 是 Python 的解包语法，把列表展开成 Sequential 的多个参数。
        # 注意：DummyTransformerBlock 目前是"透传"层——输入什么就输出什么，
        # 真实实现会包含多头自注意力（Multi-Head Self-Attention）和前馈网络（FFN）。
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # ---- 最后的 Layer Normalization（占位）----
        # 在所有 Transformer 块之后、输出头之前做一次层归一化，稳定训练。
        # 目前 DummyLayerNorm 也是透传的，不执行任何计算。
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # ---- 输出头（LM Head）----
        # 一个线性层，把 emb_dim 维的隐藏状态投影回 vocab_size 维。
        # 每个位置的输出是一个 logits 向量（未归一化的预测分数），
        # 表示模型对该位置"下一个 token"的预测。
        # bias=False 是 GPT-2 风格的做法——不用偏置项。
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        前向传播。

        参数：
            in_idx : Tensor  形状 (batch_size, seq_len)，内容是 token ID 序列

        返回：
            logits : Tensor  形状 (batch_size, seq_len, vocab_size)，
                             每个位置的 logits 向量表示对"下一个 token"的预测分数
        """
        # 从输入形状中解包出批次大小和序列长度
        batch_size, seq_len = in_idx.shape

        # Step 1: Token Embedding — 将 token ID 映射为 emb_dim 维向量
        # 输入 (batch_size, seq_len) → 输出 (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # Step 2: Position Embedding — 为序列中每个位置生成位置编码
        # torch.arange(seq_len) → [0, 1, 2, ..., seq_len-1]
        # .device=in_idx.device 确保索引和输入在同一设备（CPU/GPU）
        # 查表得到 (seq_len, emb_dim)，广播到整个 batch
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Step 3: 相加 — Token 语义 + 位置信息，逐元素相加
        x = tok_embeds + pos_embeds

        # Step 4: Dropout — 训练时随机置零部分元素，防止过拟合
        x = self.drop_emb(x)

        # Step 5: 依次通过 N 个 Transformer 块（目前为占位实现）
        x = self.trf_blocks(x)

        # Step 6: 最后的 Layer Normalization（目前为占位实现）
        x = self.final_norm(x)

        # Step 7: 线性输出头 — 从 emb_dim 投影回 vocab_size，得到 logits
        logits = self.out_head(x)

        return logits


class DummyTransformerBlock(nn.Module):
    """
    占位 Transformer 块 — 暂时什么都不做，输入原样返回。

    真实的 Transformer 块会包含：
        - 多头自注意力（Multi-Head Self-Attention）
        - 前馈网络（Feed-Forward Network / FFN）
        - 残差连接（Residual Connections）
        - Layer Normalization

    这个骨架存在的意义是让整体模型结构先跑通，
    后续替换为真实实现即可，不需要改动 DummyGPTModel.forward() 的逻辑。
    """

    def __init__(self, cfg):
        super().__init__()
        # 占位：不定义任何可学习参数，只是一个"透传"层

    def forward(self, x):
        """
        前向传播（占位）— 直接将输入返回，不做任何变换。

        参数：
            x : Tensor  形状 (batch_size, seq_len, emb_dim)

        返回：
            x : Tensor  与输入完全相同
        """
        return x


class DummyLayerNorm(nn.Module):
    """
    占位 Layer Normalization — 暂时什么都不做，输入原样返回。

    真实的 LayerNorm 会：
        1. 计算输入最后一个维度的均值和方差
        2. 做归一化：y = (x - mean) / sqrt(var + eps)
        3. 应用可学习的缩放参数（gamma / weight）和偏移参数（beta / bias）

    这里接受 normalized_shape 和 eps 参数只是为了保持接口一致，
    方便后续无缝替换为 nn.LayerNorm 或自定义实现。
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        参数：
            normalized_shape : int   归一化的维度大小（通常是 emb_dim）
            eps              : float 防止除零的小常数，默认 1e-5
        """
        super().__init__()
        # 占位：不定义 gamma / beta 等可学习参数，仅保持接口兼容

    def forward(self, x):
        """
        前向传播（占位）— 直接将输入返回，不做任何归一化计算。

        参数：
            x : Tensor  形状 (..., normalized_shape)

        返回：
            x : Tensor  与输入完全相同
        """
        return x
