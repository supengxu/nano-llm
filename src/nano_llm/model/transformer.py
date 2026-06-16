import torch
import torch.nn as nn

from nano_llm.model.transformer_block import TransformerBlock
from nano_llm.inference.generator import LayerNorm


class GPTModel(nn.Module):
    """
    GPT 风格 Transformer 模型。

    这是一个完整可训练的 GPT 模型，包含：
        - Token Embedding — 将 token ID 映射为稠密向量
        - Position Embedding — 为序列中每个位置提供位置信息
        - N × TransformerBlock — 多头自注意力 + 前馈网络，逐层变换表示
        - Final LayerNorm — 最后的归一化，稳定训练
        - Output Head — 线性投影从隐藏维度到词表大小，得到 logits

    数据流：
        Token IDs → Token Embedding → + Position Embedding → Dropout
        → N × TransformerBlock → LayerNorm → Linear Head → Logits

    这是 DummyGPTModel 的"毕业"版本——所有组件都已替换为真实实现。

    参数：
        cfg : dict  包含模型超参数的字典，需要：
            - vocab_size      : int  词表大小（token 的种类数）
            - emb_dim         : int  嵌入向量的维度
            - context_length  : int  最大序列长度
            - drop_rate       : float  Dropout 丢弃概率
            - n_layers        : int  Transformer 块的层数
            - n_heads         : int  注意力头数
            - qkv_bias        : bool  Q/K/V 注意力投影是否使用偏置
    """

    def __init__(self, cfg):
        super().__init__()

        # ---- 词嵌入层（Token Embedding）----
        # 一个可学习的查找表 (vocab_size × emb_dim)，将每个 token ID 映射为稠密向量。
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

        # ---- Transformer 块序列 ----
        # 用 nn.Sequential 把 n_layers 个 TransformerBlock 串起来，
        # 数据依次流过每个块。每个块内部包含 Pre-LN 风格的多头自注意力和前馈网络，
        # 以及残差连接和 Dropout。
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # ---- 最后的 Layer Normalization ----
        # 在所有 Transformer 块之后、输出头之前做一次层归一化，稳定训练。
        self.final_norm = LayerNorm(cfg["emb_dim"])

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
        # device=in_idx.device 确保索引和输入在同一设备（CPU/GPU）
        # 查表得到 (seq_len, emb_dim)，广播到整个 batch
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Step 3: 相加 — Token 语义 + 位置信息，逐元素相加
        # Shape: (batch_size, num_tokens, emb_dim)
        x = tok_embeds + pos_embeds

        # Step 4: Dropout — 训练时随机置零部分元素，防止过拟合
        x = self.drop_emb(x)

        # Step 5: 依次通过 N 个 Transformer 块
        # 每个块做：Pre-LN → Multi-Head Self-Attention → FFN → 残差连接
        x = self.trf_blocks(x)

        # Step 6: 最后的 Layer Normalization
        x = self.final_norm(x)

        # Step 7: 线性输出头 — 从 emb_dim 投影回 vocab_size，得到 logits
        logits = self.out_head(x)

        return logits
