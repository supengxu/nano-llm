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


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    简单的自回归文本生成（贪心解码 / Greedy Decoding）。

    每一步选择概率最高的 token（argmax），没有随机性，确定性生成。
    这是最基础的解码策略，适合验证模型是否能正常推理。

    生成流程：
        1. 将当前 token 序列截断到 context_size 以内（不能超过模型最大上下文）
        2. 前向传播得到 logits
        3. 取最后一个位置的 logits，argmax 选出下一个 token
        4. 将新 token 拼接到序列末尾
        5. 重复，直到生成 max_new_tokens 个新 token

    参数：
        model : nn.Module
            语言模型（如 GPTModel），需要支持 model(x) → logits
        idx : Tensor
            形状 (batch_size, current_seq_len) 的 token ID 序列，
            作为生成的起点（prompt）
        max_new_tokens : int
            要生成的新 token 数量
        context_size : int
            模型支持的最大上下文长度，序列超过此长度会被截断

    返回：
        idx : Tensor
            形状 (batch_size, current_seq_len + max_new_tokens)，
            包含原始 prompt + 新生成的 token 序列
    """
    model.eval()  # 切换到推理模式，关闭 Dropout

    for _ in range(max_new_tokens):
        # ---- Step 1: 截断到 context_size ----
        # 如果当前序列比模型能处理的最大长度还长，只保留最后 context_size 个 token
        idx_cond = idx[:, -context_size:]

        # ---- Step 2: 前向传播 ----
        # 得到 (batch_size, seq_len, vocab_size) 的 logits
        with torch.no_grad():
            logits = model(idx_cond)

        # ---- Step 3: 取最后一个位置的 logits ----
        # logits[:, -1, :] 得到 (batch_size, vocab_size)
        # 因为我们要预测的是"下一个"token，所以只看序列末尾的预测
        logits_last = logits[:, -1, :]

        # ---- Step 4: Argmax 选最高概率的 token ----
        # argmax(dim=-1) → (batch_size,)，每个样本的"最佳下一个 token"
        # keepdim=True 保留维度 → (batch_size, 1)，方便拼接
        idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)

        # ---- Step 5: 拼接到序列末尾 ----
        # dim=-1 表示沿序列长度维度拼接
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx
