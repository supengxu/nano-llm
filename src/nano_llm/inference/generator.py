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
