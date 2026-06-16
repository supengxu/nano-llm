import torch


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
