import torch


def text_to_token_ids(text, tokenizer):
    """
    将文本编码为 token ID 张量。

    参数：
        text : str
            要编码的输入文本
        tokenizer : tiktoken.Encoding
            GPT-2 tokenizer 实例（如 tiktoken.get_encoding("gpt2")）

    返回：
        ids : Tensor
            形状 (1, seq_len)，batch 维度为 1，
            可直接传入 GPTModel.forward()
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # 添加 batch 维度: (seq_len,) → (1, seq_len)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将 token ID 张量解码为文本。

    参数：
        token_ids : Tensor
            形状 (1, seq_len) 或 (seq_len,) 的 token ID 张量
        tokenizer : tiktoken.Encoding
            GPT-2 tokenizer 实例

    返回：
        text : str
            解码后的文本
    """
    # 如果是 (1, seq_len)，压平为 (seq_len,)
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
