"""
测试 GPTModel 的前向传播和文本生成，验证：
1. 模型能正常构建
2. 输入输出形状正确
3. 不同输入产生不同输出
4. 训练模式下 Dropout 生效
5. generate_text_simple 生成正确
6. 完整文本生成流水线
"""
import torch
import tiktoken

from nano_llm.inference.generator import generate_text_simple
from nano_llm.model.transformer import GPTModel
from nano_llm.utils.helpers import text_to_token_ids, token_ids_to_text


# GPT-2 124M 参数的配置（参考 Sebastian Raschka "Build a LLM from Scratch"）
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词表大小
    "context_length": 1024,  # 最大上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 多头注意力的头数
    "n_layers": 12,          # Transformer 块层数
    "drop_rate": 0.1,        # Dropout 概率
    "qkv_bias": False,       # 注意力 QKV 投影是否使用偏置
}


def test_gpt_model():
    """
    测试 GPTModel（真实 TransformerBlock 版本）的前向传播，验证：
    1. 模型能正常构建
    2. 输入输出形状正确
    3. 不同输入产生不同输出（不是透传的占位层）
    4. 训练模式下两次前向传播结果不同（Dropout 生效）
    """
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # ---- 测试 1: 推理模式下输出形状正确 ----
    model.eval()
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (batch_size, seq_len, GPT_CONFIG_124M["vocab_size"])
    assert logits.shape == expected_shape, \
        f"形状不匹配！got {logits.shape}, expected {expected_shape}"

    # ---- 测试 2: logits 不全为零（模型确实在做计算） ----
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "logits 不应全为零"

    # ---- 测试 3: 不同输入产生不同输出 ----
    input_ids_2 = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))
    with torch.no_grad():
        logits_2 = model(input_ids_2)

    assert not torch.allclose(logits, logits_2), \
        "不同输入应产生不同输出（模型不是透传的占位层）"

    # ---- 测试 4: 训练模式下 Dropout 生效 ----
    model.train()
    out1 = model(input_ids)
    out2 = model(input_ids)  # 相同输入
    # Dropout 开启时，两次前向传播结果应该不同
    assert not torch.allclose(out1, out2), \
        "训练模式下两次前向传播应不同（Dropout 生效）"

    print("\n✅ GPTModel 所有测试通过！")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   输入形状: {input_ids.shape}")
    print(f"   输出形状: {logits.shape}")


def test_generate_text_simple():
    """
    测试 generate_text_simple 函数，验证：
    1. 能正常生成 token 序列
    2. 输出形状正确（原始 token + max_new_tokens）
    3. 原始 prompt 被完整保留
    4. 新 token 数量正确
    """
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 准备 prompt
    prompt = "你是谁"
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    # 生成
    max_new_tokens = 10
    out_ids = generate_text_simple(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    # 验证形状: (1, len(prompt) + max_new_tokens)
    expected_len = input_ids.shape[1] + max_new_tokens
    assert out_ids.shape == (1, expected_len), \
        f"输出形状不匹配！got {out_ids.shape}, expected {(1, expected_len)}"

    # 验证原始 prompt token 被完整保留
    assert torch.equal(out_ids[0, :input_ids.shape[1]], input_ids[0]), \
        "原始 prompt token 应被完整保留"

    # 验证新生成的 token ID 在合法范围内
    new_tokens = out_ids[0, input_ids.shape[1]:]
    assert new_tokens.shape[0] == max_new_tokens, \
        f"新 token 数量应为 {max_new_tokens}, got {new_tokens.shape[0]}"
    assert torch.all(new_tokens >= 0) and torch.all(new_tokens < GPT_CONFIG_124M["vocab_size"]), \
        "新生成的 token ID 应在合法范围内"

    # 解码看输出
    generated_text = tokenizer.decode(out_ids[0].tolist())
    print(f"\n提示词:     {prompt}")
    print(f"生成文本:   {generated_text}")
    print(f"生成 token 数: {max_new_tokens}")
    print("✅ generate_text_simple 测试通过！")


def test_full_generation_pipeline():
    """
    测试完整的文本生成流水线，串联 text_to_token_ids / generate_text_simple / token_ids_to_text：
    1. text_to_token_ids 将文本编码为模型输入格式
    2. generate_text_simple 自回归生成新 token
    3. token_ids_to_text 将 token 序列解码回文本

    这是"从文本到文本"的端到端测试。
    """
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    # ---- Step 1: 文本 → Token IDs ----
    prompt = "Every effort moves you"
    input_ids = text_to_token_ids(prompt, tokenizer)
    assert input_ids.shape[0] == 1, "batch 维度应为 1"
    print(f"\n输入文本:     {repr(prompt)}")
    print(f"Token IDs:    {input_ids[0].tolist()}")

    # ---- Step 2: 自回归生成 ----
    max_new_tokens = 15
    out_ids = generate_text_simple(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    # 验证 prompt 部分没有被修改
    assert torch.equal(out_ids[0, :input_ids.shape[1]], input_ids[0]), \
        "prompt token 不应被修改"

    # ---- Step 3: Token IDs → 文本 ----
    generated_text = token_ids_to_text(out_ids, tokenizer)
    print(f"生成的 token: {out_ids[0].tolist()}")
    print(f"生成文本:     {repr(generated_text)}")

    # ---- 验证 ----
    # 解码后的文本应以原始 prompt 开头
    assert generated_text.startswith(prompt), \
        f"生成文本应以 prompt 开头: {repr(generated_text[:len(prompt)])} != {repr(prompt)}"
    # 生成的文本应比 prompt 长（有新 token）
    assert len(out_ids[0]) > len(input_ids[0]), \
        f"应有新 token 生成: {len(out_ids[0])} <= {len(input_ids[0])}"
    # 新生成的 token 数量 == max_new_tokens
    assert len(out_ids[0]) == len(input_ids[0]) + max_new_tokens, \
        f"新 token 数应为 {max_new_tokens}"

    print(f"\n提示词 token 数: {len(input_ids[0])}")
    print(f"生成后 token 数: {len(out_ids[0])}")
    print(f"新增 token 数:   {len(out_ids[0]) - len(input_ids[0])}")
    print("✅ 完整生成流水线测试通过！")


if __name__ == "__main__":
    test_gpt_model()
    test_generate_text_simple()
    test_full_generation_pipeline()
