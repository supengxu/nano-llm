"""
测试 DummyGPTModel 的前向传播，验证：
1. 模型能正常构建
2. 输入输出形状正确
3. 占位层为透传（输入=输出）
"""
import torch
import tiktoken

from nano_llm.inference.generator import DummyGPTModel, generate_text_simple
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


def test_dummy_gpt_model():
    """
    用 tiktoken 对两条文本做 tokenize，batch 后送入 DummyGPTModel，
    验证输出的 logits 形状符合预期。
    """

    # ---- 2. 创建模型 ----
    torch.manual_seed(123)  # 固定随机种子，确保每次运行结果一致
    model = DummyGPTModel(GPT_CONFIG_124M)
    model.eval()  # 切换到推理模式，关闭 Dropout

    # ---- 3. 用 tiktoken 对文本做 tokenize ----
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    # batch 中两条样本长度可能不同，stack 要求等长，这里先 pad 到相同长度
    # 也可以用 torch.nn.utils.rnn.pad_sequence，但手动 pad 更直观
    max_len = max(len(b) for b in batch)
    padded = []
    for b in batch:
        # 右边补 0（实际训练中应该用 pad_token_id）
        p = torch.nn.functional.pad(b, (0, max_len - len(b)), value=0)
        padded.append(p)

#这一步的意义是把多条独立样本组装成一个批次，让 GPU 能并行处理
    batch = torch.stack(padded, dim=0)  # 形状: (2, max_len)

    print(f"输入 batch 形状: {batch.shape}")
    print(f"batch 内容:\n{batch}")

    # ---- 4. 前向传播 ----
    # 原料（token ID）
    #     → 浸泡（embedding，变成稠密向量）
    #     → 调味（加位置信息）
    #     → 过传送带（12 层 block，词之间互看）
    #     → 抛光（LayerNorm，数值稳定）
    #     → 包装（Linear，扩到 50257 维）
    #     → 成品（logits，每个词一个分数）
    #PyTorch 默认会为所有张量操作构建计算图，用于反向传播求梯度。但在推理（测试/使用模型）时不需要梯度，这个上下文管理器告诉 PyTorch：
    with torch.enable_grad():
        logits = model(batch)

    print("Output shape:", logits.shape)
    print(logits)

    # ---- 5. 验证输出 ----
    batch_size, seq_len = batch.shape
    expected_shape = (batch_size, seq_len, GPT_CONFIG_124M["vocab_size"])

    print(f"\n输出 logits 形状: {logits.shape}")
    print(f"期望形状:         {expected_shape}")
    print(f"形状匹配: {logits.shape == expected_shape}")

    assert logits.shape == expected_shape, (
        f"形状不匹配！got {logits.shape}, expected {expected_shape}"
    )

    # 验证占位层为透传（去掉 embedding 和输出头的变换）
    # 简单验证：logits 不全是 0
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "logits 不应全为零"

    # 验证 tokenizer 解码结果（完整性检查）
    print(f"\ntxt1:            {txt1}")
    print(f"txt1 encode:     {tokenizer.encode(txt1)}")
    print(f"txt2:            {txt2}")
    print(f"txt2 encode:     {tokenizer.encode(txt2)}")
    print(f"txt1 decode 验证: {tokenizer.decode(tokenizer.encode(txt1))}")

    print("\n✅ 所有测试通过！")
    return logits


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
    from nano_llm.inference.generator import generate_text_simple

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 准备 prompt
    prompt = "Every effort moves you"
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
    test_dummy_gpt_model()
    test_gpt_model()
    test_generate_text_simple()
    test_full_generation_pipeline()
