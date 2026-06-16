"""
CLI 脚本：使用 GPTModel 进行文本生成。

用法：
    python scripts/generate.py "Hello, world" --max-new-tokens 50

依赖项目库（src/nano_llm/），运行时需确保项目根目录在 PYTHONPATH 中：
    PYTHONPATH=src python scripts/generate.py "Once upon a time"
"""

import argparse

import torch
import tiktoken

from nano_llm.model import GPTModel
from nano_llm.inference import generate_text_simple

# ---- 默认模型配置（GPT-2 124M 规模）----
DEFAULT_CONFIG = {
    "vocab_size": 50257,       # GPT-2 词表大小
    "context_length": 1024,    # 最大上下文长度
    "emb_dim": 768,            # 嵌入维度
    "n_heads": 12,             # 注意力头数
    "n_layers": 12,            # Transformer 层数
    "drop_rate": 0.1,          # Dropout 概率
    "qkv_bias": True,          # QKV 投影使用偏置
}


def load_model(checkpoint_path=None, device=None):
    """初始化模型，可选加载预训练权重。"""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    cfg = DEFAULT_CONFIG.copy()
    model = GPTModel(cfg)
    model.to(device)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()
    return model, device


def main():
    parser = argparse.ArgumentParser(description="Generate text using GPTModel")
    parser.add_argument("prompt", type=str, help="Input prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Number of tokens to generate (default: 50)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cpu/cuda/mps, default: auto)")
    args = parser.parse_args()

    # ---- 1. 加载模型 ----
    model, device = load_model(args.checkpoint, args.device)
    print(f"Model on {device}")

    # ---- 2. 加载分词器 ----
    tokenizer = tiktoken.get_encoding("gpt2")

    # ---- 3. 将 prompt 编码为 token IDs ----
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    print(f"Input tokens: {input_ids.shape[1]}")

    # ---- 4. 生成 ----
    with torch.no_grad():
        output_ids = generate_text_simple(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            context_size=DEFAULT_CONFIG["context_length"],
        )

    # ---- 5. 解码并输出 ----
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print("\n--- Generated Text ---")
    print(output_text)


if __name__ == "__main__":
    main()
