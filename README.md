# nano-llm

从零构建一个能聊天的 LLM，覆盖完整 pipeline：**数据 → Tokenizer → 预训练 → 微调 → 推理 → 评估**。

灵感来自 [nanochat](https://github.com/karpathy/nanochat)——Andrej Karpathy 的 "从零到 ChatGPT 级别聊天模型" 项目，用最简洁的代码走通全链路。

---

## 在做什么

这不是一个玩具 demo，也不是一个生产框架。它的目标是**用尽可能少的代码，让你看懂一个现代 LLM 的每一环是怎么工作的**。

核心链路：

```
原始语料 → 训练分词器 → 预训练基座模型 → SFT 微调 → RLHF/DPO → Chat 推理 → 评测
```

## 项目结构

```
nano-llm/
├── pyproject.toml
├── requirements.txt
├── config/
│   └── default.yaml              # 模型/训练/推理配置
├── src/nano_llm/
│   ├── tokenizer/
│   │   └── bpe.py                # BPE 分词器训练与编解码
│   ├── model/
│   │   ├── config.py             # GPT 模型配置 (ModelConfig)
│   │   ├── embedding.py          # Token Embedding + Positional Embedding
│   │   ├── attention.py          # 多头因果自注意力 + Flash Attention
│   │   ├── transformer_block.py  # Transformer 层 (Attention + FFN + Residual)
│   │   └── transformer.py        # 完整 GPT 模型
│   ├── training/
│   │   ├── data.py               # 数据加载与预处理
│   │   ├── optimizer.py          # 优化器与学习率调度
│   │   └── trainer.py            # 训练循环
│   ├── inference/
│   │   └── generator.py          # 文本生成 (temperature/top-k/top-p)
│   ├── evaluation/
│   │   └── metrics.py            # 评估指标 (HellaSwag, Lambada 等)
│   └── utils/
│       └── helpers.py            # 日志、检查点、分布式工具
├── scripts/
│   ├── train.py                  # 训练入口
│   ├── generate.py               # 推理入口
│   └── evaluate.py               # 评测入口
├── tests/
├── data/                         # 数据集
└── checkpoints/                  # 模型权重
```

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 训练分词器
python scripts/train.py --stage tokenizer

# 预训练基座模型
python scripts/train.py --stage pretrain

# 推理
python scripts/generate.py --prompt "你好，"
```

## 参考

- [nanochat](https://github.com/karpathy/nanochat) — Andrej Karpathy
- [nanoGPT](https://github.com/karpathy/nanoGPT) — 最小 GPT 训练实现
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

Apache 2.0
