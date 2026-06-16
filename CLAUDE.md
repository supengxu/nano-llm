# CLAUDE.md

本项目是一个从零构建 LLM 的学习项目（nano-llm），使用 PyTorch 逐步实现 GPT 风格的 Transformer。

## 提交规范

- 完成任何有意义的代码改动后（新功能、修复、重构、清理等），应**主动提交代码**，不需要等用户提醒。先 commit 再汇报结果。
- commit message 使用英文，格式为 conventional commits：`<type>: <description>`
  - `feat` — 新功能
  - `fix` — 修复 bug
  - `chore` — 杂项（依赖、配置等）
  - `refactor` — 重构
  - `test` — 测试
  - `docs` — 文档

## 项目结构

- `src/nano_llm/` — 源代码
  - `model/` — 模型定义
  - `inference/` — 推理相关（LayerNorm / generate_text_simple）
  - `training/` — 训练相关
  - `tokenizer/` — 分词器
  - `evaluation/` — 评估
  - `utils/` — 工具函数
- `tests/` — 测试文件
- `config/` — 配置文件
- `data/` — 数据集
- `checkpoints/` — 模型检查点
- `scripts/` — 脚本

## 技术栈

- Python 3.10+
- PyTorch（核心框架）
- tiktoken（GPT-2 tokenizer）
- uv（包管理器）
