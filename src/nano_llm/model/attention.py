import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）。

    将输入拆分为多个"头"，每个头在较小的子空间内独立做 scaled dot-product attention，
    最后拼接所有头的输出并做一次线性投影。多头设计让模型能同时关注不同位置的
    不同表示子空间（类似卷积中多个 filter 捕获不同模式）。

    GPT-2 风格的设计：
        - Q/K/V 通过一个大的线性层合并计算（`qkv_bias` 控制是否加偏置）
        - 使用 causal mask 确保每个 token 只能看到自己及之前的 token
        - 输出投影层将拼接后的多头表示映射回原始维度

    参数：
        d_in : int            输入特征维度
        d_out : int           输出特征维度（通常等于 d_in）
        context_length : int  最大序列长度，用于构造 causal mask
        num_heads : int       注意力头数（d_out 必须能被 num_heads 整除）
        dropout : float       Attention dropout 概率
        qkv_bias : bool       Q/K/V 投影是否使用偏置项
    """

    def __init__(self, d_in, d_out, context_length, num_heads,
                 dropout, qkv_bias=False):
        super().__init__()
        # 确保 d_out 能被 num_heads 整除，否则每个头分到的维度不同
        assert d_out % num_heads == 0, \
            f"d_out ({d_out}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个注意力头的维度

        # ---- Q/K/V 投影 ----
        # 用一个大的线性层同时计算 Q、K、V（concat 成 d_out*3 维），
        # 比分别做三次 Linear 略高效（一次矩阵乘法 vs 三次）
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # ---- 输出投影 ----
        # 将拼接后的多头输出投影回原始维度
        # bias=True 在 GPT 中也是常见做法（QKV 用 bias=False，输出投影用 bias=True）
        self.out_proj = nn.Linear(d_out, d_out)

        # ---- Attention Dropout ----
        # 在 softmax 之后的注意力权重上做 dropout
        self.dropout = nn.Dropout(dropout)

        # ---- Causal Mask ----
        # 注册为一个不参与训练的 buffer（不会被 optimizer 的 state_dict 记录，
        # 但会随模型保存/加载）。上三角矩阵，确保位置 i 只能 attend 位置 ≤ i。
        # 形状: (context_length, context_length)，值为 True 的位置需要被 mask 掉
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
            .bool()
        )

    def forward(self, x):
        """
        前向传播。

        参数：
            x : Tensor  形状 (batch_size, num_tokens, d_in)

        返回：
            out : Tensor  形状 (batch_size, num_tokens, d_out)
        """
        batch_size, num_tokens, _ = x.shape

        # ---- Step 1: 线性投影得到 Q、K、V ----
        # 每个形状: (batch_size, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # ---- Step 2: 将 d_out 拆分为 num_heads × head_dim ----
        # 变换形状：(batch, num_tokens, d_out)
        #        → (batch, num_tokens, num_heads, head_dim)
        #        → (batch, num_heads, num_tokens, head_dim)
        # 转置后每个头可以独立做矩阵乘法
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # ---- Step 3: Scaled Dot-Product Attention ----
        # attn_scores: Q @ K^T / sqrt(head_dim)
        # 形状: (batch, num_heads, num_tokens, num_tokens)
        # 即每个头对每对 (query_i, key_j) 计算一个相关性分数
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # ---- Step 4: Causal Mask ----
        # 将未来位置（j > i）的分数设为 -inf，
        # softmax 后这些位置的权重变为 0，防止信息从未来"泄露"到过去
        # mask[:num_tokens, :num_tokens] 截取当前序列长度对应的子矩阵
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )

        # ---- Step 5: Softmax → 注意力权重 ----
        # dim=-1 表示沿 key 维度归一化（每个 query 对各个 key 的权重和为 1）
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # ---- Step 6: Dropout 随机丢弃部分注意力连接 ----
        attn_weights = self.dropout(attn_weights)

        # ---- Step 7: 加权求和 ----
        # attn_weights @ values: (batch, num_heads, num_tokens, num_tokens)
        #                     @ (batch, num_heads, num_tokens, head_dim)
        #                     → (batch, num_heads, num_tokens, head_dim)
        # 本质：用注意力权重对 values 加权平均，得到 context vector
        context = attn_weights @ values

        # ---- Step 8: 拼接所有头 ----
        # transpose(1,2): (batch, num_heads, num_tokens, head_dim)
        #               → (batch, num_tokens, num_heads, head_dim)
        # .contiguous() 确保内存连续（transpose 后张量不一定连续）
        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out
        )

        # ---- Step 9: 输出投影 ----
        out = self.out_proj(context)

        return out
