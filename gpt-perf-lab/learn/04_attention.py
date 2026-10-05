"""
Transformer 基础：attention 一步一步算
运行：python learn/04_attention.py
"""

import math

import torch
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

print("=== 1. 手算：3 个 token、单个头、每个向量 2 维 ===")
# 假设 3 个 token 的 q, k, v 已经算好（真实模型里由输入 x 分别乘三个权重矩阵得到）
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 0.], [0., 2.], [3., 3.]])
d = Q.size(-1)

scores = Q @ K.T / math.sqrt(d)
print("第 1 步：分数 = Q @ K^T / sqrt(d)。第 i 行第 j 列 = token i 对 token j 的关注程度\n", scores)

mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
scores = scores.masked_fill(~mask, float("-inf"))
print("\n第 2 步：因果 mask，把“未来”位置设成 -inf（预测下一个词时不能偷看后面）\n", scores)

weights = F.softmax(scores, dim=-1)
print("\n第 3 步：softmax，每一行变成加起来等于 1 的权重\n", weights)

out = weights @ V
print("\n第 4 步：用权重对 V 加权求和，得到每个 token 的新表示\n", out)

ref = F.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=True)[0]
print("\n和 PyTorch 内置 scaled_dot_product_attention 的结果一致：", torch.allclose(out, ref))

print("\n=== 2. 真实模型里的形状变化（多头）===")
B, T, C, nh = 2, 8, 16, 4
hd = C // nh
x = torch.randn(B, T, C)
W_qkv = torch.randn(C, 3 * C) / math.sqrt(C)
q, k, v = (x @ W_qkv).split(C, dim=-1)
print(f"x {tuple(x.shape)}  →  q, k, v 各 {tuple(q.shape)}")
q = q.view(B, T, nh, hd).transpose(1, 2)
k = k.view(B, T, nh, hd).transpose(1, 2)
v = v.view(B, T, nh, hd).transpose(1, 2)
print(f"拆成 {nh} 个头：q {tuple(q.shape)} = (B, 头数, T, 每头维度)")
att = q @ k.transpose(-2, -1) / math.sqrt(hd)
print(f"分数矩阵 {tuple(att.shape)} = (B, 头数, T, T)  ← 最后两维是 T × T")
att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf")).softmax(dim=-1)
y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
print(f"输出 {tuple(y.shape)}（和输入形状一样，所以可以一层一层往上叠）")

print("\n=== 3. 为什么长序列是难题：分数矩阵随 T² 增长 ===")
n_head, bytes_per = 12, 2  # GPT-2 的 12 个头，bf16 每个数 2 字节
for T in [1_024, 8_192, 32_768, 131_072]:
    gb = T * T * n_head * bytes_per / 1024**3
    print(f"T = {T:>7,}：单层、单个样本的分数矩阵 = {gb:9.2f} GB")
print("长视频动辄几十万 token → 必须用 FlashAttention（分块计算，不保存完整的 T×T 矩阵），")
print("再加上序列并行（把一条序列切到多张卡上）——这就是 JD 职责 2 要解决的问题。")
