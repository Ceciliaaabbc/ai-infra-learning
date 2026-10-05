"""
PyTorch 基础 1：Tensor
运行：python learn/01_tensor.py
"""

import torch

print("=== 1. 创建 tensor ===")
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(a)
print("shape:", a.shape, " dtype:", a.dtype, " device:", a.device)

print("\n=== 2. dtype 决定占多少内存 ===")
for dt in [torch.float32, torch.bfloat16, torch.int64]:
    t = torch.zeros(1000, 1000, dtype=dt)
    print(f"{str(dt):15s} 每个元素 {t.element_size()} 字节，1000×1000 共 {t.numel() * t.element_size() / 1024**2:.1f} MB")
print("124M 参数的模型：fp32 存权重 = 124M × 4B ≈ 0.5GB；bf16 ≈ 0.25GB")

print("\n=== 3. 深度学习里最常见的形状：(B, T, C) ===")
B, T, C = 2, 5, 8  # 2 个句子，每句 5 个 token，每个 token 用 8 维向量表示
x = torch.randn(B, T, C)
print("x.shape =", tuple(x.shape))
print("第 0 个句子第 3 个 token 的向量：", x[0, 3])

print("\n=== 4. view / transpose：换一种方式看同一块数据 ===")
nh, hd = 2, C // 2
y = x.view(B, T, nh, hd)  # 把 C=8 拆成 2 个头 × 每头 4 维，不复制数据
y = y.transpose(1, 2)     # (B, nh, T, hd)：把“头”这一维挪到前面
print("拆成多头后：", tuple(y.shape), " 内存是否连续：", y.is_contiguous())
z = y.transpose(1, 2).contiguous().view(B, T, C)  # transpose 后内存不连续，view 之前要 contiguous()
print("拼回去：", tuple(z.shape), " 和原来相等：", torch.equal(z, x))

print("\n=== 5. 矩阵乘法与广播 ===")
W = torch.randn(C, 16)  # 一个线性层的权重：8 维 → 16 维
out = x @ W             # (B, T, 8) @ (8, 16) → (B, T, 16)，前面的维度自动对齐（广播）
print("x @ W:", tuple(out.shape))
print(f"这次乘法的计算量 = 2 × B × T × C × 16 = {2 * B * T * C * 16} FLOPs（一次乘加算 2 次运算）")
bias = torch.randn(16)
print("加上 bias（形状 (16,) 自动广播到 (B, T, 16)）:", tuple((out + bias).shape))

print("\n=== 6. device：数据放在哪里算 ===")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
xd = x.to(device)
print("x 现在在：", xd.device, "（参与同一个运算的 tensor 必须在同一个设备上）")
