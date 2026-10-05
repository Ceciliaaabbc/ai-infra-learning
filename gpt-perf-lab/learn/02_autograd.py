"""
PyTorch 基础 2：autograd 自动求导
运行：python learn/02_autograd.py
"""

import torch

print("=== 1. 最简单的例子：y = w·x + b，loss = (y - target)² ===")
x = torch.tensor(2.0)
target = torch.tensor(10.0)
w = torch.tensor(3.0, requires_grad=True)  # requires_grad=True：这是要学习的参数，需要求梯度
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b             # 前向：3×2+1 = 7
loss = (y - target) ** 2  # 前向：(7-10)² = 9
print(f"y = {y.item()}, loss = {loss.item()}")

loss.backward()           # 反向：用链式法则自动算出 loss 对每个参数的梯度
print(f"dloss/dw = {w.grad.item()}   dloss/db = {b.grad.item()}")

print("\n手算验证（链式法则）：")
print("  dloss/dy = 2(y - target) = 2 × (7 - 10) = -6")
print("  dloss/dw = dloss/dy × dy/dw = -6 × x = -12")
print("  dloss/db = dloss/dy × dy/db = -6 × 1 = -6")
print("梯度为负 → 把 w 调大能让 loss 变小。参数更新公式：w = w - lr × grad")

print("\n=== 2. 计算图：前向时 PyTorch 记录每一步是怎么算出来的 ===")
print("loss.grad_fn =", loss.grad_fn)
print("y.grad_fn    =", y.grad_fn)
print("反向需要前向的中间结果（比如 dy/dw 需要用到 x），")
print("所以前向的中间结果（激活值）要一直保存到反向结束 → 这就是训练时激活值占大量显存的原因。")
print("activation checkpointing 的思路：先不存，反向时重新算一遍，用时间换显存。")

print("\n=== 3. 梯度会累加，不会自动清零 ===")
w.grad, b.grad = None, None
for i in range(3):
    loss = (w * x + b - target) ** 2
    loss.backward()
    print(f"第 {i + 1} 次 backward 后 w.grad = {w.grad.item()}")
print("所以每一步训练前都要 optimizer.zero_grad()。")
print("反过来利用这个特性就是“梯度累积”：几个小 batch 的梯度加起来，效果等于一个大 batch。")

print("\n=== 4. 不需要梯度时用 torch.no_grad()（推理、评估）===")
with torch.no_grad():
    y2 = w * x + b
print("y2.requires_grad =", y2.requires_grad, "（不建计算图、不存中间结果，省显存也更快）")
