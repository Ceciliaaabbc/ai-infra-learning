"""
PyTorch 基础 3：训练循环
任务：让模型从带噪声的数据里学出 y = 3x + 2
运行：python learn/03_train_loop.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- 数据 ----
X = torch.randn(256, 1)
Y = 3 * X + 2 + 0.1 * torch.randn(256, 1)

# ---- 模型、损失函数、优化器 ----
model = nn.Linear(1, 1)  # 只有两个参数：weight（应该学到 3）和 bias（应该学到 2）
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

print(f"训练前：w = {model.weight.item():.3f}, b = {model.bias.item():.3f}")

# ---- 训练循环 ----
for step in range(100):
    idx = torch.randint(0, 256, (32,))  # 1. 取一个 batch
    x, y = X[idx], Y[idx]
    pred = model(x)                     # 2. 前向：做预测
    loss = loss_fn(pred, y)             # 3. 算 loss：预测和答案差多少
    optimizer.zero_grad()               #    清掉上一步的梯度
    loss.backward()                     # 4. 反向：算出每个参数的梯度
    optimizer.step()                    # 5. 更新参数
    if step % 20 == 0 or step == 99:
        print(f"step {step:3d} | loss {loss.item():.4f} | w = {model.weight.item():.3f}, b = {model.bias.item():.3f}")

print("\n=== 优化器本身也占内存 ===")
for name, p in model.named_parameters():
    print(f"参数 {name} {tuple(p.shape)}：AdamW 为它额外保存了 {list(optimizer.state[p].keys())}")
print("exp_avg (m) 和 exp_avg_sq (v) 都和参数一样大 → AdamW 额外需要 2 倍参数量的内存。")
print("合计：fp32 权重 4B + 梯度 4B + m 4B + v 4B = 每个参数 16 字节；124M 的模型约 2GB。")
