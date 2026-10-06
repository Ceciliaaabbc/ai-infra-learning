"""
为什么数据并行算出来的结果是对的？

每张卡只用一部分数据算梯度，再把所有卡的梯度求平均，
结果和“一张卡用全部数据算梯度”完全一样。下面用一个最小的模型验证这一点。

运行（Mac 上必须指定 127.0.0.1）：
  torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 learn/02_ddp_math.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn

dist.init_process_group("gloo")
rank = dist.get_rank()
world = dist.get_world_size()
assert 8 % world == 0, "8 条数据要能平均分给每个进程"

# 所有进程用同一个随机种子，所以数据和模型的初始参数都完全一样
torch.manual_seed(0)
X = torch.randn(8, 4)
Y = torch.randn(8, 1)
model = nn.Linear(4, 1)


def show(lines):
    """按 rank 顺序依次打印。"""
    for r in range(world):
        dist.barrier()
        if r == rank:
            print("\n".join(lines), flush=True)
    dist.barrier()


# ---- 方法 1：一张卡用全部 8 条数据算梯度，作为标准答案 ----
loss_full = ((model(X) - Y) ** 2).mean()
loss_full.backward()
grad_full = model.weight.grad.clone()
model.zero_grad()

# ---- 方法 2：数据并行。每个进程只用属于自己的那一份数据 ----
n = 8 // world
x_mine, y_mine = X[rank * n:(rank + 1) * n], Y[rank * n:(rank + 1) * n]
loss_mine = ((model(x_mine) - y_mine) ** 2).mean()
loss_mine.backward()
grad_mine = model.weight.grad.clone()  # 每个进程算出来的梯度各不相同

# 把所有进程的梯度加起来，再除以进程数，得到平均梯度
dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM)
model.weight.grad /= world
grad_ddp = model.weight.grad

if rank == 0:
    print(f"一共 8 条数据，分给 {world} 个进程，每个进程 {n} 条")
    print(f"标准答案（一张卡用全部数据）：{[round(v, 4) for v in grad_full.flatten().tolist()]}\n")
show([
    f"rank {rank} 用第 {rank * n}～{(rank + 1) * n - 1} 条数据：",
    f"  自己算出的梯度：  {[round(v, 4) for v in grad_mine.flatten().tolist()]}",
    f"  所有进程平均后：  {[round(v, 4) for v in grad_ddp.flatten().tolist()]}",
    f"  和标准答案的最大差距：{(grad_ddp - grad_full).abs().max().item():.2e}",
])
if rank == 0:
    print("\n结论：每个进程自己的梯度都不一样，但求平均之后，和用全部数据算出的梯度一致"
          "（差距只有浮点误差）。\n所以数据并行既能把计算分摊到多张卡上，又不会改变训练结果。"
          "\n前提是每个进程分到的数据条数相同，否则简单平均会有偏差。")

dist.destroy_process_group()
