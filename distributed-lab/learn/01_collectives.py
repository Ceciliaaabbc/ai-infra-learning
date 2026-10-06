"""
集合通信入门：用 4 个进程演示最常用的 4 种通信操作。
每个进程扮演一张显卡，rank 是它的编号（0、1、2、3）。

运行（Mac 上必须指定 127.0.0.1，否则会卡住）：
  torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 learn/01_collectives.py
"""

import torch
import torch.distributed as dist

# gloo 是 CPU 上的通信库；到了 GPU 上会换成 NVIDIA 的 nccl，用法完全一样
dist.init_process_group("gloo")
rank = dist.get_rank()
world = dist.get_world_size()


def show(title, note, before, after):
    """按 rank 顺序依次打印，避免几个进程的输出混在一起。"""
    if rank == 0:
        print(f"\n=== {title} ===\n{note}")
    for r in range(world):
        dist.barrier()  # 所有进程在这里等齐，再轮流打印
        if r == rank:
            print(f"  rank {rank}：之前 {before}  →  之后 {after}", flush=True)
    dist.barrier()


# ---- 1. broadcast：一个人把数据发给所有人 ----
t = torch.tensor([1.0, 2.0, 3.0]) if rank == 0 else torch.zeros(3)
before = t.tolist()
dist.broadcast(t, src=0)  # 把 rank 0 的数据发给所有进程
show("broadcast（广播）",
     "rank 0 把自己的数据发给所有人。DDP 开始训练前，用它让所有卡的初始参数一致。",
     before, t.tolist())

# ---- 2. all_reduce：所有人的数据加起来，每人都拿到总和 ----
t = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
before = t.tolist()
dist.all_reduce(t, op=dist.ReduceOp.SUM)
show("all_reduce（全规约）",
     "每张卡的数据加起来，每张卡都拿到总和。DDP 每一步用它把所有卡的梯度加起来。",
     before, t.tolist())

# ---- 3. all_gather：每人贡献一块，最后每人都有完整的一份 ----
t = torch.tensor([float(rank)])
out = torch.zeros(world)
dist.all_gather_into_tensor(out, t)
show("all_gather（全收集）",
     "每张卡贡献自己那一块，拼起来后每张卡都有完整的一份。ZeRO-3 计算前用它收集完整参数。",
     t.tolist(), out.tolist())

# ---- 4. reduce_scatter：先加起来，再切块，每人只拿自己那块 ----
t = torch.arange(world, dtype=torch.float32) * 10 + rank  # rank 0: [0,10,20,30]，rank 1: [1,11,21,31]……
out = torch.zeros(1)
dist.reduce_scatter_tensor(out, t, op=dist.ReduceOp.SUM)
show("reduce_scatter（规约后分发）",
     "先把所有卡的数据加起来，再切成 world 块，第 i 张卡只拿第 i 块。ZeRO 用它让每张卡只保留自己负责的那部分梯度。",
     t.tolist(), out.tolist())

# ---- 5. 一个重要的事实：all_reduce = reduce_scatter + all_gather ----
# 真实的 all_reduce（比如 NCCL 的 ring all-reduce）就是分这两步完成的。
# 这也是为什么 ZeRO 的通信量和普通 DDP 差不多：它只是把这两步拆开，分别在不同时机做。
t = torch.arange(world, dtype=torch.float32) * 10 + rank
expected = t.clone()
dist.all_reduce(expected)
piece = torch.zeros(1)
dist.reduce_scatter_tensor(piece, t)
result = torch.zeros(world)
dist.all_gather_into_tensor(result, piece)
show("all_reduce = reduce_scatter + all_gather",
     "先 reduce_scatter 再 all_gather，结果和直接 all_reduce 完全一样。",
     f"all_reduce 结果 {expected.tolist()}", f"两步拆开的结果 {result.tolist()}，一致：{torch.equal(expected, result)}")

dist.destroy_process_group()
