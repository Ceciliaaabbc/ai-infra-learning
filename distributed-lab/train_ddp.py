"""
项目 2 · 阶段 1：数据并行（DDP）训练 + 性能测量。结果追加到 results_ddp.csv。

数据并行的做法：
  1. 每张卡（每个进程）都有一份完整的模型，初始参数完全相同
  2. 每张卡拿不同的数据，各自做前向和反向，算出自己的梯度
  3. 所有卡把梯度求平均（all-reduce），于是每张卡拿到的梯度完全一样
  4. 每张卡各自用这个平均梯度更新参数，所以参数始终保持一致

三种梯度同步的实现（--ddp_impl），用来对比通信方式对速度的影响：
  naive  手写：反向结束后，每个参数张量单独做一次 all-reduce（GPT-2 有 148 个，就要通信 148 次）
  flat   手写：把所有梯度拼成一个大张量，只做 1 次 all-reduce
  torch  PyTorch 官方的 DistributedDataParallel：把梯度分桶，并且一边做反向一边通信

启动方式（必须用 torchrun，它会为每张卡各启动一个进程）：
  Mac / CPU 上验证正确性（Mac 上必须指定 127.0.0.1，否则会卡住）：
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \\
        train_ddp.py --model tiny --data synthetic --steps 20 --skip_steps 5 --ddp_impl naive
  多卡 GPU：见 run_ddp.sh
"""

import argparse
import contextlib
import csv
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 复用项目 0 的模型、测量工具和数据读取，不复制代码
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt-perf-lab"))
import perf
from model import GPT, GPTConfig, MODEL_PRESETS
from train import Batches, device_name

RESULT_FIELDS = [
    "time", "tag", "status", "gpu", "world_size", "ddp_impl", "model", "batch_per_gpu", "seq_len", "grad_accum",
    "tf32", "bf16", "compile", "attn", "adamw",
    "tokens_per_s", "tokens_per_s_per_gpu", "mfu_pct", "comm_ms", "peak_mem_gb", "final_loss", "params_in_sync",
]


def parse_args():
    p = argparse.ArgumentParser(description="数据并行（DDP）训练性能实验")
    # 模型与数据（和项目 0 的 train.py 一致）
    p.add_argument("--model", default="gpt2", choices=list(MODEL_PRESETS))
    p.add_argument("--data", default="shakespeare", choices=["synthetic", "shakespeare", "fineweb"])
    p.add_argument("--batch_size", type=int, default=4, help="每张卡的 micro batch size")
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--vocab_size", type=int, default=50257)
    # 训练
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--skip_steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    # 单卡性能开关（和项目 0 一致）
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--attn", default="naive", choices=["naive", "sdpa"])
    p.add_argument("--adamw", default="foreach", choices=["forloop", "foreach", "fused"])
    # 本阶段的核心开关
    p.add_argument("--ddp_impl", default="torch", choices=["naive", "flat", "torch"], help="梯度同步的实现方式")
    # 测量与输出
    p.add_argument("--device", default="auto", help="auto / cpu。auto 在有 NVIDIA 显卡时用 GPU，否则用 CPU")
    p.add_argument("--peak_tflops", type=float, default=None)
    p.add_argument("--tag", default="run")
    p.add_argument("--results", default=os.path.join(HERE, "results_ddp.csv"))
    p.add_argument("--log_interval", type=int, default=5)
    return p.parse_args()


# ==============================================================================
# 分布式环境
# ==============================================================================
def setup_distributed(device_arg):
    """初始化进程组，返回 (rank, world_size, local_rank, device)。

    rank：我是第几号进程（全局编号）。world_size：一共有几个进程。
    local_rank：我是这台机器上的第几号进程，用来决定用哪张显卡。
    这几个数由 torchrun 通过环境变量传进来。
    """
    if "RANK" not in os.environ:
        # 没用 torchrun、直接 python train_ddp.py 运行时，当作只有 1 个进程
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available() and device_arg != "cpu":
        device = torch.device("cuda", local_rank)  # 第 local_rank 号进程用第 local_rank 张卡
        torch.cuda.set_device(device)
        backend = "nccl"  # NVIDIA 的显卡通信库，会自动走 NVLink 或 PCIe
    else:
        device = torch.device("cpu")
        backend = "gloo"  # CPU 上的通信库，用来在 Mac 上验证代码
    dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def broadcast_params(model):
    """把 0 号进程的参数广播给所有进程，确保所有卡从同一个起点开始。"""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def sync_grads_naive(params, world_size):
    """每个参数张量单独做一次 all-reduce。

    问题在于通信次数太多：GPT-2 有 148 个参数张量，每一步就要通信 148 次。
    每次通信都有固定的启动延迟，小张量的通信尤其不划算。
    """
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size


def sync_grads_flat(params, world_size):
    """把所有梯度拼成一个大张量，只做 1 次 all-reduce，再拆回去。

    通信次数从 148 次降到 1 次。代价是需要一块和全部梯度一样大的临时缓冲区
    （124M 参数的 fp32 梯度约 0.5 GB），还要多做一次拼接和拆分。
    """
    grads = [p.grad for p in params if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


def params_in_sync(model):
    """检查所有进程的参数是否完全一致：每个参数求一个和，收集到一起逐个比较。"""
    local = torch.stack([p.detach().double().sum() for p in model.parameters()])
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return all(torch.equal(g, gathered[0]) for g in gathered)


def append_result(path, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ==============================================================================
# 主流程
# ==============================================================================
def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed(args.device)
    is_main = rank == 0

    def print0(*a, **kw):
        # 只让 0 号进程打印，否则每条日志会重复 world_size 遍
        if is_main:
            print(*a, **kw, flush=True)

    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    def autocast():
        if args.bf16:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ---- 模型：每个进程都建一份完整的模型 ----
    # 所有进程用同一个随机种子，初始参数本来就相同。
    torch.manual_seed(args.seed)
    cfg = GPTConfig(**MODEL_PRESETS[args.model], vocab_size=args.vocab_size, attn_impl=args.attn)
    T = args.seq_len or cfg.block_size
    B = args.batch_size
    raw_model = GPT(cfg).to(device)
    # 但不能依赖“种子相同”这一点（比如不同机器、不同库版本的随机数实现可能不同）。
    # 所以手写版本在开始前，把 0 号进程的参数广播给所有进程。
    # PyTorch 的 DDP 在包装模型时，也会自动做同样的事。
    if args.ddp_impl != "torch":
        broadcast_params(raw_model)

    # ---- 数据：每个进程用不同的种子取数据，保证各自拿到不同的样本 ----
    # 如果所有卡取到同样的数据，多卡就只是在重复计算，等于白白浪费。
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    batches = Batches(args.data, B, T, device)

    # ---- 包装模型 ----
    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay, args.adamw, device.type)
    params = [p for p in raw_model.parameters() if p.requires_grad]
    if args.ddp_impl == "torch":
        ddp_model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None)
        model = ddp_model
    else:
        ddp_model = None
        model = raw_model
    if args.compile:
        model = torch.compile(model)
    model.train()

    # ---- 测量用的数值 ----
    n_params, n_matmul = perf.count_params(raw_model)
    flops_per_tok = perf.flops_per_token(n_matmul, cfg.n_layer, cfg.n_embd, T)
    gpu = device_name(device)
    peak_tflops = args.peak_tflops or (perf.lookup_peak_tflops(gpu) if device.type == "cuda" else None)
    # 每一步所有卡加起来处理的 token 数
    tokens_per_step = B * T * args.grad_accum * world_size

    print0("=" * 72)
    print0(f"设备        {gpu} × {world_size}  （通信库 {dist.get_backend()}）")
    print0(f"模型        {args.model}  参数 {n_params / 1e6:.1f}M")
    print0(f"批次        每卡 B={B}  T={T}  grad_accum={args.grad_accum}  → 所有卡每步共 {tokens_per_step:,} tokens")
    print0(f"梯度同步    {args.ddp_impl}")
    print0(f"开关        tf32={args.tf32} bf16={args.bf16} compile={args.compile} attn={args.attn} adamw={args.adamw}")
    print0("=" * 72)

    lr_warmup = max(1, args.steps // 10)

    def get_lr(step):
        if step < lr_warmup:
            return args.lr * (step + 1) / lr_warmup
        progress = (step - lr_warmup) / max(1, args.steps - lr_warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    sync_grads = {"naive": sync_grads_naive, "flat": sync_grads_flat}.get(args.ddp_impl)

    def train_step(step):
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            # PyTorch 的 DDP 默认在每次 backward 时都同步梯度。
            # 梯度累积时只需要在最后一次同步，前面几次用 no_sync() 跳过通信。
            is_last = micro == args.grad_accum - 1
            no_sync = ddp_model.no_sync() if (ddp_model is not None and not is_last) else contextlib.nullcontext()
            with no_sync:
                x, y = batches.next()
                with autocast():
                    _, loss = model(x, y)
                loss = loss / args.grad_accum
                loss_accum += loss.detach()
                loss.backward()

        # 手写版本：反向全部结束后，再统一同步梯度，并单独计时。
        # torch 版本不需要这一步，梯度在 backward 过程中已经同步好了。
        comm_s = 0.0
        if sync_grads is not None:
            perf.synchronize(device.type)
            t0 = time.perf_counter()
            sync_grads(params, world_size)
            perf.synchronize(device.type)
            comm_s = time.perf_counter() - t0

        # 梯度裁剪必须放在同步之后：这样每张卡裁剪的是同一个平均梯度，结果才会一致。
        # 如果在同步之前裁剪，每张卡按自己的梯度大小裁剪，参数就会慢慢变得不一样。
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        return loss_accum, norm, lr, comm_s

    # ---- 训练循环 ----
    timer = perf.StepTimer(device.type, args.skip_steps)
    perf.reset_peak_memory(device.type)
    losses, comm_times = [], []
    for step in range(args.steps):
        timer.start()
        loss, norm, lr, comm_s = train_step(step)
        dt = timer.stop(tokens_per_step)
        comm_times.append(comm_s)

        # 每张卡的 loss 只反映自己那份数据，取所有卡的平均才是这一步的整体 loss。
        # 这次 all_reduce 放在计时之外，不影响测速。
        loss_all = loss.detach().clone()
        dist.all_reduce(loss_all, op=dist.ReduceOp.SUM)
        losses.append(loss_all.item() / world_size)

        if step % args.log_interval == 0 or step == args.steps - 1:
            tps = tokens_per_step / dt
            mfu = perf.compute_mfu(tps / world_size, flops_per_tok, peak_tflops)
            comm = f"通信 {comm_s * 1000:6.1f} ms | " if sync_grads is not None else ""
            note = "  (预热，不计入统计)" if step < args.skip_steps else ""
            mfu_str = "n/a" if mfu is None else f"{mfu * 100:.1f}%"
            print0(f"step {step:4d} | loss {losses[-1]:.4f} | norm {norm.item():.2f} | {dt * 1000:8.1f} ms | "
                   f"{comm}{tps:9.0f} tok/s | MFU {mfu_str:>6}{note}")

    # ---- 汇总 ----
    tps = timer.tokens_per_sec()  # 所有卡加起来的速度
    tps_per_gpu = tps / world_size if tps else None
    mfu = perf.compute_mfu(tps_per_gpu, flops_per_tok, peak_tflops) if tps else None
    measured_comm = comm_times[args.skip_steps:]
    comm_ms = sum(measured_comm) / len(measured_comm) * 1000 if (sync_grads is not None and measured_comm) else None
    peak_gb = perf.peak_memory_gb(device.type)
    final_loss = sum(losses[-10:]) / len(losses[-10:])
    in_sync = params_in_sync(raw_model)  # 所有进程都要参与这次通信

    print0("-" * 72)
    print0(f"总吞吐      {tps:,.0f} tokens/s（{world_size} 张卡加起来）" if tps else "总吞吐      n/a")
    print0(f"每卡吞吐    {tps_per_gpu:,.0f} tokens/s" if tps_per_gpu else "每卡吞吐    n/a")
    print0(f"MFU         {'n/a' if mfu is None else f'{mfu * 100:.1f}%'}（每张卡）")
    if comm_ms is not None:
        print0(f"每步通信    {comm_ms:.1f} ms（手写版本单独计时；torch 版本的通信和反向重叠，无法单独测）")
    print0(f"峰值显存    {'n/a（只在 CUDA 上统计）' if peak_gb is None else f'{peak_gb:.2f} GB（0 号卡）'}")
    print0(f"最终 loss   {final_loss:.4f}（所有卡的平均，最后 10 步）")
    print0(f"参数一致    {'✓ 所有卡的参数完全相同' if in_sync else '✗ 各卡参数不一致，梯度同步有问题'}")

    if is_main:
        append_result(args.results, {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tag": args.tag, "status": "ok",
            "gpu": gpu, "world_size": world_size, "ddp_impl": args.ddp_impl, "model": args.model,
            "batch_per_gpu": B, "seq_len": T, "grad_accum": args.grad_accum,
            "tf32": args.tf32, "bf16": args.bf16, "compile": args.compile, "attn": args.attn, "adamw": args.adamw,
            "tokens_per_s": round(tps) if tps else "",
            "tokens_per_s_per_gpu": round(tps_per_gpu) if tps_per_gpu else "",
            "mfu_pct": round(mfu * 100, 2) if mfu else "",
            "comm_ms": round(comm_ms, 2) if comm_ms is not None else "",
            "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else "",
            "final_loss": round(final_loss, 4), "params_in_sync": in_sync,
        })
        print0(f"结果已追加到 {args.results}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
