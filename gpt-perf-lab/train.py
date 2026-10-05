"""
单卡 GPT 训练 + 性能测量。每次运行会把结果追加到 results.csv。

冒烟测试（CPU / Mac 也能跑）：
  python train.py --model tiny --data synthetic --steps 20 --skip_steps 5

正式实验（GPU）：见 run_experiments.sh
Profiling：在任意命令后加 --profile
"""

import argparse
import contextlib
import csv
import datetime
import math
import os
import sys

import numpy as np
import torch

import perf
from model import GPT, GPTConfig, MODEL_PRESETS

ROOT = os.path.dirname(os.path.abspath(__file__))

RESULT_FIELDS = [
    "time", "tag", "status", "gpu", "model", "params_M", "batch", "seq_len", "grad_accum", "vocab",
    "tf32", "bf16", "compile", "attn", "adamw", "act_ckpt",
    "tokens_per_s", "mfu_pct", "peak_mem_gb", "static_mem_gb", "final_loss",
]


def parse_args():
    p = argparse.ArgumentParser(description="单卡 GPT 训练性能实验")
    # 模型与数据
    p.add_argument("--model", default="gpt2", choices=list(MODEL_PRESETS))
    p.add_argument("--data", default="shakespeare", choices=["synthetic", "shakespeare", "fineweb"],
                   help="synthetic = 随机 token，不需要下载数据，只用来测速度")
    p.add_argument("--batch_size", type=int, default=4, help="micro batch size (B)")
    p.add_argument("--seq_len", type=int, default=None, help="序列长度 (T)，默认等于模型的 block_size")
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--vocab_size", type=int, default=50257)
    # 训练
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--skip_steps", type=int, default=10, help="前 N 步不计入性能统计（编译、预热）")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    # 性能开关：实验的核心
    p.add_argument("--tf32", action="store_true", help="fp32 矩阵乘法用 TF32 Tensor Core")
    p.add_argument("--bf16", action="store_true", help="bf16 混合精度 (autocast)")
    p.add_argument("--compile", action="store_true", help="torch.compile")
    p.add_argument("--attn", default="naive", choices=["naive", "sdpa"], help="sdpa 在 GPU 上会用 FlashAttention")
    p.add_argument("--adamw", default="foreach", choices=["forloop", "foreach", "fused"])
    p.add_argument("--act_ckpt", action="store_true", help="activation checkpointing")
    # 测量与输出
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--peak_tflops", type=float, default=None, help="手动指定显卡 bf16 峰值算力，用于算 MFU")
    p.add_argument("--profile", action="store_true", help="用 torch.profiler 采样几步，导出 trace 后退出")
    p.add_argument("--tag", default="run", help="实验名，写进 results.csv")
    p.add_argument("--results", default=os.path.join(ROOT, "results.csv"))
    p.add_argument("--log_interval", type=int, default=5)
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return {"mps": "Apple MPS", "cpu": "CPU"}.get(device.type, device.type)


class Batches:
    """每次随机取 B 段长度为 T+1 的连续 token：前 T 个是输入 x，后 T 个是目标 y（错开一位）。"""

    def __init__(self, source, B, T, device):
        self.B, self.T, self.device = B, T, device
        self.synthetic = source == "synthetic"
        if not self.synthetic:
            path = os.path.join(ROOT, "data", source, "train.bin")
            if not os.path.exists(path):
                hint = "python prepare_data.py" + (" --source fineweb" if source == "fineweb" else "")
                sys.exit(f"找不到 {path}，请先运行：{hint}")
            self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
            assert len(self.tokens) > T + 1, "数据太短"

    def next(self):
        B, T = self.B, self.T
        if self.synthetic:
            buf = torch.randint(0, 50257, (B, T + 1), device=self.device)
            return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()
        ix = np.random.randint(0, len(self.tokens) - T - 1, size=B)
        buf = torch.from_numpy(np.stack([self.tokens[i:i + T + 1] for i in ix]).astype(np.int64))
        x, y = buf[:, :-1].contiguous(), buf[:, 1:].contiguous()
        if self.device.type == "cuda":
            # pinned memory + non_blocking：CPU→GPU 拷贝可以和 GPU 计算重叠
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def append_result(path, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def fmt(value, spec, missing="n/a"):
    return missing if value is None else format(value, spec)


def pct(ratio):
    return "n/a" if ratio is None else f"{ratio * 100:.1f}%"


def main():
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 精度设置 ----
    # "highest" = 纯 fp32；"high" = 允许 fp32 矩阵乘法走 TF32 Tensor Core（只影响 fp32 运算）
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    def autocast():
        # bf16 混合精度：矩阵乘法等用 bf16 算，softmax / loss 等对精度敏感的操作自动保持 fp32
        if args.bf16:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ---- 模型 ----
    cfg = GPTConfig(**MODEL_PRESETS[args.model], vocab_size=args.vocab_size,
                    attn_impl=args.attn, act_ckpt=args.act_ckpt)
    T = args.seq_len or cfg.block_size
    if T > cfg.block_size:
        sys.exit(f"--seq_len {T} 超过模型 block_size {cfg.block_size}")
    B = args.batch_size
    tokens_per_step = B * T * args.grad_accum

    raw_model = GPT(cfg).to(device)
    n_params, n_matmul = perf.count_params(raw_model)
    flops_per_tok = perf.flops_per_token(n_matmul, cfg.n_layer, cfg.n_embd, T)
    peak_tflops = args.peak_tflops or (perf.lookup_peak_tflops(device_name(device)) if device.type == "cuda" else None)
    static_gb = perf.static_memory_bytes(n_params) / 1024**3

    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay, args.adamw, device.type)
    model = torch.compile(raw_model) if args.compile else raw_model
    model.train()
    batches = Batches(args.data, B, T, device)

    print("=" * 72)
    print(f"设备        {device_name(device)}  (bf16 峰值 {fmt(peak_tflops, '.0f')} TFLOPS)")
    print(f"模型        {args.model}  参数 {n_params / 1e6:.1f}M  层数 {cfg.n_layer}  隐藏维度 {cfg.n_embd}  词表 {cfg.vocab_size}")
    print(f"批次        B={B}  T={T}  grad_accum={args.grad_accum}  → 每步 {tokens_per_step:,} tokens")
    print(f"开关        tf32={args.tf32} bf16={args.bf16} compile={args.compile} attn={args.attn} "
          f"adamw={args.adamw} act_ckpt={args.act_ckpt}")
    print(f"估算        每 token {flops_per_tok / 1e9:.2f} GFLOPs；参数+梯度+优化器状态 ≈ {static_gb:.2f} GB")
    print("=" * 72)

    lr_warmup = max(1, args.steps // 10)

    def get_lr(step):
        # 线性 warmup，然后余弦衰减到 10%
        if step < lr_warmup:
            return args.lr * (step + 1) / lr_warmup
        progress = (step - lr_warmup) / max(1, args.steps - lr_warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    def train_step(step):
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = batches.next()
            with autocast():
                _, loss = model(x, y)
            loss = loss / args.grad_accum  # 累积 grad_accum 次，相当于一个大 batch 的平均
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        return loss_accum, norm, lr

    if args.profile:
        run_profile(args, device, train_step)
        return

    row = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tag": args.tag, "status": "ok",
        "gpu": device_name(device), "model": args.model, "params_M": round(n_params / 1e6, 1),
        "batch": B, "seq_len": T, "grad_accum": args.grad_accum, "vocab": cfg.vocab_size,
        "tf32": args.tf32, "bf16": args.bf16, "compile": args.compile, "attn": args.attn,
        "adamw": args.adamw, "act_ckpt": args.act_ckpt, "static_mem_gb": round(static_gb, 2),
    }

    timer = perf.StepTimer(device.type, args.skip_steps)
    perf.reset_peak_memory(device.type)
    losses = []
    try:
        for step in range(args.steps):
            timer.start()
            loss, norm, lr = train_step(step)
            dt = timer.stop(tokens_per_step)
            losses.append(loss.item())
            if step % args.log_interval == 0 or step == args.steps - 1:
                tps = tokens_per_step / dt
                mfu = perf.compute_mfu(tps, flops_per_tok, peak_tflops)
                note = "  (预热，不计入统计)" if step < args.skip_steps else ""
                print(f"step {step:4d} | loss {losses[-1]:.4f} | lr {lr:.2e} | norm {norm.item():.2f} | "
                      f"{dt * 1000:8.1f} ms | {tps:9.0f} tok/s | MFU {pct(mfu):>6}{note}")
    except torch.cuda.OutOfMemoryError:
        print("\n[OOM] 显存不够。可以减小 --batch_size，或打开 --bf16 / --attn sdpa / --act_ckpt")
        append_result(args.results, {**row, "status": "OOM"})
        sys.exit(1)

    tps = timer.tokens_per_sec()
    mfu = perf.compute_mfu(tps, flops_per_tok, peak_tflops) if tps else None
    peak_gb = perf.peak_memory_gb(device.type)
    final_loss = sum(losses[-10:]) / len(losses[-10:])

    print("-" * 72)
    print(f"吞吐        {fmt(tps, ',.0f')} tokens/s  （第 {args.skip_steps} 步之后的平均）")
    print(f"MFU         {pct(mfu)}" + ("" if mfu is not None else "  （未知显卡峰值，可用 --peak_tflops 指定）"))
    if peak_gb is not None:
        print(f"峰值显存    {peak_gb:.2f} GB  （其中参数+梯度+优化器状态约 {static_gb:.2f} GB，其余主要是激活值）")
    else:
        print("峰值显存    n/a  （只在 CUDA 上统计）")
    print(f"最终 loss   {final_loss:.4f}  （最后 10 步平均，用来确认优化没有改变训练结果）")

    append_result(args.results, {
        **row,
        "tokens_per_s": round(tps) if tps else "",
        "mfu_pct": round(mfu * 100, 2) if mfu else "",
        "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else "",
        "final_loss": round(final_loss, 4),
    })
    print(f"结果已追加到 {args.results}")


def run_profile(args, device, train_step):
    from torch.profiler import ProfilerActivity, profile, schedule

    # 先跑几步预热，确保 torch.compile 编译完成，不把编译时间混进 profile
    print(f"预热 {args.skip_steps} 步…")
    for step in range(args.skip_steps):
        train_step(step)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    if device.type == "cuda":
        # PyTorch 2.4 起改名为 self_device_time_total，旧版本仍是 self_cuda_time_total
        from torch.autograd.profiler_util import FunctionEvent
        sort_key = "self_device_time_total" if hasattr(FunctionEvent, "self_device_time_total") else "self_cuda_time_total"
    else:
        sort_key = "self_cpu_time_total"
    trace_dir = os.path.join(ROOT, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"{args.tag}.json")

    def on_trace_ready(prof):
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
        prof.export_chrome_trace(trace_path)
        print(f"\ntrace 已导出到 {trace_path}，用 https://ui.perfetto.dev 打开查看时间线")

    # wait 1 步、warmup 1 步，然后正式记录 3 步
    with profile(activities=activities, schedule=schedule(wait=1, warmup=1, active=3),
                 on_trace_ready=on_trace_ready, record_shapes=True) as prof:
        for i in range(5):
            train_step(args.skip_steps + i)
            prof.step()


if __name__ == "__main__":
    main()
