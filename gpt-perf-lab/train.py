"""
单卡 GPT 训练 + 性能测量。每次运行会把结果追加到 results.csv。

冒烟测试（CPU / Mac 也能跑）：
  python train.py --model tiny --data synthetic --steps 20 --skip_steps 5

正式实验（GPU）：见 run_experiments.sh
Profiling：在任意命令后加 --profile

================================================================================
一、这个文件做什么
================================================================================

运行一次 = 做一个实验。整个流程是：

  1. 解析命令行参数    决定打开哪些优化（--tf32 --bf16 --compile ...）
  2. 设置精度          TF32 开不开、bf16 混合精度开不开
  3. 准备              创建模型、优化器、数据读取器；预先算好 FLOPs、显存估算等
  4. 训练循环          每一步：取数据 → 前向 → 算 loss → 反向 → 更新参数，同时计时
  5. 汇总结果          算出平均 tokens/s、MFU、峰值显存、最终 loss，
                       打印出来，并追加一行到 results.csv

  加了 --profile 时，第 4、5 步换成：用 profiler 采样几步，打印最耗时的算子，导出时间线文件。

第 4 步里的训练五步，和 learn/03_train_loop.py 完全一样，只是多了几样东西：
混合精度、梯度累积、梯度裁剪、学习率调度。

================================================================================
二、每个性能开关在哪里生效
================================================================================

  --tf32        本文件 main() 里的 set_float32_matmul_precision
  --bf16        本文件 main() 里的 autocast()
  --compile     本文件 main() 里的 torch.compile
  --attn        model.py 的 CausalSelfAttention.forward
  --act_ckpt    model.py 的 GPT.forward
  --adamw       model.py 的 GPT.configure_optimizer
  --vocab_size  model.py 的 GPTConfig（改变 embedding 和输出层的大小）
"""

# train.py：做一次实验
# 读命令行开关，决定这次打开哪些优化
# 创建模型、优化器和数据读取器
# 训练 50 步。每一步依次是：取数据 → 前向 → 算 loss → 反向 → 更新参数，同时计时
# 算出平均速度、MFU、峰值显存，追加一行到 results.csv


import argparse
import contextlib
import csv
import datetime
import math
import os
import sys

import numpy as np
import torch

import perf  # 测量工具：FLOPs、MFU、显存估算、计时器
from model import GPT, GPTConfig, MODEL_PRESETS

# 这个脚本所在的文件夹。数据、结果等路径都基于它来拼，
# 这样不管你在哪个目录下运行 python train.py，都能找到正确的文件。
ROOT = os.path.dirname(os.path.abspath(__file__))

# results.csv 的列。前半部分记录“这次实验的配置”（打开了哪些开关），
# 后半部分记录“测出来的结果”。有了配置列，以后看结果时才知道每一行是怎么跑出来的。
RESULT_FIELDS = [
    "time", "tag", "status", "gpu", "model", "params_M", "batch", "seq_len", "grad_accum", "vocab",
    "tf32", "bf16", "compile", "attn", "adamw", "act_ckpt",
    "tokens_per_s", "mfu_pct", "peak_mem_gb", "static_mem_gb", "final_loss",
]


# ==============================================================================
# 命令行参数
# ==============================================================================
# 所有实验条件都通过命令行开关控制，不需要改代码。
# 好处：每个实验都可以用一行命令完整复现，run_experiments.sh 只需要换不同的开关组合。
def parse_args():
    p = argparse.ArgumentParser(description="单卡 GPT 训练性能实验")

    # ---- 模型与数据 ----
    p.add_argument("--model", default="gpt2", choices=list(MODEL_PRESETS))
    p.add_argument("--data", default="shakespeare", choices=["synthetic", "shakespeare", "fineweb"],
                   help="synthetic = 随机 token，不需要下载数据，只用来测速度")
    # micro batch：一次前向/反向同时处理几条序列。受显存限制，不能无限调大。
    p.add_argument("--batch_size", type=int, default=4, help="micro batch size (B)")
    p.add_argument("--seq_len", type=int, default=None, help="序列长度 (T)，默认等于模型的 block_size")
    # 梯度累积：显存放不下大 batch 时，分几次小 batch 算梯度再一起更新，效果等于一个大 batch
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--vocab_size", type=int, default=50257)

    # ---- 训练 ----
    # 50 步对测速度足够了：我们关心的是每一步有多快，不是把模型训好。
    p.add_argument("--steps", type=int, default=50)
    # 前几步特别慢，不能算进平均速度。原因包括：CUDA 初始化、cuBLAS 挑选最快的矩阵乘法算法、
    # 显存分配器第一次申请内存，以及 torch.compile 编译（可能要几十秒）。
    p.add_argument("--skip_steps", type=int, default=10, help="前 N 步不计入性能统计（编译、预热）")
    # 学习率 6e-4 和 weight decay 0.1 都是 GPT-3 论文里小模型用的值
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    # 固定随机种子：每个实验的初始化参数和取到的数据都一样，
    # 这样不同实验之间的 loss 才能直接比较，用来确认“优化没有改变训练结果”。
    p.add_argument("--seed", type=int, default=1337)

    # ---- 性能开关：实验的核心 ----
    p.add_argument("--tf32", action="store_true", help="fp32 矩阵乘法用 TF32 Tensor Core")
    p.add_argument("--bf16", action="store_true", help="bf16 混合精度 (autocast)")
    p.add_argument("--compile", action="store_true", help="torch.compile")
    p.add_argument("--attn", default="naive", choices=["naive", "sdpa"], help="sdpa 在 GPU 上会用 FlashAttention")
    p.add_argument("--adamw", default="foreach", choices=["forloop", "foreach", "fused"])
    p.add_argument("--act_ckpt", action="store_true", help="activation checkpointing")

    # ---- 测量与输出 ----
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--peak_tflops", type=float, default=None, help="手动指定显卡 bf16 峰值算力，用于算 MFU")
    p.add_argument("--profile", action="store_true", help="用 torch.profiler 采样几步，导出 trace 后退出")
    p.add_argument("--tag", default="run", help="实验名，写进 results.csv")
    p.add_argument("--results", default=os.path.join(ROOT, "results.csv"))
    p.add_argument("--log_interval", type=int, default=5)  # 每隔几步打印一行日志
    return p.parse_args()


# 自动选择设备，优先级：NVIDIA GPU (cuda) > Apple 芯片 GPU (mps) > CPU
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 设备名称，比如 "NVIDIA GeForce RTX 4090"。
# 两个用途：在 perf.py 的峰值表里查理论算力；写进 results.csv，方便区分不同显卡的结果。
def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return {"mps": "Apple MPS", "cpu": "CPU"}.get(device.type, device.type)


# ==============================================================================
# 数据读取
# ==============================================================================
class Batches:
    """每次随机取 B 段长度为 T+1 的连续 token：前 T 个是输入 x，后 T 个是目标 y（错开一位）。

    为什么错开一位：语言模型的任务是“预测下一个 token”。
      tokens:  [我, 爱, 吃, 苹, 果]
      x:       [我, 爱, 吃, 苹]
      y:       [爱, 吃, 苹, 果]      y[t] 就是 x[t] 后面的那个 token
    答案直接来自文本本身，不需要人工标注，这就是“自监督学习”。

    为什么随机取：实现最简单，每一步都是全新的随机样本。
    这里只关心速度，不需要严格按顺序把数据完整过一遍（epoch）。
    """

    def __init__(self, source, B, T, device):
        self.B, self.T, self.device = B, T, device
        self.synthetic = source == "synthetic"
        if not self.synthetic:
            path = os.path.join(ROOT, "data", source, "train.bin")
            if not os.path.exists(path):
                hint = "python prepare_data.py" + (" --source fineweb" if source == "fineweb" else "")
                sys.exit(f"找不到 {path}，请先运行：{hint}")
            # memmap（内存映射）：不把整个文件读进内存，而是用到哪一段，操作系统才去读哪一段。
            # 数据集有几十 GB 时也能这样用，不会撑爆内存。
            # uint16 是 prepare_data.py 存文件时用的格式（每个 token 占 2 字节）。
            self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
            assert len(self.tokens) > T + 1, "数据太短"

    def next(self):
        B, T = self.B, self.T

        if self.synthetic:
            # 随机数据直接在 GPU 上生成，完全没有读文件、CPU→GPU 拷贝的开销，
            # 测出来的是纯计算速度。loss 没有意义（随机数据学不到规律）。
            # 范围固定用 0~50257，也就是真实存在的 token 编号，即使词表补齐到 50304 也一样。
            buf = torch.randint(0, 50257, (B, T + 1), device=self.device)
            # 切片 [:, :-1] 得到的是原张量的“视图”，内存不连续。
            # model.py 里会对 targets 调用 .view(-1)，它要求内存连续，所以这里先 contiguous()。
            return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()

        # 随机选 B 个起点，每个起点取 T+1 个连续 token
        ix = np.random.randint(0, len(self.tokens) - T - 1, size=B)
        # 拼成 (B, T+1)，再转成 int64：cross_entropy 的 target 必须是 int64 类型
        buf = torch.from_numpy(np.stack([self.tokens[i:i + T + 1] for i in ix]).astype(np.int64))
        x, y = buf[:, :-1].contiguous(), buf[:, 1:].contiguous()

        if self.device.type == "cuda":
            # pinned memory + non_blocking：CPU→GPU 拷贝可以和 GPU 计算重叠
            #   普通内存可能被操作系统换到硬盘上，GPU 不能直接读，要先复制到一块临时缓冲区，
            #   而且拷贝期间 CPU 只能干等。
            #   pin_memory() 把数据放进“锁页内存”，GPU 可以直接读（DMA）；
            #   non_blocking=True 让 CPU 不用等拷贝完成，可以继续往下安排计算任务。
            #   这里每批数据很小，效果不明显。但在真实的大规模数据管线里（JD 职责 4），
            #   这是防止 GPU 空等数据的基本手段。
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


# ==============================================================================
# 小工具函数
# ==============================================================================
# 把一次实验的结果追加到 CSV 的末尾。文件不存在时先写表头。
# 每次运行只追加一行，所以可以连续跑很多实验，最后用 report.py 统一汇总。
def append_result(path, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# 格式化数字；值为 None（比如 Mac 上没有显存统计）时显示 "n/a"
def fmt(value, spec, missing="n/a"):
    return missing if value is None else format(value, spec)


# 把比例显示成百分比，比如 0.123 → "12.3%"
def pct(ratio):
    return "n/a" if ratio is None else f"{ratio * 100:.1f}%"


# ==============================================================================
# 主流程
# ==============================================================================
def main():
    args = parse_args()
    device = pick_device(args.device)
    # 两个随机源都要固定：torch 负责模型初始化，numpy 负责随机取数据的位置
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 精度设置 ----
    # "highest" = 纯 fp32；"high" = 允许 fp32 矩阵乘法走 TF32 Tensor Core（只影响 fp32 运算）
    #   fp32 有 23 位尾数。TF32 只保留 10 位尾数，但指数位和 fp32 一样是 8 位，
    #   所以能表示的数值范围不变，只是精度低一些，对训练几乎没有影响。
    #   好处是可以用 Tensor Core 计算，速度快得多（check_gpu.py 在 4090 上实测：51.5 → 84.1 TFLOPS）。
    #   注意：它只影响 fp32 的矩阵乘法。打开 bf16 之后，大部分矩阵乘法已经是 bf16 了，
    #   这个开关只影响剩下少数还在用 fp32 的部分。
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    def autocast():
        # bf16 混合精度：矩阵乘法等用 bf16 算，softmax / loss 等对精度敏感的操作自动保持 fp32
        #
        # “混合”的意思：
        #   - 模型权重本身仍然以 fp32 保存，参数更新也在 fp32 上做，保证精度
        #   - 在 autocast 范围内，矩阵乘法、线性层这些计算量大的操作，会临时把输入转成 bf16 再算
        #   - softmax、LayerNorm、cross_entropy 这些对精度敏感的操作，自动保持 fp32
        #
        # 好处：bf16 矩阵乘法比 fp32 快得多（4090 实测 153.7 vs 84.1 TFLOPS）；
        #   前向保存的中间结果也变成 bf16，激活值显存减半。
        #
        # 为什么选 bf16 而不是 fp16：bf16 的指数位和 fp32 一样是 8 位，数值范围相同，
        #   不会上溢或下溢，所以不需要 fp16 那套“loss 缩放”（GradScaler），用起来简单得多。
        #   代价是尾数只有 7 位，精度较低，但对训练影响很小。
        if args.bf16:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        # nullcontext 是一个“什么都不做”的上下文管理器。
        # 这样下面统一写 `with autocast():` 就行，不用分两种情况写两遍代码。
        return contextlib.nullcontext()

    # ---- 模型 ----
    # 先取预设的模型大小，再用命令行参数覆盖：词表大小和两个性能开关
    cfg = GPTConfig(**MODEL_PRESETS[args.model], vocab_size=args.vocab_size,
                    attn_impl=args.attn, act_ckpt=args.act_ckpt)
    T = args.seq_len or cfg.block_size  # 没有指定序列长度时，用模型支持的最大长度
    if T > cfg.block_size:
        sys.exit(f"--seq_len {T} 超过模型 block_size {cfg.block_size}")
    B = args.batch_size
    # 每次参数更新一共处理多少个 token。它是计算吞吐量（tokens/s）的基本单位。
    tokens_per_step = B * T * args.grad_accum

    raw_model = GPT(cfg).to(device)

    # 这些量在训练开始前算一次就够了，后面计算 MFU 和打印时会用到
    n_params, n_matmul = perf.count_params(raw_model)
    # 每个 token 训练一次（前向 + 反向）需要多少次浮点运算，GPT-2 约 0.855 GFLOPs
    flops_per_tok = perf.flops_per_token(n_matmul, cfg.n_layer, cfg.n_embd, T)
    # 优先用命令行手动指定的峰值；否则按显卡名称查表；不是 CUDA 设备就没有峰值，MFU 显示 n/a
    peak_tflops = args.peak_tflops or (perf.lookup_peak_tflops(device_name(device)) if device.type == "cuda" else None)
    # 参数 + 梯度 + AdamW 状态，每个参数 16 字节
    static_gb = perf.static_memory_bytes(n_params) / 1024**3

    # 优化器用 raw_model（编译前的原始模型）的参数来创建。
    # 编译后的模型和原始模型共享同一份参数，所以训练效果完全一样；
    # 用原始模型更清楚，也避免编译后参数名多出 "_orig_mod." 前缀带来的麻烦。
    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay, args.adamw, device.type)

    # torch.compile：把模型的 Python 代码转换成计算图，再自动生成优化过的 GPU kernel。
    #   最主要的优化是“算子融合”：比如 GELU 和后面的加法，原本是两个 kernel，
    #   各自要从显存读一遍、写一遍数据；融合成一个 kernel 后只需要读写一次。
    #   训练中很多操作的瓶颈是显存读写速度，而不是计算速度，所以融合能明显提速。
    #   同时也减少了 Python 解释器的开销。
    # 注意：这一行并不会马上编译。真正的编译发生在第一次前向计算时，
    #   所以第 0 步会特别慢（几十秒），这也是要设置 skip_steps 的原因之一。
    model = torch.compile(raw_model) if args.compile else raw_model

    # 切换到训练模式。这个模型里 act_ckpt 会根据它判断是否启用（见 model.py 的 GPT.forward）
    model.train()
    batches = Batches(args.data, B, T, device)

    # 把这次实验的完整配置打印在日志开头，日志本身就能说明它是怎么跑出来的
    print("=" * 72)
    print(f"设备        {device_name(device)}  (bf16 峰值 {fmt(peak_tflops, '.0f')} TFLOPS)")
    print(f"模型        {args.model}  参数 {n_params / 1e6:.1f}M  层数 {cfg.n_layer}  隐藏维度 {cfg.n_embd}  词表 {cfg.vocab_size}")
    print(f"批次        B={B}  T={T}  grad_accum={args.grad_accum}  → 每步 {tokens_per_step:,} tokens")
    print(f"开关        tf32={args.tf32} bf16={args.bf16} compile={args.compile} attn={args.attn} "
          f"adamw={args.adamw} act_ckpt={args.act_ckpt}")
    print(f"估算        每 token {flops_per_tok / 1e9:.2f} GFLOPs；参数+梯度+优化器状态 ≈ {static_gb:.2f} GB")
    print("=" * 72)

    # ---- 学习率调度 ----
    # 前 10% 的步数用来 warmup
    lr_warmup = max(1, args.steps // 10)

    def get_lr(step):
        # 线性 warmup，然后余弦衰减到 10%
        #
        # 为什么要 warmup：刚开始参数是随机的，AdamW 对梯度大小的估计还不准，
        #   这时用大学习率很容易一步跨太远、训练直接发散。所以先从小学习率慢慢升上去。
        # 为什么要衰减：前期用大步长快速下降，后期用小步长精细收敛。
        #   余弦曲线让学习率平滑下降，最后停在最大值的 10%。
        #
        # 对测速度来说学习率不影响结果；加上它是为了让训练过程和真实训练一致，loss 能正常下降。
        if step < lr_warmup:
            return args.lr * (step + 1) / lr_warmup
        progress = (step - lr_warmup) / max(1, args.steps - lr_warmup)  # 从 0 走到 1
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    # ---- 一步训练 ----
    # 对应 learn/03_train_loop.py 的五步：取数据 → 前向 → 算 loss → 反向 → 更新
    def train_step(step):
        # 手动设置这一步的学习率（这里没有用 PyTorch 自带的 scheduler，直接改更直观）
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 清空上一步的梯度（梯度默认会累加，见 learn/02_autograd.py）。
        # set_to_none=True：直接把梯度设成 None，而不是填 0。
        # 省掉一次“把显存写成 0”的操作，反向时会重新分配。
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        # 梯度累积：跑 grad_accum 个小 batch，梯度一直累加，最后只更新一次参数。
        # 效果等于用一个 grad_accum 倍大的 batch 训练，但显存只需要一个小 batch 的量。
        for _ in range(args.grad_accum):
            x, y = batches.next()  # 1. 取数据
            # 2+3. 前向并算 loss。只有前向放在 autocast 里；
            #      反向会自动沿用前向时每一步的数据类型，不需要也不应该放进 autocast。
            with autocast():
                _, loss = model(x, y)
            # 为什么要除以 grad_accum：cross_entropy 算的是“一个小 batch 内的平均 loss”。
            # 如果直接把 grad_accum 个小 batch 的梯度加起来，梯度会大 grad_accum 倍。
            # 先除一下，累加后的梯度就正好等于“大 batch 的平均梯度”。
            loss = loss / args.grad_accum  # 累积 grad_accum 次，相当于一个大 batch 的平均
            # 只是为了打印日志而累加 loss 的值。
            # detach()：断开和计算图的联系。否则每一步的计算图都会被一直引用，无法释放，显存会越用越多。
            # 注意这里没有调用 .item()：.item() 会强制 CPU 等 GPU 算完，打断 GPU 的流水线。
            loss_accum += loss.detach()
            loss.backward()  # 4. 反向：算出所有参数的梯度，累加到 .grad 上

        # 梯度裁剪：先算出所有参数梯度合在一起的总长度（L2 范数）。
        # 如果超过 1.0，就把所有梯度按同一比例缩小，让总长度等于 1.0。
        # 为什么需要：偶尔会遇到一个“坏 batch”，算出特别大的梯度。
        # 如果照着更新，一步就可能把模型参数推到很糟糕的地方（loss 突然飙升）。
        # 返回值是裁剪【之前】的范数，打印出来可以观察训练是否健康：
        # 训练刚开始时 norm 常常很大（几十），之后会逐渐降到 1 左右。
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

        optimizer.step()  # 5. 更新参数：AdamW 根据梯度调整每个参数
        # 返回的 loss_accum 和 norm 仍然是 GPU 上的张量，由调用方决定什么时候取出数值
        return loss_accum, norm, lr

    # Profiling 模式走另一条路：采样几步、导出时间线后直接结束，不写 results.csv
    if args.profile:
        run_profile(args, device, train_step)
        return

    # 先把配置部分填好；结果部分（tokens/s、MFU 等）等训练完再补上。
    # 如果中途 OOM，也用这一行记录下来，status 写成 "OOM"。
    row = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tag": args.tag, "status": "ok",
        "gpu": device_name(device), "model": args.model, "params_M": round(n_params / 1e6, 1),
        "batch": B, "seq_len": T, "grad_accum": args.grad_accum, "vocab": cfg.vocab_size,
        "tf32": args.tf32, "bf16": args.bf16, "compile": args.compile, "attn": args.attn,
        "adamw": args.adamw, "act_ckpt": args.act_ckpt, "static_mem_gb": round(static_gb, 2),
    }

    # ---- 训练循环 + 计时 ----
    # 计时器会自动排除前 skip_steps 步（见 perf.py 的 StepTimer）
    timer = perf.StepTimer(device.type, args.skip_steps)
    # 从这里开始记录显存峰值。模型参数此时已经在显存里了，也会算进峰值。
    perf.reset_peak_memory(device.type)
    losses = []
    try:
        for step in range(args.steps):
            # start() 和 stop() 内部都会先调用 synchronize，等 GPU 把手上的活干完。
            # 原因：GPU 是异步执行的，Python 代码只是把任务“排进队列”就立刻返回了。
            # 不同步的话，测到的只是“排队花了多久”，而不是“GPU 实际算了多久”。
            timer.start()
            loss, norm, lr = train_step(step)
            dt = timer.stop(tokens_per_step)

            # .item() 把 GPU 上的数字取回 CPU。它会强制等待 GPU，
            # 但放在计时结束之后，所以不影响测速结果。
            losses.append(loss.item())

            if step % args.log_interval == 0 or step == args.steps - 1:
                tps = tokens_per_step / dt  # 这一步的瞬时速度
                mfu = perf.compute_mfu(tps, flops_per_tok, peak_tflops)
                note = "  (预热，不计入统计)" if step < args.skip_steps else ""
                print(f"step {step:4d} | loss {losses[-1]:.4f} | lr {lr:.2e} | norm {norm.item():.2f} | "
                      f"{dt * 1000:8.1f} ms | {tps:9.0f} tok/s | MFU {pct(mfu):>6}{note}")
    except torch.cuda.OutOfMemoryError:
        # 显存不够（OOM）也是一种实验结果，比如实验 9 就是故意测试 batch 加大后会不会 OOM。
        # 把它记进 CSV，结果表里就能看到“这个配置跑不起来”。
        # 退出码设为 1，run_experiments.sh 会提示失败，然后接着跑下一个实验。
        print("\n[OOM] 显存不够。可以减小 --batch_size，或打开 --bf16 / --attn sdpa / --act_ckpt")
        append_result(args.results, {**row, "status": "OOM"})
        sys.exit(1)

    # ---- 汇总结果 ----
    tps = timer.tokens_per_sec()  # 预热之后所有步的平均速度，比单步速度更稳定
    mfu = perf.compute_mfu(tps, flops_per_tok, peak_tflops) if tps else None
    peak_gb = perf.peak_memory_gb(device.type)
    # 单步的 loss 波动很大，取最后 10 步的平均更稳定。
    # 用途：比较不同实验的 loss 是否接近，确认优化只改变了速度，没有改变训练结果。
    final_loss = sum(losses[-10:]) / len(losses[-10:])

    print("-" * 72)
    print(f"吞吐        {fmt(tps, ',.0f')} tokens/s  （第 {args.skip_steps} 步之后的平均）")
    print(f"MFU         {pct(mfu)}" + ("" if mfu is not None else "  （未知显卡峰值，可用 --peak_tflops 指定）"))
    if peak_gb is not None:
        print(f"峰值显存    {peak_gb:.2f} GB  （其中参数+梯度+优化器状态约 {static_gb:.2f} GB，其余主要是激活值）")
    else:
        print("峰值显存    n/a  （只在 CUDA 上统计）")
    print(f"最终 loss   {final_loss:.4f}  （最后 10 步平均，用来确认优化没有改变训练结果）")

    # 把结果部分补进 row，追加到 CSV
    append_result(args.results, {
        **row,
        "tokens_per_s": round(tps) if tps else "",
        "mfu_pct": round(mfu * 100, 2) if mfu else "",
        "peak_mem_gb": round(peak_gb, 2) if peak_gb is not None else "",
        "final_loss": round(final_loss, 4),
    })
    print(f"结果已追加到 {args.results}")


# ==============================================================================
# Profiling：找出时间都花在了哪些操作上
# ==============================================================================
# 测速只能告诉你“一步要多久”，profiling 能告诉你“这些时间具体花在了哪里”。
# 比如对比 naive 和 sdpa 两种 attention，能直接看到哪些算子消失了、哪些变快了。
def run_profile(args, device, train_step):
    # 只有 profiling 模式才需要，所以放在函数里面导入
    from torch.profiler import ProfilerActivity, profile, schedule

    # 先跑几步预热，确保 torch.compile 编译完成，不把编译时间混进 profile
    print(f"预热 {args.skip_steps} 步…")
    for step in range(args.skip_steps):
        train_step(step)

    # 记录哪些活动：
    #   CPU  —— Python 代码和 PyTorch 调度算子花的时间
    #   CUDA —— GPU 上每个 kernel 实际执行的时间
    # 两边对比能看出瓶颈在哪：如果 GPU 经常空闲、在等 CPU 下发任务，
    # 说明瓶颈在 CPU 端（比如 tiny 模型），这时 torch.compile 这类减少 Python 开销的优化最有用。
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # 按“self 时间”排序：只算这个操作自己花的时间，不包括它内部调用的子操作。
    # 比如 aten::linear 内部会调用 aten::addmm，按总时间排序会把同一段时间算两次。
    if device.type == "cuda":
        # PyTorch 2.4 起改名为 self_device_time_total，旧版本仍是 self_cuda_time_total
        from torch.autograd.profiler_util import FunctionEvent
        sort_key = "self_device_time_total" if hasattr(FunctionEvent, "self_device_time_total") else "self_cuda_time_total"
    else:
        sort_key = "self_cpu_time_total"
    trace_dir = os.path.join(ROOT, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"{args.tag}.json")

    # 记录结束后自动调用：打印耗时最多的 15 个算子，并导出时间线文件。
    # 时间线文件用 https://ui.perfetto.dev 打开，能看到 CPU 和 GPU 上每个操作的先后顺序和耗时。
    def on_trace_ready(prof):
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
        prof.export_chrome_trace(trace_path)
        print(f"\ntrace 已导出到 {trace_path}，用 https://ui.perfetto.dev 打开查看时间线")

    # wait 1 步、warmup 1 步，然后正式记录 3 步
    #   wait：这一步完全不记录
    #   warmup：profiler 已经启动但丢弃结果。profiler 刚启动时自身有额外开销，会让第一步的数据失真
    #   active：正式记录这 3 步
    # 一共 1 + 1 + 3 = 5 步，所以下面循环 5 次。
    # record_shapes=True：同时记录每个算子的输入形状，方便分辨是哪一个矩阵乘法，
    #   比如输出层 lm_head 的那个就特别大。
    with profile(activities=activities, schedule=schedule(wait=1, warmup=1, active=3),
                 on_trace_ready=on_trace_ready, record_shapes=True) as prof:
        for i in range(5):
            train_step(args.skip_steps + i)
            prof.step()  # 告诉 profiler “一步结束了”，它据此切换 wait / warmup / active 阶段


# 只有直接运行 python train.py 时才执行 main()；被其他文件 import 时不执行
if __name__ == "__main__":
    main()
