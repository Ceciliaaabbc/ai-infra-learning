"""
性能测量工具：tokens/s、MFU、显存。
"""

import time

import torch

# 各显卡 bf16 Tensor Core 理论峰值（dense，不含稀疏加速），单位 TFLOPS。
# 按顺序做子串匹配，所以更具体的名字要放在前面（如 "H100 PCIe" 在 "H100" 前）。
# 表里没有的显卡请用 --peak_tflops 手动指定（查 NVIDIA 官方规格表）。
PEAK_BF16_TFLOPS = [
    ("H100 PCIe", 756),
    ("H100", 989),
    ("H800 PCIe", 756),
    ("H800", 989),
    ("H200", 989),
    ("H20", 148),
    ("A100", 312),
    ("A800", 312),
    ("A10", 125),
    ("A40", 150),
    ("RTX A6000", 155),
    ("L40S", 362),
    ("L40", 181),
    ("L20", 120),
    ("L4", 121),
    ("RTX 4090 D", 147),
    ("RTX 4090", 165),
    ("RTX 3090", 71),
]


def lookup_peak_tflops(device_name):
    for key, tflops in PEAK_BF16_TFLOPS:
        if key in device_name:
            return tflops
    return None


def count_params(model):
    """返回 (总参数量, 参与矩阵乘法的参数量)。

    计算 FLOPs 时不算 position embedding：它只是查表，不做矩阵乘法。
    token embedding 和 lm_head 共享权重，lm_head 是一次真正的矩阵乘法，所以要算。
    """
    total = sum(p.numel() for p in model.parameters())
    matmul = total - model.transformer.wpe.weight.numel()
    return total, matmul


def flops_per_token(n_matmul_params, n_layer, n_embd, seq_len):
    """训练时每个 token 的 FLOPs（前向 + 反向），采用 PaLM 论文附录 B 的估算方法。

    6N：每个参数在前向做 1 次乘加（2 FLOPs），反向是前向的 2 倍，合计 2 + 4 = 6。
    12·L·T·d：attention 里 QK^T 和 att@V 这两个矩阵乘法，它们不对应任何参数，要单独算。
    注意：开启 activation checkpointing 后硬件实际多做了一次前向，
    但 MFU 的定义只算“模型本身需要的 FLOPs”，重算的部分不计入。
    """
    return 6 * n_matmul_params + 12 * n_layer * seq_len * n_embd


def compute_mfu(tokens_per_sec, flops_per_tok, peak_tflops):
    if not peak_tflops:
        return None
    return tokens_per_sec * flops_per_tok / (peak_tflops * 1e12)


def static_memory_bytes(n_params):
    """参数 + 梯度 + AdamW 状态，每个参数 16 字节。

    混合精度（autocast）下权重本身仍是 fp32，所以无论是否开 bf16 都是：
      fp32 权重 4B + fp32 梯度 4B + Adam 一阶矩 m 4B + 二阶矩 v 4B = 16B
    实测峰值显存 − 这个数 ≈ 激活值 + 临时缓冲区。
    """
    return 16 * n_params


def synchronize(device_type):
    # GPU 是异步执行的：Python 代码跑完不代表 GPU 算完了。计时前必须同步。
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device_type):
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device_type):
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return None


class StepTimer:
    """记录每一步的耗时和 token 数。前 warmup 步不计入平均值（torch.compile 编译、CUDA 初始化都在这里）。"""

    def __init__(self, device_type, warmup):
        self.device_type = device_type
        self.warmup = warmup
        self.records = []  # (dt 秒, tokens)
        self._t0 = None

    def start(self):
        synchronize(self.device_type)
        self._t0 = time.perf_counter()

    def stop(self, tokens):
        synchronize(self.device_type)
        dt = time.perf_counter() - self._t0
        self.records.append((dt, tokens))
        return dt

    def tokens_per_sec(self):
        measured = self.records[self.warmup:]
        if not measured:
            return None
        total_time = sum(dt for dt, _ in measured)
        total_tokens = sum(n for _, n in measured)
        return total_tokens / total_time
