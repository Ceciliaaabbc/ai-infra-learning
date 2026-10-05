"""
GPU 环境自检：在 GPU 服务器上第一个运行的脚本。

  python check_gpu.py

逐项检查实验需要的 CUDA 功能（显存统计、bf16、TF32、FlashAttention、fused AdamW），
并实测矩阵乘法的算力，和 perf.py 里的理论峰值做对比。
"""

import sys
import time

import torch
import torch.nn.functional as F

import perf

results = []  # (状态, 检查项, 说明)


def check(name, fn):
    try:
        ok, note = fn()
        results.append(("[OK]  " if ok else "[注意]", name, note))
    except Exception as e:
        results.append(("[失败]", name, f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"))


def matmul_tflops(dtype, size=8192, iters=10):
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * size**3 * iters / (time.perf_counter() - t0) / 1e12


def main():
    print(f"PyTorch {torch.__version__}")
    if not torch.cuda.is_available():
        sys.exit("没有检测到 CUDA GPU。请在带 NVIDIA 显卡的机器上运行。")

    name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    peak = perf.lookup_peak_tflops(name)
    print(f"CUDA {torch.version.cuda}  |  {name}  |  计算能力 sm{cap[0]}{cap[1]}  |  显存 {total_gb:.1f} GB")
    print("=" * 72)

    def mem_stats():
        torch.cuda.reset_peak_memory_stats()
        x = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")  # 256MB
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        del x
        return peak_gb >= 0.25, f"分配 256MB 后峰值显示 {peak_gb:.2f} GB"
    check("显存统计", mem_stats)

    check("峰值算力表", lambda: (peak is not None,
          f"{peak} TFLOPS (bf16)" if peak else "表里没有这张卡，跑实验时请加 --peak_tflops <官方 bf16 dense 峰值>"))

    check("bf16", lambda: (torch.cuda.is_bf16_supported(),
          "支持" if torch.cuda.is_bf16_supported() else "不支持，实验 2 之后的 bf16 结果没有意义"))

    check("TF32", lambda: (cap >= (8, 0),
          "支持（Ampere 及以后的显卡）" if cap >= (8, 0) else "需要 sm80 以上，实验 1 不会有提升"))

    def flash():
        q = torch.randn(1, 12, 1024, 64, device="cuda", dtype=torch.bfloat16)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # PyTorch >= 2.3
            ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except ImportError:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        with ctx:
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        return True, "可用，--attn sdpa 会走 FlashAttention"
    check("FlashAttention", flash)

    def fused_adamw():
        p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
        p.grad = torch.randn_like(p)
        torch.optim.AdamW([p], lr=1e-3, fused=True).step()
        return True, "可用"
    check("fused AdamW", fused_adamw)

    def compile_check():
        fn = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)
        fn(torch.randn(1024, device="cuda"))
        return True, "可用（首次编译需要几十秒属于正常）"
    check("torch.compile", compile_check)

    for status, item, note in results:
        print(f"{status} {item:16s} {note}")

    print("=" * 72)
    print("实测 8192×8192 矩阵乘法算力（理论峰值是上限，能达到 60%–90% 都算正常）：")
    for label, dtype, precision in [("fp32（关闭 TF32）", torch.float32, "highest"),
                                    ("fp32（开启 TF32）", torch.float32, "high"),
                                    ("bf16", torch.bfloat16, "highest")]:
        torch.set_float32_matmul_precision(precision)
        try:
            tf = matmul_tflops(dtype)
            ratio = f"  = bf16 峰值的 {tf / peak * 100:.0f}%" if peak else ""
            print(f"  {label:16s} {tf:7.1f} TFLOPS{ratio}")
        except Exception as e:
            print(f"  {label:16s} 失败：{type(e).__name__}")
    torch.set_float32_matmul_precision("highest")

    failed = [item for status, item, _ in results if status != "[OK]  "]
    print("=" * 72)
    if failed:
        print(f"需要留意的项：{', '.join(failed)}。完整实验建议用 RTX 3090 / 4090 / A100 / H100 这类 sm80 以上的显卡。")
    else:
        print("全部通过，可以开始实验：bash run_experiments.sh")


if __name__ == "__main__":
    main()
