"""
把 results_ddp.csv 打印成 Markdown 表格，并计算扩展效率。

  python report_ddp.py                    # 读取 results_ddp.csv
  python report_ddp.py my_results.csv

扩展效率 = 多卡时每张卡的速度 ÷ 单卡 baseline 的速度。
  100%：加卡之后，每张卡一点都没变慢，总速度随卡数成倍增长（理想情况）
  越低：说明通信等额外开销越大，加卡带来的收益越少
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_ddp.csv")
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}，先运行 train_ddp.py 或 run_ddp.sh")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # 单卡 baseline：同一种显卡、同一个模型、同样的每卡 batch 和序列长度下，第一个 1 卡的实验
    baselines = {}
    for r in rows:
        if r["status"] == "ok" and r["world_size"] == "1" and r["tokens_per_s_per_gpu"]:
            baselines.setdefault((r["gpu"], r["model"], r["batch_per_gpu"], r["seq_len"]), float(r["tokens_per_s_per_gpu"]))

    print("| 实验 | 卡数 | 梯度同步 | 总 tokens/s | 每卡 tokens/s | 扩展效率 | 每步通信 (ms) | MFU | 峰值显存 (GB) | 参数一致 | loss |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        if r["status"] != "ok":
            print(f"| {r['tag']} | {r['world_size']} | {r['ddp_impl']} | **{r['status']}** | | | | | | | |")
            continue
        per_gpu = float(r["tokens_per_s_per_gpu"])
        base = baselines.get((r["gpu"], r["model"], r["batch_per_gpu"], r["seq_len"]))
        eff = f"{per_gpu / base * 100:.0f}%" if base else ""
        mfu = f"{float(r['mfu_pct']):.1f}%" if r["mfu_pct"] else "n/a"
        comm = r["comm_ms"] or "—"
        mem = r["peak_mem_gb"] or "n/a"
        sync = "✓" if r["params_in_sync"] == "True" else "✗"
        print(f"| {r['tag']} | {r['world_size']} | {r['ddp_impl']} | {float(r['tokens_per_s']):,.0f} | {per_gpu:,.0f} | "
              f"{eff} | {comm} | {mfu} | {mem} | {sync} | {r['final_loss']} |")


if __name__ == "__main__":
    main()
