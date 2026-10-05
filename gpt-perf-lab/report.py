"""
把 results.csv 打印成 Markdown 表格，可以直接贴进博客或 README。

  python report.py                 # 读取 results.csv
  python report.py my_results.csv
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results.csv")
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}，先运行 train.py 或 run_experiments.sh")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # 每个（显卡, 模型）组合各自以第一个成功的实验作为 baseline，
    # 这样冒烟测试的 tiny 模型不会被当成 gpt2 实验的 baseline
    baselines = {}
    for r in rows:
        if r["status"] == "ok" and r["tokens_per_s"]:
            baselines.setdefault((r["gpu"], r["model"]), float(r["tokens_per_s"]))

    print("| 实验 | 显卡 | 模型 | B | tokens/s | 相对 baseline | MFU | 峰值显存 (GB) | loss |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["status"] != "ok":
            print(f"| {r['tag']} | {r['gpu']} | {r['model']} | {r['batch']} | **{r['status']}** | | | | |")
            continue
        tps = float(r["tokens_per_s"])
        baseline = baselines.get((r["gpu"], r["model"]))
        speedup = f"{tps / baseline:.2f}x" if baseline else ""
        mfu = f"{float(r['mfu_pct']):.1f}%" if r["mfu_pct"] else "n/a"
        mem = r["peak_mem_gb"] or "n/a"
        print(f"| {r['tag']} | {r['gpu']} | {r['model']} | {r['batch']} | {tps:,.0f} | {speedup} | {mfu} | {mem} | {r['final_loss']} |")


if __name__ == "__main__":
    main()
