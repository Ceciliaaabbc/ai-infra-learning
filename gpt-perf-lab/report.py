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

    # 以第一个成功的实验作为 baseline
    baseline = next((float(r["tokens_per_s"]) for r in rows if r["status"] == "ok" and r["tokens_per_s"]), None)

    print("| 实验 | 显卡 | B | tokens/s | 相对 baseline | MFU | 峰值显存 (GB) | loss |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["status"] != "ok":
            print(f"| {r['tag']} | {r['gpu']} | {r['batch']} | **{r['status']}** | | | | |")
            continue
        tps = float(r["tokens_per_s"])
        speedup = f"{tps / baseline:.2f}x" if baseline else ""
        mfu = f"{float(r['mfu_pct']):.1f}%" if r["mfu_pct"] else "n/a"
        mem = r["peak_mem_gb"] or "n/a"
        print(f"| {r['tag']} | {r['gpu']} | {r['batch']} | {tps:,.0f} | {speedup} | {mfu} | {mem} | {r['final_loss']} |")


if __name__ == "__main__":
    main()
