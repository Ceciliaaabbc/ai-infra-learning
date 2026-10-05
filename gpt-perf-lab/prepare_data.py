"""
准备训练数据：下载文本 → 用 GPT-2 tokenizer 转成 token → 存成 uint16 二进制文件。

  python prepare_data.py                                        # tiny shakespeare，约 1MB，几秒完成
  python prepare_data.py --source fineweb --num_tokens 20000000  # FineWeb-Edu 子集，需要先 pip install datasets

输出：data/<source>/train.bin 和 val.bin
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import tiktoken

ROOT = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download(url, attempts=5):
    for i in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:  # 网络不稳时常见：连接超时、读到一半断开
            print(f"  第 {i} 次下载失败：{e}")
    sys.exit(f"下载失败。可以手动下载 {url} 放到 data/shakespeare/input.txt，"
             "或者先用 --data synthetic 跑实验（AutoDL 上可以先执行 source /etc/network_turbo）")


def write_split(tokens, out_dir, val_frac=0.1):
    # GPT-2 词表 50257 < 65536，用 uint16 存，比 int64 省 4 倍空间
    arr = np.asarray(tokens, dtype=np.uint16)
    n_val = int(len(arr) * val_frac)
    os.makedirs(out_dir, exist_ok=True)
    arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
    arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))
    print(f"写入 {out_dir}：train {len(arr) - n_val:,} tokens，val {n_val:,} tokens")


def prepare_shakespeare(enc):
    out_dir = os.path.join(ROOT, "data", "shakespeare")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "input.txt")
    if not os.path.exists(txt_path):
        print(f"下载 {SHAKESPEARE_URL}")
        data = download(SHAKESPEARE_URL)
        # 先写临时文件再改名，避免中途失败留下半个文件
        with open(txt_path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(txt_path + ".tmp", txt_path)
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    write_split(enc.encode_ordinary(text), out_dir)


def prepare_fineweb(enc, num_tokens):
    from datasets import load_dataset  # 可选依赖，只有用 fineweb 时才需要

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    chunks, total = [], 0
    for doc in ds:
        # 每篇文档前加 <|endoftext|>，让模型知道文档边界
        ids = [enc.eot_token] + enc.encode_ordinary(doc["text"])
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if len(chunks) % 2000 == 0:
            print(f"  已处理 {total:,} / {num_tokens:,} tokens")
        if total >= num_tokens:
            break
    write_split(np.concatenate(chunks)[:num_tokens], os.path.join(ROOT, "data", "fineweb"))


def main():
    p = argparse.ArgumentParser(description="准备训练数据")
    p.add_argument("--source", default="shakespeare", choices=["shakespeare", "fineweb"])
    p.add_argument("--num_tokens", type=int, default=20_000_000, help="fineweb 要取多少 tokens")
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    if args.source == "shakespeare":
        prepare_shakespeare(enc)
    else:
        prepare_fineweb(enc, args.num_tokens)


if __name__ == "__main__":
    main()
