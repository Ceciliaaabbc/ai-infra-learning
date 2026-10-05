# GPT 单卡性能实验（项目 0）

**目标**：在单张 GPU 上训练 GPT-2 (124M)，从纯 fp32 的 baseline 出发，逐项加上优化，
**用数据**说明每一项优化让速度提升了多少、显存省了多少、为什么有效。

参考：Karpathy 的视频 *Let's reproduce GPT-2 (124M)* 和仓库 `build-nanogpt`。

## 实验结果（RTX 4090）

**吞吐提升 5.05 倍（23,653 → 119,443 tokens/s），MFU 从 12.3% 提升到 61.9%，同 batch 下峰值显存降低 54%。**

| 实验 | tokens/s | 比 baseline | MFU | 峰值显存 (GB) |
|---|---:|---:|---:|---:|
| fp32 baseline（B=4） | 23,653 | 1.00x | 12.3% | 9.03 |
| + TF32 | 27,973 | 1.18x | 14.5% | 9.03 |
| + bf16 | 35,930 | 1.52x | 18.6% | 8.63 |
| + torch.compile | 65,584 | 2.77x | 34.0% | 6.43 |
| + FlashAttention | 84,502 | 3.57x | 43.8% | 4.16 |
| + 词表补齐到 50304 | 83,888 | 3.55x | 43.5% | 4.17 |
| + fused AdamW | 96,784 | 4.09x | 50.2% | 4.17 |
| batch 加到 16 | **119,443** | **5.05x** | **61.9%** | 11.47 |

**核心发现**：用 Amdahl 定律分析 TF32 的结果，baseline 中矩阵乘法只占约 40% 的时间，瓶颈在显存读写。所以提升最大的是减少显存读写的 compile（×1.83）和 FlashAttention（×1.29）。

完整分析（含 activation checkpointing 的取舍、OOM 原因、词表补齐为什么无效等）见 [results/rtx4090/README.md](results/rtx4090/README.md)。

## 目录结构

```
gpt-perf-lab/
├── learn/                  # 前置知识：可直接运行的小例子（CPU 就能跑）
│   ├── 01_tensor.py        #   tensor、shape、dtype、矩阵乘法
│   ├── 02_autograd.py      #   自动求导、计算图、梯度累加
│   ├── 03_train_loop.py    #   完整的训练循环
│   └── 04_attention.py     #   attention 一步一步算
├── check_gpu.py            # GPU 环境自检：到服务器上第一个运行它
├── model.py                # GPT-2 模型（带 naive / sdpa attention 和 activation checkpointing 开关）
├── perf.py                 # 测量工具：tokens/s、MFU、显存估算
├── train.py                # 训练 + 测量，结果追加到 results.csv
├── prepare_data.py         # 下载数据并转成 token
├── run_experiments.sh      # 一键跑完整个优化阶梯
├── report.py               # 把 results.csv 打印成 Markdown 表格
└── results/                # 各显卡的实验结果和分析
    └── rtx4090/
```

## 1. 先学前置知识

```bash
python learn/01_tensor.py
python learn/02_autograd.py
python learn/03_train_loop.py
python learn/04_attention.py
```

把每个脚本的输出和代码对照着看懂，再去读 `model.py`。

## 2. 环境

**本地冒烟测试**（Mac / 没有 GPU 的电脑也能跑，只用来确认代码能运行）：

```bash
pip install -r requirements.txt
python train.py --model tiny --data synthetic --steps 20 --skip_steps 5
```

**GPU 服务器**（比如 AutoDL 租一张 4090 或 A100，镜像选 PyTorch 2.x）：

```bash
pip install -r requirements.txt
python check_gpu.py             # 自检：显存统计、bf16、TF32、FlashAttention、fused AdamW、实测算力
python train.py --model tiny --data synthetic --steps 20 --skip_steps 5   # 冒烟测试
python prepare_data.py          # 下载 tiny shakespeare（约 1MB）
```

用完记得在控制台**关机**，按量计费的实例开着就一直收费。

如果在 AutoDL 上下载 GitHub 的文件失败，先执行 `source /etc/network_turbo` 打开学术加速；
或者直接用 `--data synthetic`（随机 token），测速度不受影响。

## 3. 跑实验

**跑单个实验**：

```bash
python train.py --tag my_baseline                               # 纯 fp32 baseline
python train.py --tf32 --bf16 --compile --attn sdpa --tag fast  # 打开几项优化
```

**跑完整的优化阶梯**（约 15–20 分钟）：

```bash
bash run_experiments.sh                       # 默认 B=4，适合 24GB 显卡
BATCH=8 BIG_BATCH=32 bash run_experiments.sh  # 80GB 显卡可以调大
python report.py                              # 打印结果表格
```

| # | 实验 | 加了什么 |
|---|---|---|
| 0 | baseline_fp32 | 什么都不加 |
| 1 | tf32 | fp32 矩阵乘法改用 TF32 Tensor Core |
| 2 | bf16 | bf16 混合精度 |
| 3 | compile | `torch.compile` |
| 4 | flash_attn | `F.scaled_dot_product_attention`（FlashAttention） |
| 5 | vocab_50304 | 词表从 50257 补齐到 50304 |
| 6 | fused_adamw | fused AdamW |
| 7–10 | big_batch / ckpt | 增大 batch、activation checkpointing，体会显存和速度的取舍 |

### 每个实验改了代码的哪里

所有实验共用同一份 `model.py` 和 `train.py`。不同的写法都写在代码里，用 `if ... else ...` 分成两条路，由命令行开关决定走哪一条。

实验是**一层层叠加**的：每个实验保留前面所有开关，再多打开一个。表里的“新增开关”只列出比上一步多出来的那一个。每个开关都经过三个位置：**① 在哪一行下达 → ② 在哪一行被读进来 → ③ 在哪一行真正起作用**。

| 实验 | 新增开关 | ① 命令所在行 | ② 读取参数的行 | ③ 真正起作用的行 |
|---|---|---|---|---|
| 0 baseline | 无，全部使用默认值 | [run_experiments.sh:35](run_experiments.sh#L35) | — | 见表格下方的说明 |
| 1 TF32 | `--tf32` | [run_experiments.sh:36](run_experiments.sh#L36) | [train.py:108](train.py#L108) | [train.py:245](train.py#L245) `set_float32_matmul_precision` |
| 2 bf16 | `--bf16` | [run_experiments.sh:37](run_experiments.sh#L37) | [train.py:109](train.py#L109) | [train.py:261](train.py#L261) `torch.autocast` |
| 3 compile | `--compile` | [run_experiments.sh:38](run_experiments.sh#L38) | [train.py:110](train.py#L110) | [train.py:301](train.py#L301) `torch.compile` |
| 4 FlashAttention | `--attn sdpa` | [run_experiments.sh:39](run_experiments.sh#L39) | [train.py:111](train.py#L111) | 经 [train.py:270](train.py#L270) 传入模型，在 [model.py:172](model.py#L172) 判断，[model.py:177](model.py#L177) 执行 `F.scaled_dot_product_attention` |
| 5 词表补到 50304 | `--vocab_size 50304` | [run_experiments.sh:40](run_experiments.sh#L40) | [train.py:92](train.py#L92) | 经 [train.py:269](train.py#L269) 传入模型，[model.py:286](model.py#L286)（`wte`）和 [model.py:304](model.py#L304)（`lm_head`）的大小随之改变 |
| 6 fused AdamW | `--adamw fused` | [run_experiments.sh:41](run_experiments.sh#L41) | [train.py:112](train.py#L112) | 经 [train.py:292](train.py#L292) 传入，[model.py:411](model.py#L411) `fused=True` |
| 7 batch 16 | `--batch_size 16` | [run_experiments.sh:44](run_experiments.sh#L44) | [train.py:88](train.py#L88) | [train.py:274](train.py#L274) `B = args.batch_size`，取数据时用在 [train.py:187](train.py#L187) |
| 8 checkpointing | `--act_ckpt` | [run_experiments.sh:45](run_experiments.sh#L45) | [train.py:113](train.py#L113) | 经 [train.py:270](train.py#L270) 传入模型，在 [model.py:353](model.py#L353) 判断，[model.py:362](model.py#L362) 执行 `checkpoint(block, x)` |
| 9 batch 32 | `--batch_size 32` | [run_experiments.sh:46](run_experiments.sh#L46) | [train.py:88](train.py#L88) | 同实验 7 |
| 10 batch 32 + checkpointing | `--batch_size 32 --act_ckpt` | [run_experiments.sh:47](run_experiments.sh#L47) | [train.py:88](train.py#L88)、[train.py:113](train.py#L113) | 同实验 7 和实验 8 |

**说明：**

1. **实验 0 没有加任何开关**，所以每个判断都走默认那条路：
   - [train.py:245](train.py#L245)：选 `"highest"`，也就是纯 fp32 计算。
   - [train.py:261](train.py#L261)：`if` 条件不成立，不开 bf16。
   - [model.py:172](model.py#L172)：走下面的 `else` 分支，用手写的四步 attention。
   - [model.py:411](model.py#L411)：用 foreach 版本的优化器。
2. **从实验 6 开始，命令里出现了 `$OPTS`。** 它在 [run_experiments.sh:24](run_experiments.sh#L24) 定义，就是把前面所有开关加上 `--adamw fused` 打包成一个变量。
3. **batch 的数字在脚本开头设置。** `$BATCH`（默认 4）在 [run_experiments.sh:14](run_experiments.sh#L14)，`$BIG_BATCH`（默认 16）在 [run_experiments.sh:15](run_experiments.sh#L15)。实验 9 和 10 用的 `$((BIG_BATCH * 2))` 就是 32。
4. **有些开关要先经过 `train.py`，再交给 `model.py`。** `--attn`、`--act_ckpt`、`--vocab_size` 这三个，是在 [train.py:269](train.py#L269) 写进模型的规格表 `GPTConfig`，模型创建之后再按规格表上的值选择走哪条路。
5. 行号对应的是当前版本的代码。以后改代码导致行号变化时，可以用表里写的关键代码（比如 `torch.compile`）在文件里搜索。

## 4. 指标说明

| 指标 | 含义 |
|---|---|
| **tokens/s** | 每秒训练多少个 token。跳过前 `--skip_steps` 步，因为编译、预热会让前几步特别慢 |
| **MFU** | `tokens/s × 每 token FLOPs ÷ 显卡峰值 FLOPs`，表示用上了显卡理论算力的百分之几。公式见 `perf.py` |
| **峰值显存** | `torch.cuda.max_memory_allocated()`。减去“参数+梯度+优化器状态”（每参数 16 字节），剩下的主要是激活值 |
| **loss** | 最后 10 步的平均。用来确认优化**没有改变训练结果**（实验 2 之后数值会有细微差别，属于正常的精度差异） |

如果显卡不在 `perf.py` 的列表里，用 `--peak_tflops` 手动指定 bf16 峰值算力。

## 5. Profiling：时间都花在哪了

```bash
python train.py --tf32 --bf16 --attn sdpa --profile --tag prof_sdpa
```

终端会打印耗时最多的 15 个算子，同时导出 `traces/prof_sdpa.json`，
用 https://ui.perfetto.dev 打开可以看到 CPU 和 GPU 的时间线。

## 6. 练习清单

- [ ] 跑完优化阶梯，把 `report.py` 的表格贴进自己的博客
- [ ] **手算显存**：124M 参数 × 16 字节 ≈ ? GB；和实测峰值相差多少？差的部分是什么？
- [ ] 对比实验 3 和 4 的峰值显存：FlashAttention 省下了多少？和 `learn/04_attention.py` 的 T² 估算对得上吗？
- [ ] 对比实验 7 和 8：activation checkpointing 让速度慢了多少、显存省了多少？
- [ ] 实验 9 是不是 OOM 了？实验 10 为什么能跑？
- [ ] 用 profile 对比 naive 和 sdpa 的 attention：哪些算子消失了？
- [ ] 改 `--seq_len`（256 / 512 / 1024），看 naive attention 的显存怎么变化

**思考题**（这些也是面试常见题）：

1. bf16 为什么比 fp32 快？它和 fp16 有什么区别？可能带来什么风险？
2. `torch.compile` 到底做了什么？为什么前几步特别慢？
3. FlashAttention 为什么能省显存，而且还更快？
4. 词表从 50257 改成 50304 为什么会变快？
5. MFU 为什么很难达到 100%？你的瓶颈在哪里？

## 7. 下一步

做完这个项目后，在同一份代码上继续：

1. **项目 2**：把训练扩展到多卡，手写 DDP → ZeRO → Tensor Parallel
2. **项目 1**：针对 profiling 发现的瓶颈，用 Triton 写 fused 算子替换掉
3. **项目 3**：把模型换成 DiT、数据换成视频，加上序列并行
