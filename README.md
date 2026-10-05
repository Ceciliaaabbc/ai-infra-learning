# AI Infra 学习记录

从单卡训练出发，在同一份代码上逐步扩展到多卡、自定义算子和视频模型。

| 项目 | 内容 | 状态 |
|---|---|---|
| [项目 0：GPT 单卡性能实验](gpt-perf-lab/) | 单卡训练 GPT-2 (124M)，逐项加优化，测量 tokens/s、MFU、显存 | 进行中 |
| 项目 2：手写分布式训练 | DDP → ZeRO → Tensor Parallel | 计划中 |
| 项目 1：Triton 算子 | 针对 profiling 发现的瓶颈写 fused kernel | 计划中 |
| 项目 3：视频 DiT 训练系统 | VAE latent 缓存、高效数据加载、序列并行 | 计划中 |

## 项目 0 阅读顺序

**第 0 步：先把 [learn/](gpt-perf-lab/learn/) 里的 01 到 04 跑一遍并看懂**

后面的代码都建立在这些基础上。

**第 1 步：[README.md](gpt-perf-lab/README.md)（5 分钟）**

先了解项目目标和 11 个实验分别是什么。

**第 2 步：[model.py](gpt-perf-lab/model.py)，从上往下读**

1. `GPTConfig`：先看模型有哪些参数（层数、维度、词表大小）
2. `GPT.forward`：先看整条数据流，从 token 到 loss
3. `Block`：每一层由 attention 和 MLP 组成
4. `CausalSelfAttention.forward`：**重点**。对照 `learn/04_attention.py`，两边是同一套计算
5. `configure_optimizer`：为什么有些参数做 weight decay，有些不做

边读边问自己：这一步的 tensor 形状是什么？不确定的话，就在 forward 里加一行 `print(x.shape)`，跑一下看看。

**第 3 步：[train.py](gpt-perf-lab/train.py)，按执行顺序读 `main()`**

1. 设置精度
2. 创建模型，然后 compile
3. `train_step`：**重点**。它就是 `learn/03` 里的训练五步，多加了混合精度、梯度累积和梯度裁剪
4. 主循环：如何计时、如何打印日志
5. 写入结果
6. 最后看 `Batches`：为什么 x 和 y 要错开一位

**第 4 步：[perf.py](gpt-perf-lab/perf.py)**

- `flops_per_token`：MFU 公式的来源。可以自己手算一下 GPT-2 的结果，应该约等于 0.855 GFLOPs
- `StepTimer`：为什么计时前一定要调用 `synchronize`

**第 5 步：[run_experiments.sh](gpt-perf-lab/run_experiments.sh) 和 [report.py](gpt-perf-lab/report.py)**

这两个很简单，几分钟就能看完。

**第 6 步：最后看 [prepare_data.py](gpt-perf-lab/prepare_data.py) 和 [check_gpu.py](gpt-perf-lab/check_gpu.py)**

它们是辅助工具，不影响理解核心逻辑。
