# 项目 2 · 阶段 1：数据并行（DDP）

**目标**：把项目 0 的单卡训练扩展到多张卡，用数据回答两个问题：
- 加一张卡，训练速度能翻倍吗？实际快了多少？
- 没能翻倍的那部分时间，花在了哪里？

模型、测量工具和数据读取都直接复用 [项目 0](../gpt-perf-lab/) 的代码，本目录只写多卡相关的部分。

## 数据并行是怎么工作的

```
          卡 0                          卡 1
  完整的模型（参数相同）          完整的模型（参数相同）
  取第 0 份数据                   取第 1 份数据
  前向 + 反向 → 梯度 g0           前向 + 反向 → 梯度 g1
             ╲                      ╱
              all-reduce：求平均 (g0 + g1) / 2
             ╱                      ╲
  用平均梯度更新参数              用平均梯度更新参数
           → 两张卡的参数仍然完全相同
```

打个比方：几个学生做同一套题的不同部分，各自算出“该怎么改”，再开会取平均，每个人都按同样的结论修改自己的笔记，所以大家的笔记始终一样。

## 三种梯度同步方式

`train_ddp.py` 用 `--ddp_impl` 选择：

| 实现 | 做法 | 特点 |
|---|---|---|
| `naive` | 反向结束后，每个参数张量单独做一次 all-reduce | GPT-2 有 148 个参数张量，每步要通信 148 次 |
| `flat` | 把所有梯度拼成一个大张量，只做 1 次 all-reduce | 通信次数降到 1 次，但要多一块临时缓冲区 |
| `torch` | PyTorch 官方的 `DistributedDataParallel` | 梯度分桶，并且一边做反向一边通信 |

## 目录结构

```
distributed-lab/
├── learn/
│   ├── 01_collectives.py   # 4 种集合通信操作，一次看懂
│   └── 02_ddp_math.py      # 验证：各卡梯度求平均 = 用全部数据算出的梯度
├── train_ddp.py            # 数据并行训练 + 测量，结果追加到 results_ddp.csv
├── run_ddp.sh              # 在多卡机器上一键跑完所有实验
└── report_ddp.py           # 打印结果表格，计算扩展效率
```

## 1. 先在 Mac 上学习和验证（免费）

PyTorch 可以在 CPU 上用 gloo 通信库，一台电脑上启动多个进程，模拟“多张卡”。代码写对之后，再租 GPU 测速度。

安装依赖（和项目 0 相同）：

```bash
pip install -r ../gpt-perf-lab/requirements.txt
```

**注意：Mac 上必须加 `--master_addr=127.0.0.1`。** 用 `torchrun --standalone` 会一直卡住，不报错也不输出。

```bash
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 learn/01_collectives.py
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 learn/02_ddp_math.py
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 train_ddp.py --model tiny --data synthetic --steps 20 --skip_steps 5 --ddp_impl naive
```

把最后一条命令里的 `--ddp_impl` 分别换成 `naive`、`flat`、`torch` 各跑一次，检查两件事：
- 三种实现的 loss **每一步都完全一样**：说明梯度同步的结果正确。
- 最后一行显示 **参数一致 ✓**：说明所有进程的参数始终保持相同。

Mac 上测出的速度没有参考价值：几个进程在抢同一个 CPU，多进程反而比单进程慢。

## 2. 在多卡 GPU 上测速度

在 AutoDL 上租一台**同一台机器上有 2 到 4 张卡**的实例。4090 或 3090 适合入门；A100、A800 带 NVLink，更接近真实的训练环境。

先看看显卡之间是怎么连接的，后面分析通信时间会用到：

```bash
nvidia-smi topo -m
```

准备代码和数据：

```bash
git clone https://github.com/Ceciliaaabbc/ai-infra-learning.git
cd ai-infra-learning/gpt-perf-lab && pip install -r requirements.txt && python prepare_data.py
```

跑实验（在后台运行，关掉浏览器也不会中断）：

```bash
cd ../distributed-lab && nohup bash run_ddp.sh > ddp.log 2>&1 &
```

```bash
tail -f ddp.log
```

`run_ddp.sh` 会依次运行：

| 实验 | 说明 |
|---|---|
| `1gpu_torch` | 单卡 baseline，用来计算扩展效率 |
| `2gpu_naive` / `2gpu_flat` / `2gpu_torch` | 2 张卡，三种同步方式对比 |
| `4gpu_*`、`8gpu_*` | 机器上有足够的卡时才会运行 |

每张卡的配置和项目 0 实验 7 相同：B=16，打开全部单卡优化。跑完后运行 `python report_ddp.py` 生成结果表。

如果某张卡显存不够，`torchrun` 会停掉这次实验的所有进程，脚本会提示失败，然后接着跑下一个实验。

## 3. 指标说明

| 指标 | 含义 |
|---|---|
| **总 tokens/s** | 所有卡加起来，每秒训练多少个 token |
| **每卡 tokens/s** | 总速度 ÷ 卡数 |
| **扩展效率** | 每卡速度 ÷ 单卡 baseline 的速度。100% 表示加卡后每张卡一点没变慢，是理想情况 |
| **每步通信** | 手写版本（naive、flat）同步梯度花的时间。torch 版本的通信和反向计算重叠在一起，没法单独测 |
| **参数一致** | 训练结束时，检查所有卡的参数是否完全相同 |

## 4. 练习与思考题

- [ ] 在 Mac 上跑通 `learn/01` 和 `learn/02`，能用自己的话解释 4 种通信操作
- [ ] 在 Mac 上用三种 `--ddp_impl` 跑 `train_ddp.py`，确认 loss 每一步都一样
- [ ] **估算通信量**：每一步要同步的梯度有多大？结合 `nvidia-smi topo -m` 看到的连接方式，估算一次 all-reduce 需要多久，再和实测的“每步通信”对比
- [ ] naive 为什么比 flat 慢？
- [ ] torch 版本是怎么把通信“藏”进反向计算里的？（关键词：bucket、overlap）
- [ ] 扩展效率随卡数怎么变化？为什么卡越多，效率越难保持？
- [ ] 为什么梯度裁剪必须放在梯度同步之后？
- [ ] 为什么训练开始前，要把 0 号卡的参数广播给所有卡？
- [ ] 2 张卡、每张卡 B=16，相当于单卡 B 等于多少？loss 为什么比单卡实验低？

## 下一步：阶段 2（ZeRO / FSDP）

数据并行要求每张卡都放得下一份完整的模型。参数、梯度、优化器状态每个参数共占 16 字节，模型一大，单卡就放不下了。阶段 2 会把这 16 字节分摊到多张卡上。
