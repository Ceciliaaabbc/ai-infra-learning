# AI Infra 学习记录

从单卡训练出发，在同一份代码上逐步扩展到多卡、自定义算子和视频模型。

| 项目 | 内容 | 状态 |
|---|---|---|
| [项目 0：GPT 单卡性能实验](gpt-perf-lab/) | 单卡训练 GPT-2 (124M)，逐项加优化，测量 tokens/s、MFU、显存 | 进行中 |
| 项目 2：手写分布式训练 | DDP → ZeRO → Tensor Parallel | 计划中 |
| 项目 1：Triton 算子 | 针对 profiling 发现的瓶颈写 fused kernel | 计划中 |
| 项目 3：视频 DiT 训练系统 | VAE latent 缓存、高效数据加载、序列并行 | 计划中 |
