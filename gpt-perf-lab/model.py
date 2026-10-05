"""
GPT-2 模型（精简版），结构与 OpenAI GPT-2 完全一致。

================================================================================
一、整体数据流（读懂这张图，就读懂了整个文件）
================================================================================

  输入 idx: (B, T)         每个数字是一个 token 的编号，比如 "Hello" → 15496
     │
     │  wte 查表（token 向量） + wpe 查表（位置向量）
     ▼
  x: (B, T, C)             每个 token 变成一个 C 维向量。
     │                     这条一路往下传的 x 叫“残差流”（residual stream）
     │
     │  重复 n_layer 次 Block：
     │     x = x + Attention(LayerNorm(x))    token 之间交换信息
     │     x = x + MLP(LayerNorm(x))          每个 token 各自加工信息
     ▼
  ln_f（最后一次 LayerNorm） → lm_head（线性层）
     ▼
  logits: (B, T, V)        每个位置上，对词表里每个词打一个分：分越高，越可能是下一个词
     │
     │  cross_entropy：和正确答案 targets 比较
     ▼
  loss: 一个数字           越小说明预测越准

  符号约定（整个文件通用）：
    B  = batch size，一次处理几条序列
    T  = 序列长度，每条序列有几个 token
    C  = n_embd，每个 token 向量的维度（GPT-2 是 768）
    nh = n_head，注意力头数（GPT-2 是 12）
    hd = head_dim = C / nh，每个头的维度（GPT-2 是 64）
    V  = vocab_size，词表大小（GPT-2 是 50257）

================================================================================
二、参数都在哪里（GPT-2 124M）
================================================================================

  wte（token embedding）  50257 × 768        ≈ 38.6M   占 31%，和 lm_head 共享
  wpe（位置 embedding）    1024 × 768         ≈  0.8M
  12 个 Block             每个约 7.09M        ≈ 85.1M   占 68%
      ├─ attention        4 × 768²           ≈  2.4M   （c_attn 3C² + c_proj C²）
      └─ MLP              8 × 768²           ≈  4.7M   （c_fc 4C² + c_proj 4C²）
  ln_f                                        ≈  1.5K
  合计                                        ≈ 124.4M

================================================================================
三、两个性能开关（只改变“怎么算”，不改变“算什么”，结果在数值误差内完全一样）
================================================================================

  attn_impl = "naive"  手写 attention，会显式生成 (B, nh, T, T) 的分数矩阵，显存随 T² 增长
              "sdpa"   调用 F.scaled_dot_product_attention，GPU 上会走 FlashAttention
  act_ckpt  = True     activation checkpointing：前向不保存 Block 内部的中间结果，
                       反向时重新算一遍。用大约多 1/3 的计算量，换取大量显存
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ==============================================================================
# 模型配置
# ==============================================================================
# 用 dataclass 把所有超参数集中在一起，创建模型时只需要传一个 config 对象。
@dataclass
class GPTConfig:
    # 最大序列长度（上下文窗口）。决定了位置 embedding 表有多少行，
    # 模型一次最多能“看到”多少个 token。
    block_size: int = 1024

    # 词表大小。GPT-2 的词表 = 256 个单字节 + 50000 个 BPE 合并词 + 1 个结束符 <|endoftext|>。
    # 改成 50304 是一个经典的性能实验：50304 是 64 和 128 的倍数，
    # GPU 做矩阵乘法时按 64 或 128 分块，尺寸刚好整除时不需要处理“边角料”，算得更快。
    # 多出来的 47 个词在数据里从不出现，模型会自己学会给它们打很低的分，不影响结果。
    vocab_size: int = 50257

    n_layer: int = 12  # Block 的层数。层数越多，模型越“深”
    n_head: int = 12   # 注意力头数。每个头独立关注一种关系
    n_embd: int = 768  # 每个 token 向量的维度 C。维度越大，模型越“宽”

    # 下面两个是性能实验开关，详见文件开头的说明
    attn_impl: str = "naive"
    act_ckpt: bool = False


# 预设的几种模型大小，train.py 里用 --model 选择。
MODEL_PRESETS = {
    # 约 6.9M 参数，其中 94% 是 embedding（50257 × 128）。
    # 只用于冒烟测试，CPU 也能跑。因为词表占比太大，它的性能特征和真实模型差别很大。
    "tiny": dict(n_layer=2, n_head=4, n_embd=128, block_size=256),
    # 约 30M 参数，适合显存较小的卡
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=1024),
    # 124M 参数，OpenAI GPT-2 最小的版本，也是本项目的主角
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),
    # 350M 参数
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
}


# ==============================================================================
# 因果自注意力（Causal Self-Attention）
# ==============================================================================
# 作用：让每个 token 从它【前面】的 token 那里收集信息。
#   - “自”注意力：q、k、v 都来自同一个序列
#   - “因果”：每个 token 只能看自己和前面的 token，不能看后面的（后面是要预测的答案）
# 计算过程可以对照 learn/04_attention.py，那里有一个 3 个 token 的手算例子。
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # C 必须能被头数整除，才能平均拆成 nh 个头（768 / 12 = 64）
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attn_impl = config.attn_impl

        # 把 x 投影成 q、k、v。本来需要 3 个 C→C 的线性层，这里合并成一个 C→3C 的线性层：
        # 一次大的矩阵乘法，比三次小的矩阵乘法更能把 GPU 跑满，也少启动两次 kernel。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 输出投影：多个头的结果拼接起来之后，再用一个线性层把各个头的信息混合在一起，
        # 然后加回残差流。
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 打一个标记，告诉 GPT._init_weights：这是残差分支的最后一层，初始化时要缩小。
        self.c_proj.SCALE_INIT = True

        # 只有手写版本需要自己准备因果 mask（sdpa 版本用 is_causal=True，内部自动处理）。
        if self.attn_impl == "naive":
            # 下三角矩阵：第 i 行的前 i+1 个位置是 True（可以看），其余是 False（不能看）。
            #   [[T, F, F],
            #    [T, T, F],
            #    [T, T, T]]
            mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            # register_buffer：mask 会跟着模型一起 .to(device) 搬到 GPU 上，
            #   但它不是参数，不会被训练，也不会交给优化器。
            # view(1, 1, ...)：前面加两个长度为 1 的维度，计算时自动广播到 (B, nh, ...)。
            # persistent=False：不保存进 state_dict。它随时可以重新生成，
            #   而且这样 naive 和 sdpa 两个版本的模型可以互相加载权重。
            self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size()

        # 一次算出 q、k、v：(B, T, C) → (B, T, 3C)，再沿最后一维切成三份，每份 (B, T, C)。
        #   q（query）：我在找什么信息
        #   k（key）  ：我能提供什么信息（相当于标签）
        #   v（value）：我实际提供的内容
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 拆成多头：(B, T, C) → view → (B, T, nh, hd) → transpose → (B, nh, T, hd)
        # 为什么要 transpose？矩阵乘法 @ 只作用在最后两维上。把 (T, hd) 放到最后，
        # 前面的 (B, nh) 就成了“批次”维度：每个样本的每个头都独立做一次 attention，
        # GPU 可以把这 B × nh 个小矩阵乘法一起并行计算。
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.attn_impl == "sdpa":
            # PyTorch 内置实现，数学上和下面的 naive 版本完全相同。
            # PyTorch 会自动挑选最快的后端：在 GPU + bf16/fp16 下走 FlashAttention，
            # 它把计算分成小块在片上高速缓存里完成，【从不在显存里存完整的 T×T 矩阵】，
            # 所以又快又省显存。fp32 下会退回到其他实现。
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # ---- 手写 attention：四步 ----

            # 第 1 步：算分数。(B, nh, T, hd) @ (B, nh, hd, T) → (B, nh, T, T)
            #   att[b, h, i, j] = token i 的 query 和 token j 的 key 的点积 = i 对 j 的关注程度。
            #   为什么除以 sqrt(hd)：两个 hd 维向量的点积，数值大小会随维度增长
            #   （标准差约为 sqrt(hd)，hd=64 时约等于 8）。数太大会让 softmax 变得极端，
            #   几乎只剩一个 1 其余全是 0，梯度就消失了。除以 sqrt(hd) 把数值拉回正常范围。
            #   ⚠ 显存：这个矩阵大小随 T² 增长，而且要一直保存到反向结束。
            #     B=4、T=1024、fp32 时，每层约 0.2GB，12 层就是 2.4GB 以上。
            #     这正是 FlashAttention 要解决的问题。
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # 第 2 步：因果 mask。把“未来”位置（mask 为 False 的地方）填成 -inf，
            #   softmax 之后 e^(-inf) = 0，这些位置的权重就变成 0。
            #   为什么需要：训练时所有位置是同时预测的。如果位置 i 能看到位置 i+1，
            #   它就直接“偷看”到了答案，学不到任何东西。
            #   [:, :, :T, :T]：mask 是按最大长度 block_size 准备的，这里截取当前长度 T。
            #   ~ 是按位取反：True 变成 False，False 变成 True。
            att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

            # 第 3 步：softmax，把每一行变成加起来等于 1 的权重。
            att = F.softmax(att, dim=-1)

            # 第 4 步：用权重对 v 加权求和。(B, nh, T, T) @ (B, nh, T, hd) → (B, nh, T, hd)
            #   每个 token 得到一个新向量 = 它关注的那些 token 的 value 的加权平均。
            y = att @ v

        # 把多个头拼回去：(B, nh, T, hd) → transpose → (B, T, nh, hd) → view → (B, T, C)
        # 为什么要 contiguous()：transpose 只是改变“看数据的方式”，内存里的实际排列没变，
        # 数据在内存中不再连续；而 view 要求内存连续，所以先用 contiguous() 复制成连续的一块。
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影，混合各个头的信息
        return self.c_proj(y)


# ==============================================================================
# MLP（前馈网络）
# ==============================================================================
# 作用：每个 token【各自独立】加工自己的向量，token 之间不交换信息。
# 常见的说法是：attention 负责“交流”，MLP 负责“思考”。
# MLP 占了每个 Block 约 2/3 的参数（8C²，attention 只有 4C²）。
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 先放大到 4 倍：768 → 3072。更宽的中间层能表达更复杂的变换。
        # “4 倍”是从原始 Transformer 论文沿用下来的惯例。
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # 非线性激活函数。必须有它：如果全是线性层，叠多少层在数学上都等于一个线性层。
        # approximate="tanh" 是 GPT-2 原版用的近似公式（当年精确版计算较慢）。
        # 这里保留是为了和 GPT-2 原版完全一致。
        self.gelu = nn.GELU(approximate="tanh")

        # 再缩回原来的宽度：3072 → 768，这样才能加回残差流
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # 同样是残差分支的最后一层，初始化时要缩小（见 GPT._init_weights）
        self.c_proj.SCALE_INIT = True

    def forward(self, x):
        # (B, T, C) → (B, T, 4C) → GELU → (B, T, C)
        return self.c_proj(self.gelu(self.c_fc(x)))


# ==============================================================================
# Transformer Block（一层）
# ==============================================================================
# GPT-2 就是把 12 个完全相同结构的 Block 叠起来（参数各不相同）。
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # LayerNorm：把每个 token 的向量归一化成均值 0、方差 1，再乘上可学习的缩放、
        # 加上可学习的偏移。作用是让数值保持在稳定范围内，训练不容易发散。
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 两个关键设计：
        #
        # 1. 残差连接 x = x + f(x)：
        #    每个子层不是替换 x，而是在 x 上“加一点修改”。
        #    反向传播时，加法会把梯度原封不动地往回传，形成一条“梯度高速公路”，
        #    让很深的网络也能训练起来。没有它，梯度穿过几十层后会变得极小（梯度消失）。
        #
        # 2. Pre-LN（先 LayerNorm 再进子层）：
        #    原始 Transformer 是先加再 LayerNorm（Post-LN）。GPT-2 改成先 LayerNorm，
        #    这样残差这条主路上没有任何归一化挡着，梯度传得更顺，训练更稳定。
        x = x + self.attn(self.ln_1(x))  # token 之间交换信息
        x = x + self.mlp(self.ln_2(x))   # 每个 token 各自加工
        return x


# ==============================================================================
# 完整的 GPT 模型
# ==============================================================================
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 模块的命名（transformer.wte、transformer.h 等）和 HuggingFace 版 GPT-2 保持一致，
        # 以后想加载 OpenAI 官方权重做对比时，参数名可以直接对上。
        self.transformer = nn.ModuleDict(dict(
            # token embedding：一张 V × C 的表。每个 token 编号对应一行，查表得到它的向量。
            # (B, T) 的编号 → (B, T, C) 的向量
            wte=nn.Embedding(config.vocab_size, config.n_embd),

            # position embedding：一张 block_size × C 的表，第 i 行代表“第 i 个位置”。
            # 为什么需要：attention 只看向量内容，本身不知道 token 的先后顺序。
            # 把位置向量加到 token 向量上，模型才能区分“猫追狗”和“狗追猫”。
            # GPT-2 用的是可学习的绝对位置向量；现代模型（如 LLaMA）多用 RoPE。
            wpe=nn.Embedding(config.block_size, config.n_embd),

            # n_layer 个 Block 依次排列
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # 最后一次 LayerNorm。因为用的是 Pre-LN，最后一个 Block 输出的残差流
            # 没有被归一化过，送进输出层之前要补一次。
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # 输出层：把每个位置的 C 维向量，变成对词表里 V 个词的打分（logits）。
        # bias=False 和 GPT-2 原版一致。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享（weight tying）：输入的 wte 和输出的 lm_head 用同一个矩阵。
        #   形状刚好一样：Embedding(V, C) 的权重是 (V, C)，Linear(C, V) 的权重也是 (V, C)。
        #   好处 1：省参数。这个矩阵有 3860 万个参数，占整个模型的 31%。
        #   好处 2：语义上合理。“输入是哪个词”和“输出该是哪个词”应该用同一套词向量来表示。
        self.transformer.wte.weight = self.lm_head.weight

        # 递归地对每个子模块调用 _init_weights，完成参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """按 GPT-2 论文的方式初始化参数。初始化不好，训练一开始就可能发散或者学不动。"""
        if isinstance(module, nn.Linear):
            # 权重用均值 0、标准差 0.02 的正态分布随机初始化（GPT-2 的取值）
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                # 残差分支的最后一层，标准差再乘以 1/sqrt(2 × n_layer)。
                # 原因：残差流是一路累加的。每个 Block 往上加 2 次（attention 一次、MLP 一次），
                # 总共加 2 × n_layer 次。每加一次方差就变大一些，
                # 如果不缩小，残差流的数值会随层数越来越大。
                # 缩小 1/sqrt(累加次数) 之后，初始时总方差大致保持不变。
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # 偏置全部初始化为 0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # LayerNorm 不需要处理：PyTorch 默认的缩放 = 1、偏移 = 0 就是正确的初始值

    def forward(self, idx, targets=None):
        """
        idx:     (B, T)  输入的 token 编号
        targets: (B, T)  每个位置的正确答案，也就是“下一个 token”的编号。
                         推理时没有答案，传 None 即可。
        返回:    logits (B, T, V) 和 loss（targets 为 None 时 loss 也是 None）
        """
        B, T = idx.size()
        # 位置表只有 block_size 行，序列再长就查不到了
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"

        # 位置编号 [0, 1, 2, ..., T-1]，要和 idx 放在同一个设备上
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # token 向量 (B, T, C) + 位置向量 (T, C)。位置向量会自动广播到 B 个样本上。
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            if self.config.act_ckpt and self.training:
                # activation checkpointing：
                #   普通情况下，前向时 Block 内部的所有中间结果都要保存下来，留给反向用。
                #   checkpoint 包一层之后，前向只保存这个 Block 的输入 x，
                #   反向时用 x 把这个 Block 重新算一遍，得到中间结果再求梯度。
                #   代价：每个 Block 多算一次前向，总计算量大约增加 1/3。
                #   收益：激活值显存大幅下降，可以用更大的 batch。
                #   self.training：只在训练时开启。推理不做反向，本来就不保存中间结果。
                #   use_reentrant=False：PyTorch 官方推荐的新实现，旧实现有一些限制。
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        # (B, T, C) → (B, T, V)：每个位置对 V 个词的打分。
        # ⚠ 这个张量很大：B=4、T=1024、V=50257、fp32 时约 0.8GB，是显存大户之一。
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 交叉熵 = -log(模型给正确答案的概率)，然后对所有位置取平均。
            # 一条长度为 T 的序列同时提供了 T 道“预测下一个词”的题，所以训练效率很高。
            # F.cross_entropy 要求输入是二维 (N, V) 和一维 (N,)，
            # 所以先把 (B, T, V) 展平成 (B×T, V)，(B, T) 展平成 (B×T,)。
            # 小知识：刚初始化时模型对每个词一视同仁，概率都是 1/V，
            # 所以初始 loss ≈ ln(50257) ≈ 10.8。如果一开始就远高于这个值，初始化可能有问题。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, impl, device_type):
        """
        创建 AdamW 优化器。

        impl 决定参数更新这一步“怎么在 GPU 上执行”，数学上完全一样：
          "forloop"  一个参数张量一个参数张量地更新。GPT-2 有 148 个参数张量，
                     每个都要启动好几个小 kernel，加起来有几百次启动，开销很大
          "foreach"  把很多张量打包，用少数几个 kernel 一起更新（PyTorch 在 GPU 上的默认值）
          "fused"    整个 AdamW 更新合并成一个 CUDA kernel，启动次数和显存读写都最少
        """
        # 只优化需要梯度的参数。
        # 注意：wte 和 lm_head 共享同一个权重，parameters() 会自动去重，只出现一次。
        params = [p for p in self.parameters() if p.requires_grad]

        # weight decay（权重衰减）：每一步把权重往 0 拉一点，防止权重变得过大，是一种正则化。
        #   二维及以上的参数（线性层权重、embedding）：做 weight decay
        #   一维的参数（偏置、LayerNorm 的缩放和偏移）：不做。
        #     它们参数很少，不会导致过拟合；而把 LayerNorm 的缩放往 0 拉，反而会削弱信号。
        #   这是 GPT-2/GPT-3 以及大多数大模型训练的通用做法。
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]

        # 本项目只在 CUDA 上启用 fused 版本，其他设备自动退回 foreach
        if impl == "fused" and device_type != "cuda":
            print(f"[warn] fused AdamW 需要 CUDA，当前设备 {device_type}，改用 foreach")
            impl = "foreach"
        kwargs = {"fused": True} if impl == "fused" else {"foreach": impl == "foreach"}

        # betas=(0.9, 0.95)：GPT-3 论文的取值。
        #   0.9 控制梯度平均（动量）的记忆长度；
        #   0.95 控制梯度平方平均的记忆长度。比默认的 0.999 记得短一些，
        #   对梯度的变化反应更快，大模型训练更稳定。
        # eps=1e-8：防止除以 0 的小常数。
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **kwargs)
