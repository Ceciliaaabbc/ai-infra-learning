"""
GPT-2 模型（精简版），结构与 OpenAI GPT-2 一致。

为了做性能实验，保留了两个开关：
  - attn_impl: "naive"  手写 attention，会显式构造 (B, nh, T, T) 的分数矩阵
               "sdpa"   F.scaled_dot_product_attention，GPU 上会走 FlashAttention
  - act_ckpt:  activation checkpointing，前向不保存 block 内部的激活值，反向时重算
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    block_size: int = 1024   # 最大序列长度
    vocab_size: int = 50257  # GPT-2 词表大小；改成 50304（64 的倍数）是一个经典实验
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attn_impl: str = "naive"
    act_ckpt: bool = False


MODEL_PRESETS = {
    "tiny": dict(n_layer=2, n_head=4, n_embd=128, block_size=256),  # 冒烟测试用，CPU 也能跑
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=1024),  # 显存较小的卡
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),  # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),  # 350M
}


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attn_impl = config.attn_impl
        # 一次矩阵乘法同时算出 q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = True
        if self.attn_impl == "naive":
            mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, nh, T, hd)：把 C 拆成 nh 个头，每个头独立做 attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # 分数矩阵 (B, nh, T, T)：显存随 T² 增长，这正是 FlashAttention 要解决的问题
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        # (B, nh, T, hd) -> (B, T, C)：把各个头的结果拼回去
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # token 之间交换信息
        x = x + self.mlp(self.ln_2(x))   # 每个 token 各自做非线性变换
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),  # position embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重共享：输出层和输入 embedding 用同一个矩阵（GPT-2 的做法）
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                # 残差分支的输出层按层数缩小初始化，防止残差累加后方差过大
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            if self.config.act_ckpt and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, impl, device_type):
        """impl: "forloop" 逐个参数更新 | "foreach" 批量更新（PyTorch 默认）| "fused" 单个 CUDA kernel"""
        params = [p for p in self.parameters() if p.requires_grad]
        # 矩阵（权重、embedding）做 weight decay；向量（bias、LayerNorm）不做
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        if impl == "fused" and device_type != "cuda":
            print(f"[warn] fused AdamW 需要 CUDA，当前设备 {device_type}，改用 foreach")
            impl = "foreach"
        kwargs = {"fused": True} if impl == "fused" else {"foreach": impl == "foreach"}
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **kwargs)
