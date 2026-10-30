"""Decoder-only transformer (GPT style) in PyTorch, the architecture of tiny-transformer-scratch."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the fused QKV and output projections."""
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Attend each position to itself and earlier positions."""
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        shape = (b, t, self.n_heads, c // self.n_heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        y = nnf.scaled_dot_product_attention(q, k, v, is_causal=True)
        out: Tensor = self.proj(y.transpose(1, 2).reshape(b, t, c))
        return out


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention and MLP with residuals."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the two sub-layers."""
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention then MLP, each with a residual."""
        x = x + self.attn(self.ln1(x))
        out: Tensor = x + self.mlp(self.ln2(x))
        return out


class GPT(nn.Module):
    """Token and position embeddings, a stack of blocks and a tied output head."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the layers and initialise weights."""
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(_init_weights)

    def forward(self, idx: Tensor) -> Tensor:
        """Logits of shape (batch, time, vocab)."""
        _, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        positions = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        logits: Tensor = self.ln_f(x) @ self.tok.weight.T
        return logits

    def loss(self, idx: Tensor, targets: Tensor) -> Tensor:
        """Mean next-token cross-entropy."""
        logits = self(idx)
        out: Tensor = nnf.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return out


def _init_weights(module: nn.Module) -> None:
    """GPT-2 style normal init for linear and embedding weights."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


def param_count(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_state_bytes(model: nn.Module) -> int:
    """Bytes of parameters, gradients and AdamW moments that DDP keeps on every rank."""
    return sum(4 * p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
