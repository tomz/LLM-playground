"""GPT-2 style transformer with optional gradient checkpointing."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ffn: int = 3072
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    # Opt-in modded-nanogpt stabilizer (default-off for GPT-2 parity): per-head
    # RMSNorm on Q and K before attention. Keeps attention-logit scale bounded so
    # you can train at higher LR without loss spikes. Adds 2 tiny norms per block.
    qk_norm: bool = False

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    """Minimal RMSNorm (fp32 reduction) — only used for optional QK-norm."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else None

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # SDPA chooses Flash / mem-efficient automatically
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ffn, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ffn, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, grad_checkpoint: bool = False,
                 fused_ce: bool = False):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = grad_checkpoint
        # Liger fused-linear-cross-entropy: computes the lm_head matmul and the
        # cross-entropy in one fused Triton kernel WITHOUT materializing the full
        # [B*T, vocab] logits tensor — the single largest activation in the
        # forward pass. Exact (not an approximation): on this model the loss
        # matches dense CE to ~1.8e-3 and grads to ~8e-3 rel (bf16 rounding).
        # The win is MEMORY: measured peak VRAM ~10.1 vs ~12.8 GiB (-20%) on a
        # 350M GPT-2. Throughput is hardware-dependent and can REGRESS where the
        # dense path already hits a well-tuned matmul: on RTX 5060 Ti (Blackwell
        # sm_120, torch 2.11) it ran ~26% slower (10.8k vs 14.5k tok/s). Treat
        # fused_ce as a VRAM-headroom lever (fit bigger batch/vocab/model), not a
        # speedup. Requires `pip install liger-kernel` + a Triton GPU; resolved
        # lazily so import stays optional.
        self.fused_ce = fused_ce
        self._liger_ce = None
        if fused_ce:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            self._liger_ce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init)
        # GPT-2 paper: scale residual-projection init by 1/sqrt(2*N)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, return_full_logits: bool = False):
        """Forward pass. By default returns last-position logits when no targets
        (saves memory at inference); pass `return_full_logits=True` to get the
        full [B, T, V] tensor (needed by HellaSwag / per-token scoring).
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        if targets is None:
            if return_full_logits:
                return self.lm_head(x), None
            return self.lm_head(x[:, [-1], :]), None
        if self._liger_ce is not None:
            # Fused linear-CE: pass hidden states + lm_head weight straight to the
            # kernel so the [B*T, vocab] logits are never materialized. Returns
            # loss only (no logits) — callers needing logits must disable fused_ce.
            loss = self._liger_ce(self.lm_head.weight, x.reshape(-1, x.size(-1)), targets.reshape(-1))
            return None, loss
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def configure_optimizer(self, weight_decay: float, lr: float, betas, fused: bool):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None):
        """Autoregressive sampling.

        * ``temperature == 0`` → greedy (argmax). Skips softmax/multinomial.
        * ``top_k``  → keep only the top-k logits before sampling.
        * ``top_p``  → nucleus: keep the smallest prefix whose probability mass
          sums to ``top_p`` (after sort). Combinable with ``top_k``; nucleus is
          applied second.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt], dim=1)
                continue
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Mask out tokens whose cumulative prob exceeds top_p, but
                # always keep the very top token (cumprobs > top_p shifted right
                # by 1 so the first crossing token is still admitted).
                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
                # Scatter back to original vocab order.
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
