"""Decoder-only transformer: RoPE, RMSNorm, SwiGLU, optional GQA. ~180 lines."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6
    d_model: int = 384
    d_ffn: int = 1024
    dropout: float = 0.0
    rope_base: float = 10000.0
    # --- modded-nanogpt-style speedrun knobs (all default-off for parity) ---
    qk_norm: bool = False        # RMSNorm on Q and K before attention; stabilises
                                 # training and lets you push LR higher.
    zero_init_proj: bool = False  # zero-init the residual-write matrices (attn
                                 # o_proj + ffn down-proj) so each block starts
                                 # as identity → stable warmup at higher LR.
    tie_embeddings: bool = True  # share tok_emb with lm_head. The speedrun found
                                 # *untying* helps loss once you have the tokens
                                 # to support the extra params; set False to A/B.
    # --- Multi-Token Prediction (DeepSeek-V3 style, simplified) ---
    mtp_tokens: int = 0          # number of *extra* future tokens to predict
                                 # (0 = off). Each adds one auxiliary head that
                                 # predicts token n+2, n+3, ... from the same
                                 # final hidden state. Denser gradient → better
                                 # sample efficiency. Train-only: generate()
                                 # uses the main head only, so zero infer cost.
    mtp_weight: float = 0.3      # λ on the averaged auxiliary loss (DeepSeek-V3
                                 # used 0.3).

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, D]; compute in fp32 for stability
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.type_as(x)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)        # [T, head_dim/2]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; rotate halves
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        # QK-Norm: per-head RMSNorm over head_dim, applied to Q and K *after*
        # projection but *before* RoPE. Keeps attention-logit scale in check.
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else None

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, Hk, D = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        k = self.k_proj(x).view(B, T, Hk, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if Hk != H:
            # repeat KV groups to match heads
            rep = H // Hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # PyTorch's SDPA picks Flash/mem-eff if available
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)   # gate
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)   # up
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # tie weights (saves params, common in small models). Untying can help
        # loss at scale once you have tokens to support the extra params — the
        # modded-nanogpt speedrun unties; see GPTConfig.tie_embeddings.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # Multi-Token Prediction heads: one extra Linear per future offset.
        # Head j (0-indexed) predicts token at position i+2+j from h_i. These
        # are auxiliary — discarded at inference — so we keep them simple linear
        # projections off the same final hidden state rather than full DeepSeek
        # MTP modules (which add a transformer block per depth).
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(cfg.mtp_tokens)]
        )
        self.apply(self._init)
        # Zero-init the residual-write projections (attn o_proj + ffn down-proj)
        # so every block starts as the identity map. muP-like; stabilises the
        # early/high-LR phase. Done *after* apply(_init) so it isn't clobbered.
        if cfg.zero_init_proj:
            for blk in self.blocks:
                nn.init.zeros_(blk.attn.o_proj.weight)
                nn.init.zeros_(blk.ffn.w2.weight)
        # cache RoPE on first forward; key includes (device, dtype, seq_len)
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count. With tied embeddings the lm_head.weight is the
        same Tensor as tok_emb.weight, so `parameters()` already counts it once.
        `non_embedding=True` subtracts the (single) embedding table, matching
        the convention used by GPT-2 / Chinchilla papers."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        # Rebuild cache if device, dtype, or required seq_len has changed.
        cache = self._rope_cache
        if (cache is None
                or cache[0].device != x.device
                or cache[0].dtype != x.dtype
                or cache[0].shape[0] < T):
            self._rope_cache = build_rope_cache(
                self.cfg.block_size, self.cfg.head_dim, self.cfg.rope_base, x.device, x.dtype
            )
        cos, sin = self._rope_cache
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # Multi-Token Prediction auxiliary loss. `targets[:, t]` is the token
        # one step after x[:, t]; MTP head j predicts the token (j+2) ahead,
        # i.e. `targets` shifted left by (j+1). We drop the trailing positions
        # that fall off the end of the sequence. Averaged over heads, scaled by
        # mtp_weight, and added to the main next-token loss.
        #
        # Train-only: gated on self.training so evaluate() (model.eval()) reports
        # the pure next-token CE — keeps val loss comparable to a non-MTP run.
        if self.mtp_heads and self.training:
            T = targets.size(1)
            aux = 0.0
            for j, head in enumerate(self.mtp_heads):
                shift = j + 1
                if T - shift <= 0:
                    continue
                pred = head(x[:, : T - shift, :])          # predict from h_t
                tgt = targets[:, shift:]                    # token (j+2) ahead
                aux = aux + F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)), tgt.reshape(-1)
                )
            loss = loss + self.cfg.mtp_weight * (aux / len(self.mtp_heads))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def hidden(self, idx):
        """Run the trunk and return the post-final-norm hidden states [B, T, D].

        The shared primitive behind both the main `lm_head` and the auxiliary
        MTP heads — exposing it lets the speculative-decoding path read the MTP
        drafts off the same hidden state without re-running the trunk.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        cache = self._rope_cache
        if (cache is None
                or cache[0].device != x.device
                or cache[0].dtype != x.dtype
                or cache[0].shape[0] < T):
            self._rope_cache = build_rope_cache(
                self.cfg.block_size, self.cfg.head_dim, self.cfg.rope_base, x.device, x.dtype
            )
        cos, sin = self._rope_cache
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.final_norm(x)

    @torch.no_grad()
    def generate_greedy(self, idx, max_new_tokens: int):
        """Plain greedy autoregressive decoding using the main head only.

        The baseline the MTP-speculative path must match token-for-token.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits = self.lm_head(self.hidden(idx_cond))[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def generate_mtp_speculative(self, idx, max_new_tokens: int):
        """Medusa-style self-speculative greedy decoding using the MTP heads.

        Each step:
          1. Run the trunk once over the context. From the last hidden state the
             *main* head gives the true next token `a`, and MTP head j gives a
             cheap draft for the token (j+2) ahead — a chain of K candidate
             tokens [a, d0, d1, ..., d_{K-1}] produced by a single trunk pass.
          2. Verify the chain with one more trunk pass over the appended
             candidates: the main head's greedy argmax at each candidate position
             is the "true" continuation. Accept the longest matching prefix; on
             the first mismatch keep the corrected true token and stop.

        Greedy verification makes the output **identical** to `generate_greedy`,
        but each verification pass can emit up to K+2 tokens (the true token, all
        K accepted drafts, plus one bonus token from the final verified position)
        for two trunk evaluations — that's the serving speedup the MTP heads buy
        for free (they were trained only as an auxiliary loss). Batch size 1 (the
        latency-bound serving regime). Returns (idx, stats) where stats records
        accepted-token counts per verification round.
        """
        assert idx.size(0) == 1, "speculative path benchmarks batch size 1"
        K = len(self.mtp_heads)
        if K == 0:
            return self.generate_greedy(idx, max_new_tokens), {"rounds": 0, "accepted": []}
        block = self.cfg.block_size
        accepted_per_round: list[int] = []
        produced = 0
        while produced < max_new_tokens:
            h_last = self.hidden(idx[:, -block:])[:, -1, :]        # [1, D]
            a = self.lm_head(h_last).argmax(-1)                    # true next token [1]
            drafts = [head(h_last).argmax(-1) for head in self.mtp_heads]  # K drafts
            # Candidate chain: true token then the K drafts.
            chain = torch.cat([a] + drafts).view(1, -1)           # [1, K+1]
            cand = torch.cat([idx, chain], dim=1)
            # One verification pass: main-head greedy argmax at each candidate
            # position gives the true continuation after consuming that token.
            verify_logits = self.lm_head(self.hidden(cand[:, -block:]))   # [1, L, V]
            # The candidate tokens occupy the last (K+1) positions; the true
            # next-token *after* candidate position p is argmax at that position.
            true_next = verify_logits[0, -(K + 1):, :].argmax(-1)  # [K+1]
            # a (chain[0]) is always correct by construction → accept it. Then
            # accept draft d_i iff it equals the verified true token at the
            # previous candidate position.
            accepted = [int(a)]
            for i in range(K):
                if int(drafts[i]) == int(true_next[i]):
                    accepted.append(int(drafts[i]))
                else:
                    # First mismatch: take the corrected true token and stop.
                    accepted.append(int(true_next[i]))
                    break
            else:
                # All drafts accepted → the token after the last draft is a free
                # bonus from the final verified position.
                accepted.append(int(true_next[K]))
            # Trim to the token budget and append.
            take = accepted[: max_new_tokens - produced]
            idx = torch.cat([idx, torch.tensor([take], device=idx.device, dtype=idx.dtype)], dim=1)
            produced += len(take)
            accepted_per_round.append(len(take))
        return idx, {"rounds": len(accepted_per_round), "accepted": accepted_per_round}
