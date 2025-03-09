import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from config import GPTConfig

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10_000):
        super().__init__()
        self.dim = dim # dim must be head dimension
        self.base = base # A scaling constatnt used to determine the frequentices for the sinusoidal functions
        self.inv_freq = None # will hold the inverse frequency values computed from the base and dim
        self.seq_len_cached = None # Keeps track of the last sequence length processed so that the sine and cosine matrices can be resued if the sequence length hasn't changed
        self.cos_cached = None # Store precomputed cosine matrices
        self.sin_cached = None # Store precomputed sine matrices

    def forward(self, q, k):
        seq_len = q.shape[1] # q, k shape (batch, seq_len, n_heads, head_dim)
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=q.device)))
            print("inf_freq shape is:", self.inv_freq.shape)
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().type_as(q)
            print("cos_cached shape is:", self.cos_cached.shape)
            self.sin_cached = freqs.sin().type_as(q)
            print("sin_cached shape is:", self.sin_cached.shape)

        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        q_ = self.apply_rotary_emb(q, cos, sin)
        k_ = self.apply_rotary_emb(k, cos, sin)

        return q_, k_
    
    def apply_rotary_emb(self, x, cos, sin):
        assert x.ndim == 4 # multihead attention
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_head == 0 # Ensure that the embedding size is evenly divisible by the number of attention heads
        self.head_dim = config.n_embed // config.n_head
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print("Not using flash attention")
            self.register_butter(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        if config.use_rotary:
            self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape # (batch_size, seq_len, n_embed)
        q, k, v = self.c_attn(x).split(self.config.n_embed, dim=2)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        # Apply rotary embeddings if enabled
        if self.config.use_rotary:
            q, k = self.rotary(q, k)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True
            )
        else:
            attn_pattern = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
            attn_pattern = attn_pattern.masked_fill(
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )
            attn = F.softmax(attn_pattern, dim=-1)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y
    
class FeedForward(nn.Module):
    def __inti__(self, config):
        super().__init__()
        hideen_dim = 4 * config.n_embed
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.n_embed, hideen_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, hideen_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))