import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

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

