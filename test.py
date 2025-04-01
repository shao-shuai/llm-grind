import torch
from model import Rotary, CausalSelfAttention
from config import GPTConfig

# # define parametes
# dim = 64
# seq_len = 10
# batch_size = 2
# num_heads = 8

# # create dummy query and key tensors
# q = torch.randn(batch_size, seq_len, num_heads, dim // num_heads)
# k = torch.randn(batch_size, seq_len, num_heads, dim // num_heads)

# # Instantiate and apply Rotart embedding
# rotary = Rotary(dim // num_heads)
# q_rot, k_rot = rotary(q, k)

# # Print results
# # print("Original query shape:", q.shape)
# # print("Rotary transformed query shape:", q_rot.shape)
# # print("Original key shape:", k.shape)
# # print("ROtary transformed key shape:", k_rot.shape)

# print("This is inverse freq:", rotary.inv_freq.shape)
# print("This is cos cacaed:", rotary.cos_cached.shape)
# print("This is sin cached:", rotary.sin_cached.shape)
# # inv = 10_000 ** torch.arange(0, dim, 2, device=q.device) / dim
# # print("This is inv shape:", inv.shape)

x = torch.rand(4, 512, 512)
attn = CausalSelfAttention(GPTConfig)
y = attn(x)
print("This is the reuslt after self attention:", y) 
print("The shape of y is:", y.shape)