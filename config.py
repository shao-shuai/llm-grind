from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 512
    dropout: float = 0.2
    bias: bool = False
    use_rotary: bool = False

