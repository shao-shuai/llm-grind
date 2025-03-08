# LLM Grind

### Day 1 positional encoding

Positional encoding will encode token embedding with positional information.

The implementation hereis rotary positional embedding (RoPE), the idea is each token embedding will be applied a rotation matrix and the relative position between 2 tokens can be calculated.

The advantage of using this than the postional encoding of vanilla transformer is it includes positional information but also keep token embedding unpolluted.

The input and output of RoPE is unchanged [batch, seq_len, n_heads, head_dim].

Why only do RoPE on query and key? - Because only query and key do self attention.

### Day 2 self attention

Self attention is the core of transformer architecture.

Let's say the input prompt is "Tell me a one sentence joke".

First the sentence will be split into tokens and do token embedding, each token will be represented with a vector.

```python
Tell            [0.9679, 0.5974, 0.3854],
me              [0.9098, 0.7725, 0.6224],
a               [0.5503, 0.2176, 0.4234],
one             [0.2098, 0.8982, 0.9083],
sentence        [0.7632, 0.4860, 0.0338],
joke            [0.2821, 0.2571, 0.7760]
```

Next we need to calculate the correlation of any two tokens, for example, if we want to know how close is it between `Tell` and `me`, we can compute the dot product of the 2 vectors

```python
[0.9679, 0.5974, 0.3854] @ [0.9098, 0.7725, 0.6224] = 1.5802
```

To calculate the coorelation of any two tokens, we can do it in matrix:

Each value of the matrix is called attention score and indicates the correlation between 2 tokens/vectors, for example, row 1 column 6 `0.7257` denotes the correlation between `Tell` and `joke`. The values of the first row indicate the correlation between `Tell` and each of the six tokens including itself.

```python
>>> a
tensor([[0.9679, 0.5974, 0.3854],
        [0.9098, 0.7725, 0.6224],
        [0.5503, 0.2176, 0.4234],
        [0.2098, 0.8982, 0.9083],
        [0.7632, 0.4860, 0.0338],
        [0.2821, 0.2571, 0.7760]])
>>> attn = a @ a.T
>>> attn
tensor([[1.4423, 1.5820, 0.8258, 1.0897, 1.0421, 0.7257],
        [1.5820, 1.8119, 0.9323, 1.4501, 1.0908, 0.9382],
        [0.8258, 0.9323, 0.5294, 0.6955, 0.5401, 0.5397],
        [1.0897, 1.4501, 0.6955, 1.6758, 0.6273, 0.9950],
        [1.0421, 1.0908, 0.5401, 0.6273, 0.8198, 0.3665],
        [0.7257, 0.9382, 0.5397, 0.9950, 0.3665, 0.7479]])
```

Next we do softmax to normalize the score for each of the tokens. After softmax, the sum of each row would be 1.

```python
F.softmax(attn, dim=1)
tensor([[0.2197, 0.2527, 0.1186, 0.1544, 0.1473, 0.1073],
        [0.2086, 0.2625, 0.1089, 0.1828, 0.1276, 0.1096],
        [0.1910, 0.2124, 0.1420, 0.1676, 0.1435, 0.1435],
        [0.1552, 0.2225, 0.1046, 0.2788, 0.0977, 0.1411],
        [0.2162, 0.2270, 0.1309, 0.1428, 0.1731, 0.1100],
        [0.1640, 0.2029, 0.1362, 0.2147, 0.1145, 0.1677]])
```
Since we now know the attention score between any two tokens, we can calcualte a weighted sum vector based on the attention score for any given token.

For example for the first token, the weighted sum vector would be:

```python
0.2197*[0.9679, 0.5974, 0.3854] + 0.2527*[0.9098, 0.7725, 0.6224] + ... + 0.1073*[0.2821, 0.2571, 0.7760]
```

Scale it with matrix operation

```python
attn_w @ a
tensor([[0.6829, 0.5901, 0.5207],
        [0.6673, 0.6055, 0.5453],
        [0.6414, 0.5663, 0.5344],
        [0.5831, 0.6216, 0.6087],
        [0.6809, 0.5737, 0.5010],
        [0.5980, 0.5760, 0.5762]])
```

The matrix has the same shape as the input matrix we see in the begining, but now it carries information of not the token itself, but also information of how other tokens affect it. The purpose of self attention is to get a weighted sum matrix like this.