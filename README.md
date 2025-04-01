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

### Day 3 feed forward

The FeedForward class is a crucial component of the transformer atchitecture, providing non-linearity and additional transformation to the output of the self-attention mechanism.

w1 expands the input `n_embed` to `hidden_dim`

w2 projects back to original embedding size, `hidden_dim` to `n_embed`

w3 another transformation from `n_embed` to  `hidden_dim` to modulate activations

Dropout is applied after the second transformation to prevent overfitting

### Day 4 GRPO training

I am kind of out of track. Today I tried GRPO fine tuning with Qwen-2.5-1.5B with `openai/gsm8k` dataset. This is for getting a hands-on experience of DeepSeek-R1's receipe and thanks `unsloth` for making this possible. The result is amazing, with a few very simple reward functions, after 1000 steps of fine-tuning, the model is able to generate answers with <reasoning></reasoning> and <answer></answer> tags.

Sample before traning:

```xml
Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? 
Answer:
1040 
Response:
To determine how much more Janet spends on piano lessons than clarinet lessons in a year, we need to calculate the weekly and annual costs for both lessons.

For clarinet lessons:
- She pays $40/hour for 3 hours per week, so her weekly cost is \(40 \times 3 = \$120\).
- Assuming there are 52 weeks in a year, her annual cost for clarinet lessons is \(120 \times 52 = \$6,240\).

For piano lessons:
- She pays $28/hour for 5 hours per week, so her weekly cost is \(28 \times 5 = \$140\).
- Again, assuming there are 52 weeks in a year, her annual cost for piano lessons is \(140 \times 52 = \$7,280\).

Now, to find out how much more she spends on piano lessons than clarinet lessons in a year, we subtract the annual cost of clarinet lessons from the annual cost of piano lessons:
- \(7,280 - 6,240 = \$1,040\).

Therefore, Janet spends $1,040 more on piano lessons than on clarinet lessons in a year. 
Extracted:
To determine how much more Janet spends on piano lessons than clarinet lessons in a year, we need to calculate the weekly and annual costs for both lessons.

For clarinet lessons:
- She pays $40/hour for 3 hours per week, so her weekly cost is \(40 \times 3 = \$120\).
- Assuming there are 52 weeks in a year, her annual cost for clarinet lessons is \(120 \times 52 = \$6,240\).

For piano lessons:
- She pays $28/hour for 5 hours per week, so her weekly cost is \(28 \times 5 = \$140\).
- Again, assuming there are 52 weeks in a year, her annual cost for piano lessons is \(140 \times 52 = \$7,280\).

Now, to find out how much more she spends on piano lessons than clarinet lessons in a year, we subtract the annual cost of clarinet lessons from the annual cost of piano lessons:
- \(7,280 - 6,240 = \$1,040\).

Therefore, Janet spends $1,040 more on piano lessons than on clarinet lessons in a year.
```

Sample after training:

```xml
John runs a telethon to raise money.  For the first 12 hours, he generates $5000 per hour.  The remaining 14 hours, he generates 20% more per hour.  How much total money does he make? 
Answer:
144000 
Response:
<reasoning>
John generates money for the first 12 hours at $5000 per hour, which is a total of 12 * $5000 = $60,000. For the remaining 14 hours, he generates 20% more per hour, meaning he generates 120% of $5000, which is $5000 * 1.20 = $6000 per hour. Therefore, he generates a total of 14 * $6000 = $84,000 in the last 14 hours. Adding the money from the first 12 hours to the money from the last 14 hours, John makes a total of $60,000 + $84,000 = $144,000.
</reasoning>
<answer>
144000
</answer>
 
Extracted:
144000
```

Day 5 Recap LLM generation sampling

The way LLM sampling works is, it picks next token based on weighted random sampling, not always picking the token with the highest probability.

Why? Beaause greedy decoding (chossing the token with the highest probability) makes generation highly predictable and often repetitive.


Let's say the input tokens are `I like`, transformer will process the tokens and generate a list of logits of vovabulary size and each indicates the probability of the corresponding token. For example the logits are [0.1, 0.2, 0.3, 0.4] (make sure they sum to 1), then the weighted random sampleing is performed by use `torch.multinomial()`, if you only pick one token, it's guaranteed that the last token will always be picked, the result is somewhat random but not completely random. If you pick multiple times, the distribution should align with its probability.

This is how diversity is injested into LLM inference, and remember `LLM is not generating token with the highest probability`.