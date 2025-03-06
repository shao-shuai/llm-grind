# LLM Grind

### Day 1 positional encoding

Positional encoding will encode token embedding with positional information.

The implementation hereis rotary positional embedding (RoPE), the idea is each token embedding will be applied a rotation matrix and the relative position between 2 tokens can be calculated.

The advantage of using this than the postional encoding of vanilla transformer is it includes positional information but also keep token embedding unpolluted.

The input and output of RoPE is unchanged [batch, seq_len, n_heads, head_dim].

Why only do RoPE on query and key? - Because only query and key do self attention.