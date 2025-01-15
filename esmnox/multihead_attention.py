"""Multi-head attention implementation for the ESM2 model.

This module implements the multi-head attention mechanism used in the transformer
layers, optionally including rotary positional embeddings.
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from esmnox.rotary_embedding import RotaryEmbedding

def softmax(x, axis=-1):
    return jax.nn.softmax(x.astype(jnp.float32), axis=axis)


class MultiheadAttention(eqx.Module):
    """Multi-head attention with optional rotary embeddings.
    
    This class implements the scaled dot-product attention mechanism with multiple
    attention heads. It can optionally use rotary positional embeddings for
    better handling of sequence position information.
    
    Attributes:
        embed_dim (int): Total embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        head_dim (int): Dimension of each attention head (embed_dim // num_heads)
        scaling (float): Scaling factor for attention scores
        rot_emb (Optional[RotaryEmbedding]): Rotary embedding layer if enabled
    """
    embed_dim: int
    num_heads: int
    dropout: float
    head_dim: int
    scaling: float
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    q_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    rot_emb: Optional[RotaryEmbedding]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        key=None,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5

        if key is not None:
            keys = jax.random.split(key, 4)
        else:
            keys = [None] * 4

        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])
        
        self.rot_emb = RotaryEmbedding(self.head_dim) if use_rotary_embeddings else None

    def __call__(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        need_head_weights=False,
        attn_mask=None,
        rng=None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        tgt_len, bsz, embed_dim = query.shape
        src_len = tgt_len

        scaling = self.scaling

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= scaling

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        if self.rot_emb is not None:
            q, k = self.rot_emb(q, k)

        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1))

        if attn_mask is not None:
            attn_weights = attn_weights + jnp.expand_dims(attn_mask, 0)

        if key_padding_mask is not None:
            attn_weights = jnp.where(
                key_padding_mask.transpose(1, 0),
                jnp.full_like(attn_weights, float('-inf')),
                attn_weights
            )

        attn_weights = softmax(attn_weights, axis=-1)
        
        if rng is not None:
            attn_weights = jax.random.bernoulli(
                rng, 
                p=1-self.dropout, 
                shape=attn_weights.shape
            ) * attn_weights / (1-self.dropout)

        attn = jnp.matmul(attn_weights, v)
        
        attn = attn.transpose(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.reshape(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(0, 1, 2, 3)
            if not need_head_weights:
                attn_weights = attn_weights.mean(axis=0)
        else:
            attn_weights = None

        return attn, attn_weights 