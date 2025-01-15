"""Core modules for the ESM2 protein language model.

This module contains the building blocks used in the ESM2 model implementation,
including layer normalization, transformer layers, and prediction heads.
"""

import math
from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx

def gelu(x):
    """Gaussian Error Linear Unit activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with GELU activation applied
    """
    return x * 0.5 * (1.0 + jnp.erf(x / math.sqrt(2.0)))

def symmetrize(x):
    """Make tensor symmetric in final two dimensions.
    
    Used for contact prediction to ensure symmetry in the contact map.
    
    Args:
        x: Input tensor of shape (..., N, N)
        
    Returns:
        Symmetric tensor where x[...,i,j] = x[...,j,i]
    """
    return x + x.transpose(-1, -2)

def apc(x):
    """Perform average product correction for contact prediction.
    
    This function implements the average product correction (APC) procedure used
    to remove background noise from contact prediction matrices.
    
    Args:
        x: Input tensor of shape (batch_size, channels, seq_len, seq_len)
        
    Returns:
        Tensor with same shape as input after APC correction
    """
    a1 = x.sum(axis=-1, keepdims=True)
    a2 = x.sum(axis=-2, keepdims=True)
    a12 = x.sum(axis=(-1, -2), keepdims=True)

    avg = a1 * a2
    avg = avg / a12
    normalized = x - avg
    return normalized

class ESM1LayerNorm(eqx.Module):
    """Layer normalization implementation from ESM1.
    
    Normalizes input tensors along the last dimension with learnable scale and shift.
    
    Attributes:
        weight: Scale parameter
        bias: Shift parameter
        eps: Small constant for numerical stability
    """
    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float

    def __init__(self, hidden_size, eps=1e-12, key=None):
        super().__init__()
        self.eps = eps
        self.weight = jnp.ones(hidden_size)
        self.bias = jnp.zeros(hidden_size)

    def __call__(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.weight.shape)))
        means = jnp.mean(x, axis=dims, keepdims=True)
        x_zeromean = x - means
        variances = jnp.mean(jnp.square(x_zeromean), axis=dims, keepdims=True)
        x = x_zeromean / jnp.sqrt(variances + self.eps)
        return (self.weight * x) + self.bias

class TransformerLayer(eqx.Module):
    """Transformer layer block for ESM2.
    
    Implements a standard transformer layer with self-attention followed by
    a feed-forward network, with residual connections and layer normalization.
    
    Attributes:
        embed_dim (int): Embedding dimension
        ffn_embed_dim (int): Feed-forward network hidden dimension
        attention_heads (int): Number of attention heads
        self_attn (MultiheadAttention): Self-attention module
        self_attn_layer_norm (ESM1LayerNorm): Layer norm before self-attention
        fc1 (eqx.nn.Linear): First feed-forward layer
        fc2 (eqx.nn.Linear): Second feed-forward layer
        final_layer_norm (ESM1LayerNorm): Final layer normalization
        use_rotary_embeddings (bool): Whether to use rotary positional embeddings
    """
    embed_dim: int
    ffn_embed_dim: int
    attention_heads: int
    self_attn: eqx.Module  # Will be MultiheadAttention
    self_attn_layer_norm: ESM1LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    final_layer_norm: ESM1LayerNorm
    use_rotary_embeddings: bool

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        key,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        
        keys = jax.random.split(key, 5)
        
        from .multihead_attention import MultiheadAttention  # Import here to avoid circular imports
        self.self_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
            key=keys[0]
        )
        self.self_attn_layer_norm = ESM1LayerNorm(embed_dim, key=keys[1])
        
        self.fc1 = eqx.nn.Linear(embed_dim, ffn_embed_dim, key=keys[2])
        self.fc2 = eqx.nn.Linear(ffn_embed_dim, embed_dim, key=keys[3])
        
        self.final_layer_norm = ESM1LayerNorm(embed_dim, key=keys[4])

    def __call__(
        self, 
        x, 
        self_attn_mask=None, 
        self_attn_padding_mask=None, 
        need_head_weights=False,
        key=None
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
            key=key
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn

class RobertaLMHead(eqx.Module):
    """Head for masked language modeling prediction.
    
    This module transforms the transformer outputs into vocabulary logits
    for masked token prediction.
    
    Attributes:
        dense (eqx.nn.Linear): Dense projection layer
        layer_norm (ESM1LayerNorm): Layer normalization
        weight (jnp.ndarray): Output projection weight (shared with embedding)
        bias (jnp.ndarray): Output bias
    """
    dense: eqx.nn.Linear
    layer_norm: ESM1LayerNorm
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, embed_dim, output_dim, weight, key):
        super().__init__()
        keys = jax.random.split(key, 2)
        self.dense = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.layer_norm = ESM1LayerNorm(embed_dim, key=keys[1])
        self.weight = weight
        self.bias = jnp.zeros(output_dim)

    def __call__(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = jnp.dot(x, self.weight.T) + self.bias
        return x 

class ContactPredictionHead(eqx.Module):
    """Head for protein contact prediction.
    
    Performs symmetrization and average product correction on attention maps,
    followed by a learned projection to predict protein contacts.
    
    Attributes:
        in_features (int): Number of input features (layers * heads)
        prepend_bos (bool): Whether sequences have BOS token prepended
        append_eos (bool): Whether sequences have EOS token appended
        eos_idx (Optional[int]): Index of the EOS token if used
        regression (eqx.nn.Linear): Linear projection for contact prediction
    """
    
    in_features: int
    prepend_bos: bool
    append_eos: bool
    eos_idx: Optional[int]
    regression: eqx.nn.Linear
    
    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        eos_idx: Optional[int] = None,
        key = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        
        # Split key for potential future use (e.g. dropout)
        if key is not None:
            keys = jax.random.split(key, 2)
        else:
            keys = [None, None]
        
        # Initialize regression layer
        self.regression = eqx.nn.Linear(in_features, 1, key=keys[0])

    def __call__(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = (tokens != self.eos_idx).astype(attentions.dtype)
            eos_mask = eos_mask[:, None] * eos_mask[:, None, None]
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
            
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
            
        batch_size, layers, heads, seqlen, _ = attentions.shape
        attentions = attentions.reshape(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = apc(symmetrize(attentions))
        # Permute from B x C x T x T to B x T x T x C
        attentions = jnp.transpose(attentions, (0, 2, 3, 1))
        
        # Apply regression and sigmoid
        contacts = jax.nn.sigmoid(self.regression(attentions).squeeze(3))
        return contacts 