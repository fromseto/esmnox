import jax.numpy as jnp
import equinox as eqx

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :x.shape[-2], :]
    sin = sin[:, :x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(eqx.Module):
    dim: int
    inv_freq: jnp.ndarray
    _seq_len_cached: int
    _cos_cached: jnp.ndarray
    _sin_cached: jnp.ndarray

    def __init__(self, dim: int, key=None):
        super().__init__()
        self.dim = dim
        # Generate inverse frequency buffer
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2).astype(float) / dim))
        self.inv_freq = inv_freq
        
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]
        
        if seq_len != self._seq_len_cached:
            t = jnp.arange(x.shape[seq_dimension])
            freqs = jnp.einsum('i,j->ij', t, self.inv_freq)
            emb = jnp.concatenate((freqs, freqs), axis=-1)
            
            self._cos_cached = jnp.expand_dims(jnp.cos(emb), 0)
            self._sin_cached = jnp.expand_dims(jnp.sin(emb), 0)
            self._seq_len_cached = seq_len
            
        return self._cos_cached, self._sin_cached

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        ) 