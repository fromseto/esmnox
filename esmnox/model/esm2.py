"""ESM2 protein language model implementation."""

from typing import Union
import jax
import jax.numpy as jnp
import equinox as eqx

from esmnox.data import Alphabet
from esmnox.modules import (
    TransformerLayer,
    ESM1LayerNorm,
    RobertaLMHead,
    ContactPredictionHead,
)

class ESM2(eqx.Module):
    """ESM2 transformer model for protein sequences.
    
    This class implements the core ESM2 architecture, including the embedding layer,
    transformer layers, and prediction heads for masked language modeling and contact prediction.
    
    Attributes:
        num_layers (int): Number of transformer layers
        embed_dim (int): Dimension of token embeddings
        attention_heads (int): Number of attention heads per transformer layer
        alphabet (Alphabet): Token alphabet defining the vocabulary and special tokens
        padding_idx (int): Index used for padding tokens
        mask_idx (int): Index used for masked tokens
        cls_idx (int): Index for the CLS token
        eos_idx (int): Index for the EOS token
        prepend_bos (bool): Whether to prepend BOS token to sequences
        append_eos (bool): Whether to append EOS token to sequences
        token_dropout (bool): Whether to apply token dropout during training
    """
    num_layers: int
    embed_dim: int
    attention_heads: int
    alphabet: Alphabet
    padding_idx: int
    mask_idx: int
    cls_idx: int
    eos_idx: int
    prepend_bos: bool
    append_eos: bool
    token_dropout: bool
    
    # Layers
    embed_scale: float
    embed_tokens: eqx.nn.Embedding
    layers: list[TransformerLayer]
    emb_layer_norm_after: ESM1LayerNorm
    lm_head: RobertaLMHead
    # contact_head: ContactPredictionHead

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
        key = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        
        # if not isinstance(alphabet, Alphabet):
        #     alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        # Split PRNG key
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num_layers + 5)

        # Initialize layers
        self.embed_scale = 1
        self.embed_tokens = eqx.nn.Embedding(
            len(alphabet),
            embed_dim,
            # embedding_init=jax.nn.initializers.normal(1.0),
            key=keys[0]
        )

        self.layers = [
            TransformerLayer(
                embed_dim,
                4 * embed_dim,
                attention_heads,
                key=keys[i+1],
                use_rotary_embeddings=True,
            )
            for i in range(num_layers)
        ]

        self.emb_layer_norm_after = ESM1LayerNorm(embed_dim, key=keys[-2])

        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            output_dim=len(alphabet),
            weight=self.embed_tokens.weight,
            key=keys[-1]
        )

        # self.contact_head = ContactPredictionHead(
        #     in_features=num_layers * attention_heads,
        #     prepend_bos=self.prepend_bos,
        #     append_eos=self.append_eos,
        #     eos_idx=self.eos_idx,
        #     key=keys[-1]
        # )

    def __call__(
        self, 
        tokens, 
        repr_layers=[], 
        need_head_weights=False, 
        return_contacts=False,
        rng=None
    ):
        # Enable attention weights if contact prediction is requested
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        # Create padding mask for input tokens
        padding_mask = tokens == self.padding_idx  # B, T

        # Embed tokens and apply scaling
        x = self.embed_scale * self.embed_tokens(tokens)

        # Apply token dropout during training
        if self.token_dropout:
            mask = (tokens == self.mask_idx)[..., None]
            x = jnp.where(mask, 0.0, x)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = jnp.sum(~padding_mask, axis=-1)
            mask_ratio_observed = (
                jnp.sum(tokens == self.mask_idx, axis=-1) / src_lengths
            )
            x = x * (1 - mask_ratio_train) / (
                1 - mask_ratio_observed[:, None, None]
            )

        # Zero out embeddings for padding tokens
        if padding_mask is not None:
            x = x * (1 - padding_mask[..., None])

        # Initialize storage for intermediate representations if requested
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        # Initialize storage for attention weights if needed
        if need_head_weights:
            attn_weights = []

        # Transpose input for transformer layers: batch x seq x dim -> seq x batch x dim
        # (B, T, E) => (T, B, E)
        x = x.transpose(1, 0, 2)

        # Optimize: remove padding mask if no padding tokens present
        if not jnp.any(padding_mask):
            padding_mask = None

        # Generate per-layer RNG keys if random number generation is needed
        if rng is not None:
            rngs = jax.random.split(rng, len(self.layers))
        else:
            rngs = [None] * len(self.layers)

        # Pass input through transformer layers
        for layer_idx, (layer, layer_rng) in enumerate(zip(self.layers, rngs)):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
                rng=layer_rng
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(1, 0, 2)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0, 2, 3))

        # Apply final layer norm and transpose back
        x = self.emb_layer_norm_after(x)
        x = x.transpose(1, 0, 2)  # (T, B, E) => (B, T, E)

        # Store final layer representation if requested
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
            
        # Get logits from language modeling head
        x = self.lm_head(x)

        # Prepare output dictionary with logits and representations
        result = {"logits": x, "representations": hidden_representations}

        # Process attention weights and generate contacts if requested
        if need_head_weights:
            # Stack attention weights: B x L x H x T x T
            attentions = jnp.stack(attn_weights, axis=1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.astype(attentions.dtype)
                attention_mask = attention_mask[:, None] * attention_mask[:, None, None]
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions, rng=rng)
                result["contacts"] = contacts

        return result 