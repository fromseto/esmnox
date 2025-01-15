"""Utilities for loading model weights."""

import torch
import jax
import equinox as eqx
import jax.numpy as jnp
from pathlib import Path
from typing import NamedTuple
import re
import os
import json

# Use absolute imports
from esmnox.model.esm2 import ESM2
from esmnox.data import Alphabet

# We will be porting weights on CPU.
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch  # noqa: E402
import jax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from model.esm2 import ESM2  # noqa: E402
from data import Alphabet
import re


"""Utility functions for handling model weights and configuration.

This module provides functionality for loading and converting weights from PyTorch
to JAX/Equinox format, as well as model configuration management.
"""

def port_weights_from_torch(torch_weights, eqx_model):
    """Port weights from a PyTorch state dict to an Equinox model.
    
    This function recursively maps weights from a PyTorch model's state dictionary
    to the corresponding layers in an Equinox model. It handles both weights and biases,
    converting them to bfloat16 precision.

    Args:
        torch_weights (dict): PyTorch state dict containing model weights and biases
        eqx_model (eqx.Module): Equinox model instance with matching architecture

    Returns:
        eqx.Module: Updated Equinox model with ported weights

    Raises:
        ValueError: If encountering unsupported path types during weight mapping
    """

    def load_weights(path, leaf):
        path_pieces = []
        for path_elem in path:
            if isinstance(path_elem, jax.tree_util.GetAttrKey):
                path_pieces.append(path_elem.name)
            elif isinstance(path_elem, jax.tree_util.SequenceKey):
                path_pieces.append(str(path_elem.idx))
            else:
                raise ValueError(f"Unsupported path type {type(path_elem)}")

        path_pieces = ".".join(path_pieces)
        
        if "weight" in path_pieces or "bias" in path_pieces:
            weight = torch_weights[path_pieces]
            # `bfloat16` weights cannot be directly converted to numpy. Hence
            # we first upscale them to `float32`, and then load them in
            # `bfloat16`
            weight = jnp.asarray(weight.float().numpy())
            assert weight.shape == leaf.shape
            assert weight.dtype == leaf.dtype
            return weight
        else:
            print(f"Weights not ported for: {path_pieces}")
            return leaf

    return jax.tree_util.tree_map_with_path(load_weights, eqx_model)


class ModelArgs(NamedTuple):
    """Configuration arguments for the ESM2 model.

    Attributes:
        num_layers (int): Number of transformer layers in the model
        embed_dim (int): Dimension of the token embeddings
        attention_heads (int): Number of attention heads in each transformer layer
        alphabet (str): Token alphabet/vocabulary type (default: 'ESM-1b')
        token_dropout (bool): Whether to apply token dropout during training (default: True)
    """
    num_layers: int
    embed_dim: int
    attention_heads: int
    alphabet: str
    token_dropout: bool

def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict




if __name__ == "__main__":
    # 1. Path to the assets
    model_files_path = Path("./esmnox_model_files")

    # 2. Set the device to CPU for torch
    device = torch.device("cpu")

    # 3. Load the ESM2 model weights
    model_data = torch.load(model_files_path / "esm2_t36_3B_UR50D.pt")
    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = Alphabet.from_architecture("ESM-1b")

    # 4. Load the args required to build the ESM2 model
    args = ModelArgs(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    config_dict = {
        'num_layers': args.num_layers,
        'embed_dim': args.embed_dim,
        'attention_heads': args.attention_heads,
        'alphabet': args.alphabet.__class__.__name__,
        'token_dropout': args.token_dropout
    }
    with open(model_files_path / "esm2_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # 5. Build the ESM2 model in Equinox
    model = ESM2(
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        attention_heads=args.attention_heads,
        alphabet=args.alphabet,
        token_dropout=args.token_dropout,
        key=jax.random.PRNGKey(1),
    )

    # 6. Port weights from torch to the Equinox model
    model = port_weights_from_torch(state_dict, model)

    # 7. Serialize the Equinox model so that we can load it directly
    # eqx.tree_serialise_leaves(model_files_path / "esm2_jax_port_fast.eqx", model)
    # 8. Save the model configuration to JSON for future reference
    

