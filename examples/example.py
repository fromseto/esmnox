import json
import jax
import equinox as eqx

from esmnox.model.esm2 import ESM2
from esmnox.data import Alphabet

args = json.loads(open("esm2_config.json").read())
alphabet = Alphabet.from_architecture("ESM-1b")

model = ESM2(
    num_layers=args.num_layers,
    embed_dim=args.embed_dim,
    attention_heads=args.attention_heads,
    alphabet=alphabet,
    token_dropout=args.token_dropout,
    key=jax.random.PRNGKey(1),
)

model_loaded = eqx.tree_deserialise_leaves("esm2_jax_port_fast.eqx", model)

print(model_loaded)
breakpoint()