import dataclasses

import jax
import jax.numpy as jnp

from .utils import ParamSpec, ParamInitializer
from .utils import jax_pytree_struct, layer_repr, kernel_init


@jax_pytree_struct
class Embedding(ParamInitializer):
    vocab_size: int = dataclasses.field(metadata=dict(static=True))
    embed_dim: int = dataclasses.field(metadata=dict(static=True))
    embedding: jax.Array | ParamSpec
    embedding_logical_axes: tuple = dataclasses.field(
        default=("vocab_in", "embed_dim"), metadata=dict(static=True)
    )

    @classmethod
    def param_specs(
        cls,
        vocab_size,
        embed_dim,
        dtype=jnp.float32,
        embedding_logical_axes=(None, None),
    ):
        embedding = ParamSpec(
            shape=(vocab_size, embed_dim),
            dtype=dtype,
            logical_axes=embedding_logical_axes,
            initializer=kernel_init(0, 1),
        )
        return Embedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding=embedding,
            embedding_logical_axes=embedding_logical_axes,
        )

    @classmethod
    def init(cls,
        key,
        cfg,
        vocab_size,
        embed_dim,
        dtype=jnp.float32,
        embedding_logical_axes=(None, None),
    ):
        return cls._init_fn(
            key,
            cfg,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dtype=dtype,
            embedding_logical_axes=embedding_logical_axes,
        )

    def __repr__(self):
        return layer_repr(self)
