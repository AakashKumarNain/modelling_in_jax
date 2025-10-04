import dataclasses

import jax
import jax.numpy as jnp

from .utils import ParamSpec, ParamInitializer
from .utils import jax_pytree_struct, layer_repr, kernel_init


@jax_pytree_struct
class Attention(ParamInitializer):
    d_emb: int = dataclasses.field(metadata=dict(static=True))
    q_heads: int = dataclasses.field(metadata=dict(static=True))
    kv_heads: int = dataclasses.field(metadata=dict(static=True))
    head_dim: int = dataclasses.field(metadata=dict(static=True))
    wqkv: jax.Array | ParamSpec
    wo: jax.Array | ParamSpec
    wqkv_logical_axes: tuple = dataclasses.field(
        default=(None, None), metadata=dict(static=True)
    )
    wo_logical_axes: tuple = dataclasses.field(
        default=(None, None, None), metadata=dict(static=True)
    )

    @classmethod
    def param_specs(
        cls,
        d_emb,
        q_heads,
        kv_heads,
        head_dim,
        dtype=jnp.float32,
        wqkv_logical_axes=(None, None),
        wo_logical_axes=(None, None, None),
    ):
        total_head_dim = (q_heads + 2 * kv_heads) * head_dim

        wqkv = ParamSpec(
            shape=(d_emb, total_head_dim),
            dtype=dtype,
            logical_axes=wqkv_logical_axes,
            initializer=kernel_init(1),
        )
        wo = ParamSpec(
            shape=(q_heads, head_dim, d_emb),
            dtype=dtype,
            logical_axes=wo_logical_axes,
            initializer=kernel_init(1),
        )

        return Attention(
            d_emb=d_emb,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            wqkv=wqkv,
            wo=wo,
            wqkv_logical_axes=wqkv_logical_axes,
            wo_logical_axes=wo_logical_axes,
        )

    @classmethod
    def init(
        cls,
        key,
        cfg,
        d_emb,
        q_heads,
        kv_heads,
        head_dim,
        dtype=jnp.float32,
        wqkv_logical_axes=(None, None),
        wo_logical_axes=(None, None, None),
    ):
        return cls._init_fn(
            key,
            cfg,
            d_emb=d_emb,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            wqkv_logical_axes=wqkv_logical_axes,
            wo_logical_axes=wo_logical_axes,
        )

    def __repr__(self):
        return layer_repr(self)
