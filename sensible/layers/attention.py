import dataclasses

import jax

from .utils import ParamSpec, ParamInitializer
from .utils import jax_pytree_struct, format_repr


@jax_pytree_struct
class Attention(ParamInitializer):
    d_emb: int = dataclasses.field(metadata=dict(static=True))
    q_heads: int = dataclasses.field(metadata=dict(static=True))
    kv_heads: int = dataclasses.field(metadata=dict(static=True))
    head_dim: int = dataclasses.field(metadata=dict(static=True))
    wqkv: jax.Array | ParamSpec
    wo: jax.Array | ParamSpec

    @classmethod
    def _param_specs(cls, cfg, d_emb, q_heads, kv_heads, head_dim):
        total_head_dim = (q_heads + 2 * kv_heads) * head_dim

        def kernel_init(*out_axes):
            return jax.nn.initializers.he_normal(in_axis=0, out_axis=out_axes)

        wqkv = ParamSpec(
            shape=(d_emb, total_head_dim),
            dtype=cfg.dtype,
            logical_axes=(None, None),  # TODO: Get it from config directly
            initializer=kernel_init(1),
        )
        wo = ParamSpec(
            shape=(q_heads, head_dim, d_emb),
            dtype=cfg.dtype,
            logical_axes=(None, None),  # TODO: Get it from config directly
            initializer=kernel_init(1),
        )

        return Attention(
            d_emb=d_emb,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            wqkv=wqkv,
            wo=wo,
        )

    @classmethod
    def init(cls, key, cfg, d_emb, q_heads, kv_heads, head_dim):
        return cls._init_fn(
            key,
            cfg,
            d_emb=d_emb,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

    def __repr__(self):
        return format_repr(self)
