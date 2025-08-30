import dataclasses
from typing import Tuple

import jax

from .utils import ParamSpec, ParamInitializer
from .utils import jax_pytree_struct, format_repr


@jax_pytree_struct
class Linear(ParamInitializer):
    in_features: int = dataclasses.field(metadata=dict(static=True))
    out_features: int = dataclasses.field(metadata=dict(static=True))
    weight: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec
    use_bias: bool = dataclasses.field(default=False, metadata=dict(static=True))
    weight_logical_axes: Tuple[str, str] = dataclasses.field(
        default=(None, None), metadata=dict(static=True)
    )
    bias_logical_axes: Tuple[str] = dataclasses.field(
        default=(None,), metadata=dict(static=True)
    )

    @classmethod
    def _param_specs(
        cls,
        cfg,
        in_features,
        out_features,
        use_bias=False,
        weight_logical_axes=(None, None),
        bias_logical_axes=(None,),
    ):
        def kernel_init(*out_axes):
            return jax.nn.initializers.he_normal(in_axis=0, out_axis=out_axes)

        weight = ParamSpec(
            shape=(in_features, out_features),
            dtype=cfg.dtype,
            logical_axes=weight_logical_axes,
            initializer=kernel_init(1),
        )
        if use_bias:
            bias = ParamSpec(
                shape=(out_features,),
                dtype=cfg.dtype,
                logical_axes=bias_logical_axes,
                initializer=jax.nn.initializers.zeros,
            )
        else:
            bias = None
        return Linear(
            weight=weight,
            bias=bias,
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            weight_logical_axes=weight_logical_axes,
            bias_logical_axes=bias_logical_axes,
        )

    @classmethod
    def init(
        cls,
        key,
        cfg,
        in_features,
        out_features,
        *,
        use_bias=False,
        weight_logical_axes=("linear_in", "linear_out"),
        bias_logical_axes=("linear_out",),
    ):
        return cls._init_fn(
            key,
            cfg,
            in_features,
            out_features,
            use_bias=use_bias,
            weight_logical_axes=weight_logical_axes,
            bias_logical_axes=bias_logical_axes,
        )

    def __repr__(self):
        return format_repr(self)
