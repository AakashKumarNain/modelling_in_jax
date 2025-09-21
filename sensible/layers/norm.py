import dataclasses
from typing import Tuple

import jax
# Ensure to use the right pallas kernel depending on the accelerator type

from .utils import ParamSpec, ParamInitializer
from .utils import jax_pytree_struct, format_repr


@jax_pytree_struct
class RMSNorm(ParamInitializer):
    """Builds parameters for RMSNorm pallas kernel.

    **Note:** Ideally the `bias` parameter should be optional, but the way
    pallas kernel is written as of now makes bias a mandatory parameter.
    TODO: Raise this as an issue on JAX repo.
    """

    weight: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec
    shape: tuple = dataclasses.field(metadata=dict(static=True))
    eps: float = dataclasses.field(default=1e-5, metadata=dict(static=True))
    weight_logical_axes: Tuple[str,] = dataclasses.field(
        default=(None,), metadata=dict(static=True)
    )
    bias_logical_axes: Tuple[str,] = dataclasses.field(
        default=(None,), metadata=dict(static=True)
    )

    @classmethod
    def _param_specs(
        cls,
        cfg,
        shape,
        eps=1e-5,
        affine=True,
        weight_logical_axes=(None,),
        bias_logical_axes=(None,),
    ):
        weight = ParamSpec(
            shape=shape,
            dtype=cfg.dtype,
            logical_axes=weight_logical_axes,
            initializer=jax.nn.initializers.constant(1.0),
        )
        bias = ParamSpec(
            shape=shape,
            dtype=cfg.dtype,
            logical_axes=bias_logical_axes,
            initializer=jax.nn.initializers.constant(0.0),
        )
        return RMSNorm(
            weight=weight,
            bias=bias,
            shape=shape,
            eps=eps,
            weight_logical_axes=weight_logical_axes,
            bias_logical_axes=bias_logical_axes,
        )

    @classmethod
    def init(
        cls,
        key,
        cfg,
        shape,
        *,
        affine=False,
        eps=1e-5,
        weight_logical_axes=(None,),
        bias_logical_axes=(None,),
    ):
        return cls._init_fn(
            key,
            cfg,
            shape,
            eps=eps,
            affine=affine,
            weight_logical_axes=weight_logical_axes,
            bias_logical_axes=bias_logical_axes,
        )

    def __repr__(self):
        return format_repr(self)


## Nuances to be taken care of with pallas rmsnorm kernel
# 1. Pallas kernels are jitted by default. Ensure to not double-jit it when implementing models
# 2. The inputs to the pallas kernel operates on an extra dimensions. Say if the inputs are of
#   shape [4, 32], then weights and bias would be of shape (32, ), and inputs have to be expanded
#   as jnp.expand_dims(x, 0).astype(jnp.float32). Ensure to use `out[0]` before returning the results
# 3. Ensure weight and bias are always in fp32, and so is the input.
