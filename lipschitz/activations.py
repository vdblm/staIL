# Source: https://github.com/cemanil/LNets

import jax
import jax.numpy as jnp
from functools import partial

from typing import Any

Array = Any


@partial(jax.jit, static_argnames=['num_units'])
def max_min(x: Array, num_units: Array) -> Array:
    x_size = list(x.shape)
    num_channels = x_size[-1]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    x_size[-1] = -1
    x_size += [num_channels // num_units]
    x = jnp.reshape(x, tuple(x_size))
    maxmin = jnp.concatenate([jnp.max(x, axis=-1), jnp.min(x, axis=-1)], axis=-1)
    return maxmin
