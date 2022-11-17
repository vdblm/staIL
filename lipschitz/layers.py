# Source: https://github.com/cemanil/LNets

import abc
import dataclasses
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

from flax.linen.initializers import orthogonal, lecun_uniform
from flax.linen.module import compact
from flax.linen.module import Module
from flax.linen.dtypes import promote_dtype
from jax import eval_shape
from jax import lax
from jax import ShapedArray
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]


def bjorck_orthonormalize_order1(weight, beta, iters):
    for _ in range(iters):
        w_t_w = jnp.matmul(jnp.transpose(weight), weight)
        weight = (1 + beta) * weight - beta * jnp.matmul(weight, w_t_w)

    return weight


def get_safe_bjorck_scaling(weight):
    return jnp.sqrt(weight.shape[0] * weight.shape[1])


class BjorckLinear(Module):
    """A Bjorck linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """
    features: int
    use_bias: bool = True
    beta: float = 0.5
    iters: int = 20
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = orthogonal()  # TODO scale with 1 / sqrt(fan_in)
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = lecun_uniform()

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.param('kernel',
                            self.kernel_init,
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                              self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        scaling = get_safe_bjorck_scaling(kernel)
        y = lax.dot_general(inputs, bjorck_orthonormalize_order1(kernel / scaling, self.beta, self.iters),
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y
