import flax.linen as nn
import flax.linen.initializers
from lipschitz.layers import BjorckLinear
from lipschitz.activations import max_min
from typing import Sequence, Callable, Optional, Any, Tuple
import jax
import jax.numpy as jnp
from jax import custom_vjp

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class MLP(nn.Module):
    features: Sequence[int]
    activation: str
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = flax.linen.initializers.orthogonal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = flax.linen.initializers.zeros
    batch_norm: bool = False
    use_running_average: Optional[bool] = None
    lipschitz: bool = False,
    lipschitz_constant: float = 1.0

    @nn.compact
    def __call__(self, x, use_running_average=None):
        if not self.batch_norm:
            use_running_average = False
        use_running_average = nn.merge_param('use_running_average', self.use_running_average, use_running_average)
        if self.activation == 'relu':
            activation_fn = jax.nn.relu
        elif self.activation == 'gelu':
            activation_fn = jax.nn.gelu
        elif self.activation == 'tanh':
            activation_fn = jax.nn.tanh
        elif self.activation == 'maxmin':
            activation_fn = lambda inp: max_min(inp, inp.shape[-1] // 2)
        else:
            raise ValueError(f"Expected relu, tanh, or gelu, got {self.activation}")
        for (i, feat) in enumerate(self.features):
            if self.lipschitz:
                x = BjorckLinear(feat, name=f"layer_{i}", kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
                if x.shape[-1] % 2:  # and not inp.shape[-1] == 1:
                    raise ValueError(f"Number of features ({x.shape[-1]}) is not a multiple of 2")
                if x.shape[-1] == 1:
                    x = jnp.concatenate([x, x], axis=-1)
                x = max_min(x, x.shape[-1] // 2)
                # if inp.shape[-1] == 1:
                #     inp = (inp[:, 0] + inp[:, 1]) / 2.
                if i == len(self.features) - 1:
                    x = x * self.lipschitz_constant
            else:
                # print(x.shape)
                x = nn.Dense(feat, name=f"layer_{i}", kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
                if i != len(self.features) - 1:
                    if self.batch_norm:
                        x = nn.BatchNorm(use_running_average=use_running_average,
                                         momentum=0.9,
                                         epsilon=1e-5, axis_name='batch')(x)
                    x = activation_fn(x)
        return x


def safe_norm(x, min_norm, *args, **kwargs):
    """Returns jnp.maximum(jnp.linalg.norm(inp), min_norm) with correct gradients.
      The gradients of jnp.maximum(jnp.linalg.norm(inp), min_norm) at 0.0 is NaN,
      because jax will evaluate both branches of the jnp.maximum.
      The version in this function will return the correct gradient of 0.0 in this
      situation.
      Args:
      inp: jax array.
      min_norm: lower bound for the returned norm.
    """
    norm = jnp.linalg.norm(x, *args, **kwargs)
    x = jnp.where(norm < min_norm, jnp.ones_like(x), x)
    return jnp.where(norm < min_norm, min_norm,
                     jnp.linalg.norm(x, *args, **kwargs))


@custom_vjp
def clip_gradient(x, lo, hi):
    return x  # identity function


def clip_gradient_fwd(x, lo, hi):
    return x, (lo, hi)  # save bounds as residuals


def clip_gradient_bwd(res, g):
    lo, hi = res
    return jnp.clip(g, lo, hi), None, None  # use None to indicate zero cotangents for lo and hi


clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)


def scale_clip_grads(g, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = safe_norm(g, 1e-9)
    return jnp.where(norm < max_norm, g, g * max_norm / (norm))


@custom_vjp
def scale_clip_bp(x, max_norm):
    return x  # identity function


def scale_clip_bp_fwd(x, max_norm):
    return x, max_norm  # save bounds as residuals


def scale_clip_bp_bwd(max_norm, g):
    return scale_clip_grads(g, max_norm), None  # use None to indicate zero cotangents for lo and hi


class LipschitzSchedule:
    def __init__(self, init_value, wait_steps, end_value, rate=0.75):
        self.lip_const = init_value
        self.wait_steps = wait_steps
        self.end_value = end_value
        self.rate = rate
        self.min_loss = 1e9
        self.waited = 0

    def __call__(self, loss):
        if loss < self.min_loss and self.waited < self.wait_steps:
            self.waited += 1
        elif loss < self.min_loss:
            self.min_loss = loss
            self.waited = 0
            self.lip_const = max(self.lip_const * self.rate, self.end_value)
        return self.lip_const


scale_clip_bp.defvjp(scale_clip_bp_fwd, scale_clip_bp_bwd)
