from models import MLP
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(1234)


def data(n_samples, k):
    k1, k2 = jax.random.split(k)
    x = jax.random.normal(k1, (n_samples, 5))
    W = jax.random.normal(k2, (5, 1))
    y = jnp.dot(x, W)
    return x, y


if __name__ == '__main__':
    model = MLP((64, 64, 64, 5), 'relu', lipschitz=True, lipschitz_constant=5)
    rng, key = jax.random.split(rng)
    params = model.init(key, jnp.zeros(5), use_running_average=False)
    epochs = 101

    @jax.jit
    def mse(params, x, y):
        y_hat = model.apply(params, x)
        return jnp.mean((y - y_hat) ** 2)

    @jax.jit
    def update_params(params, lr, grads):
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return params

    loss_grad_fn = jax.value_and_grad(mse)
    x_train, y_train = data(1000, rng)
    for i in range(epochs):
        loss_val, grad = loss_grad_fn(params, x_train, y_train)
        params = update_params(params, 0.01, grad)
        if i % 10 == 0:
            print(f'Loss step {i}: ', loss_val)
