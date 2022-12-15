from util.models import MLP
import jax
import jax.numpy as jnp

from util.plot_trajectory import plot_trajectories, plot_gamma


def data(n_samples, k):
    k1, k2 = jax.random.split(k)
    x = jax.random.normal(k1, (n_samples, 6))
    W = jax.random.normal(k2, (6, 1))
    y = jnp.dot(x, W)
    return x, y


def test_lip_networks():
    rng = jax.random.PRNGKey(1234)
    model = MLP((64, 64, 64, 6), 'relu', lipschitz=False, lipschitz_constant=5)
    rng, key = jax.random.split(rng)
    params = model.init(key, jnp.zeros(6), use_running_average=False)
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

    mean_jac = lambda params: jax.vmap(jax.jacobian(model.apply, argnums=(1)), in_axes=(None, 0))(params, x_train).mean()
    print(jax.grad(mean_jac)(params))


if __name__ == '__main__':
    plot_trajectories('./results/default_None_noisy_demo_2_0.0001/', length=100, traj_num=50)
    # test_lip_networks()