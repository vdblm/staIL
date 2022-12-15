"""Plot x v.s. t for the expert and the learned policy."""
import json
import os
import pickle

import jax
from attrdict import AttrDict
from matplotlib import pyplot as plt
import seaborn as sns

from imitation_learning import NormalGenerator
from imitation_learning.envs import make_system, make_dataset
from util.models import safe_norm
from util.rng import PRNGSequence
from jax.random import PRNGKey

import jax.numpy as jnp
import numpy as np


def _plot_trajectories(config, learned_params, length, output_path, traj_num=10):
    rng = PRNGSequence(PRNGKey(config.seed))
    generator, expert, policy, init_params, expert_noisy_dem, expert_params = make_system(config, next(rng))
    generator = generator.with_key(next(rng))
    # generator = generator.with_init_gen(NormalGenerator(config.state_dim, mean=config.test_mean_normal))
    expert_generator = generator.with_length(length)
    policy_generator = generator.with_policy(lambda x, r: policy(learned_params, x))
    delta_err_mean = 0
    for n in range(traj_num):
        rng_key = next(rng)
        expert_trajs = expert_generator._generate(rng_key)
        policy_trajs = policy_generator._generate(rng_key)
        final_diff = jax.vmap(lambda x: safe_norm(x, 1e-8))(expert_trajs['x'] - policy_trajs['x'])
        # Take the L2 norm of the diff
        delta_err = jnp.max(final_diff)
        delta_err_mean += delta_err
    # np.array(diffs).dump(os.path.join(output_path, 'traj_diffs.npy'))

    # for i in range(expert_trajs['x'].shape[-1]):
    #     sns.displot(diffs[:, i], kind='kde')
    #     plt.savefig(os.path.join(output_path, f'diffs_{i}.png'))
    #     plt.close()

    # for i in range(expert_trajs['x'].shape[-1]):
    #     traj_path = os.path.join(output_path, f'traj_{i}.png')
    #     plt.plot(jnp.arange(length), expert_trajs['x'][:, i], label='expert')
    #     plt.plot(jnp.arange(length), policy_trajs['x'][:, i], label='learned')
    #     plt.legend()
    #     plt.savefig(traj_path)
    #     plt.close()
    #
    #
    # learned_policy = lambda x: policy(learned_params, x)
    # policy_us = jax.vmap(learned_policy)(expert_trajs['x'])
    #
    # for j in range(expert_trajs['u'].shape[-1]):
    #     action_path = os.path.join(output_path, f'action_{j}.png')
    #     plt.plot(jnp.arange(length), expert_trajs['u'][:, j], label='expert')
    #     plt.plot(jnp.arange(length), policy_us[:, j], label='learned')
    #     plt.legend()
    #     plt.savefig(action_path)
    #     plt.close()


def _plot_trajectories(config, learned_params, length, output_path, traj_num=10):
    rng = PRNGSequence(PRNGKey(config.seed))
    generator, expert, policy, init_params, expert_noisy_dem, expert_params = make_system(config, next(rng))
    generator = generator.with_key(next(rng))
    # generator = generator.with_init_gen(NormalGenerator(config.state_dim, mean=config.test_mean_normal))
    expert_generator = generator.with_length(length)
    policy_generator = generator.with_policy(lambda x, r: policy(learned_params, x))
    delta_err_mean = 0
    for n in range(traj_num):
        rng_key = next(rng)
        expert_trajs = expert_generator._generate(rng_key)
        policy_trajs = policy_generator._generate(rng_key)
        final_diff = jax.vmap(lambda x: safe_norm(x, 1e-8))(expert_trajs['x'] - policy_trajs['x'])
        # Take the L2 norm of the diff
        delta_err = jnp.max(final_diff)
        delta_err_mean += delta_err

    return delta_err_mean / traj_num


def plot_gamma(file_name):
    gammas = [0.1, 0.5, 1, 2, 3]
    jacob_errs = []
    default_errs = []
    for gamma in gammas:
        run_path = f'/h/vdblm/projects/staIL/results/jacob_reg_0.1_noisy_demo_1.0_gamma_{gamma}'
        jacob_err = plot_trajectories(run_path, 100, 10)
        jacob_errs.append(jacob_err)

        run_path = f'/h/vdblm/projects/staIL/results/default_None_noisy_demo_1.0_gamma_{gamma}'
        def_err = plot_trajectories(run_path, 100, 10)
        default_errs.append(def_err)

    print(jacob_errs)
    print(default_errs)
    plt.plot(gammas, jacob_errs, label='Jacobian')
    plt.plot(gammas, default_errs, label='Baseline')
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def plot_trajectories(run_path, length, traj_num):
    base_path = os.path.join(run_path, f'base.json')
    learned_params_path = os.path.join(run_path, f'base.pk')
    with open(base_path, 'rb') as f:
        config = AttrDict(json.load(f)['config'])
    with open(learned_params_path, 'rb') as f:
        learned_params = pickle.load(f)

    return _plot_trajectories(config, learned_params, length, run_path, traj_num)
