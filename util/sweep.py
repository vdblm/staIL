from pathlib import Path

from loguru import logger
import itertools
import os
import json
import pickle
import wandb
import numpy as np
from scipy.io import savemat


def common_config(configs):
    items = [set(c.items()) for c in configs]
    common_items = items[0].intersection(*items[1:])
    return dict(common_items)


def diff_config(base_config, config):
    diff = {k: v for (k, v) in config.items() if v != base_config.get(k, None)}
    return diff


def make_name(diff):
    items = [f'{k}_{v}' for (k, v) in diff.items()]
    if not items:
        return 'base'
    else:
        return '__'.join(items)


def log_info(dict):
    for (k, v) in dict.items():
        logger.opt(colors=True).info(f'    <blue>{k}</blue>: {v}')


def run_all(train_fn, configs, name=None, output_base=None, use_wandb=False, args=None):
    sweep_base = name or "manual_run"
    if use_wandb:
        assert len(configs) == 1
        config = configs[0]
        wandb.init(project="safe_imitation_learning", config=config)
        sweep_base = f'{wandb.run.name}__{wandb.run.id}'
    if output_base and output_base != "None":
        for i in itertools.count():
            sweep_name = f'{sweep_base}_{i}' if i > 0 else sweep_base
            if not os.path.exists(os.path.join(output_base, sweep_name)):
                break
        output_path = os.path.join(output_base, sweep_name)
        logger.opt(colors=True).info(f"Using output directory <blue>{output_path}</blue>")
        os.makedirs(output_path, exist_ok=True)
        sweeps = len(configs)

    base_config = common_config(configs)
    for (i, config) in enumerate(configs):
        config_diff = diff_config(base_config, config)
        # Diff the config and log the sweep 
        if sweeps > 1:
            logger.opt(colors=True).info(f"<red>Sweep run</red> <blue>{i + 1}</blue>/<blue>{sweeps}</blue>")
            log_info(config_diff)
        stats, final_params, expert_params = train_fn(config)
        if use_wandb:
            for k, v in stats.items():
                wandb.run.summary[k] = v
        # Save the output as JSON
        if output_path:
            run_name = make_name(config_diff)
            output = {'name': run_name, 'config': config, 'config_diff': config_diff, 'stats': stats}
            output_json_file = os.path.join(output_path, f'{run_name}.json')
            output_weight_file = os.path.join(output_path, f'{run_name}.pk')
            output_weight_mat = os.path.join(output_path, f'{run_name}.mat')

            expert_output = os.path.join(output_path, f'{run_name}_expert.pk')
            logger.opt(colors=True).info(f"Saving stats to <blue>{output_json_file}</blue>")
            with open(output_json_file, 'w') as f:
                json.dump(output, f)
            logger.opt(colors=True).info(f"Saving weights to <blue>{output_weight_file}</blue>")

            with open(output_weight_file, "wb") as f:
                pickle.dump(final_params, f)
            with open(output_weight_mat, "wb") as f:
                savemat(f, get_numpy_weights(final_params))
            with open(expert_output, "wb") as f:
                pickle.dump(expert_params, f)
            results = {'delta_err': [stats['test']['delta_err']], 'mean_imitation_err': [stats['test']['mean_imitation_err']],
                       'main': [args.main], 'noise': [args.noises], 'seed': [config.seed], 'lr': [config.learning_rate],
                       'noisy_demo_std': [config.noisy_demo_std], 'gamma': [config.gamma],
                       'lip_const': [config.lip_const], 'jacob_lambda': [config.jacob_lambda]}
            save_results(results, 'summary_gamma.csv')


def read_all(output_base, **sweep_names):
    sweeps = []
    for s in sweep_names:
        sweep_path = os.path.join(output_base)
    return sweeps


import pandas as pd


def save_results(results: dict, save_path: str):
    """Save the results to a csv file"""
    df = pd.DataFrame(results)
    if Path(save_path).is_file():
        df = pd.concat([pd.read_csv(save_path), df])
    df.to_csv(save_path, index=False)


def get_numpy_weights(params):
    layers = list(params['params'].keys())
    weights = []
    for layer in layers:
        weights.append(np.array(params['params'][layer]['kernel']).astype('float64').T)
    return {'weights': weights}
