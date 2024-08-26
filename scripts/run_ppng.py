#!/usr/bin/env python3
import argparse
import os
import json

import numpy as np

import shutil
import time

from common import *
from scenes import *

import math
from tqdm import tqdm
from pathlib import Path

import pyngp as ngp # noqa

from bake import bake

default_conf = {
    "loss": {
        "otype": "Huber"
    },
    "optimizer": {
        "otype": "Ema",
        "decay": 0.95,
        "nested": {
            "otype": "ExponentialDecay",
            "decay_start": 20000,
            "decay_interval": 10000,
            "decay_base": 0.33,
            "nested": {
                "otype": "Adam",
                "learning_rate": 1e-2,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-15,
                "l2_reg": 1e-5
            }
        }
    },
    "encoding": {
        "otype": "PPNG1",
        "n_quants": 80,
        "n_features": 4,
        "n_frequencies": 4,
        "log2_min_freq": 0,
        "log2_max_freq": 6,
        "rank":1
    },
    "network": {
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 16,
        "n_hidden_layers": 0
    },
    "dir_encoding": {
        "otype": "Composite",
        "nested": [
            {
                "n_dims_to_encode": 3,
                "otype": "SphericalHarmonics",
                "degree": 4
            },
            {
                "otype": "Identity"
            }
        ]
    },
    "rgb_network": {
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 16,
        "n_hidden_layers": 1
    },
    "distortion_map": {
        "resolution": [32, 32],
        "optimizer": {
            "otype": "ExponentialDecay",
            "decay_start": 10000,
            "decay_interval": 5000,
            "decay_end": 25000,
            "decay_base": 0.33,
            "nested": {
                "otype": "Adam",
                "learning_rate": 1e-4,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-8
            }
        }
    },
    "envmap": {
        "loss": {
            "otype": "RelativeL2"
        },
        "optimizer": {
            "otype": "Ema",
            "decay": 0.99,
            "nested": {
                "otype": "ExponentialDecay",
                "decay_start": 10000,
                "decay_interval": 5000,
                "decay_base": 0.33,
                "nested": {
                    "otype": "Adam",
                    "learning_rate": 1e-2,
                    "beta1": 0.9,
                    "beta2": 0.99,
                    "beta3": 0.9,
                    "beta_shampoo": 0.0,
                    "epsilon": 1e-10,
                    "identity": 0.0001,
                    "cg_on_momentum": False,
                    "frobenius_normalization": True,
                    "l2_reg": 1e-10
                }
            }
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--scene_path", type=str, help="Path to the json file.")
    parser.add_argument("--ppng_type", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--n_quants", type=int, default=80)
    parser.add_argument("--max_freq", type=int, default=6)
    parser.add_argument("--num_freq", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2, help="Rank of the PPNG encoding. For PPNG3, this is always 1.")
    parser.add_argument("--n_features", type=int, default=4, help="Number of features to use in the PPNG encoding. For now, only supports 4.")
    parser.add_argument("--n_steps", type=int, default=50000, help="Number of steps to train for before quitting.")

    return parser.parse_args()


def main(args):
    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    scene_json = args.scene_path

    testbed.load_training_data(scene_json)
    default_conf['encoding']['otype'] = f'PPNG{args.ppng_type}'
    rank = args.rank if args.ppng_type != 3 else 1
    default_conf['encoding']['log2_min_freq'] = int(args.min_freq)
    default_conf['encoding']['log2_max_freq'] = int(args.max_freq)
    default_conf['encoding']['n_frequencies'] = int(args.num_freq)
    default_conf['encoding']['n_quants'] = int(args.n_quants)
    default_conf['encoding']['rank'] = int(args.rank)
    default_conf['encoding']['n_features'] = min(int(args.n_features), 4)

    with open(output_path/ f'{args.run_name}.json', 'w') as f:
        json.dump(default_conf, f, indent=4)

    testbed.reload_network_from_file(str(output_path / f'{args.run_name}.json'))

    testbed.nerf.sharpen = 0.0
    testbed.exposure = 0.0
    testbed.shall_train = True
    testbed.nerf.render_with_lens_distortion = True

    old_training_step = 0
    n_steps = args.n_steps

    # training
    tqdm_last_update = 0
    with tqdm(desc="Training", total=n_steps, unit="steps") as t:
        while testbed.frame():
            if testbed.training_step >= n_steps:
                break

            # Update progress bar
            if testbed.training_step < old_training_step or old_training_step == 0:
                old_training_step = 0
                t.reset()

            now = time.monotonic()
            if now - tqdm_last_update > 0.1:
                t.update(testbed.training_step - old_training_step)
                t.set_postfix(loss=testbed.loss)
                old_training_step = testbed.training_step
                tqdm_last_update = now

    testbed.save_snapshot(str(output_path / f"{args.run_name}.ingp"), False)
    bake(output_path / f"{args.run_name}.ingp", output_path / f"{args.run_name}.ppng")


if __name__ == "__main__":
    args = parse_args()
    main(args)
