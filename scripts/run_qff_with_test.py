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
            "decay_interval":10000,
            "decay_base": 0.33,
            "nested": {
                "otype": "Adam",
                "learning_rate": 1e-2,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-15,
                "l2_reg": 1e-6,
            }
        }
    },
    "encoding": {
        "otype": "QFF1",
        "n_quants": 80,
        "n_features": 4,
        "n_frequencies": 4,
        "log2_min_freq": 0,
        "log2_max_freq": 4,
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
    parser.add_argument("--test_scene_path", type=str, required=True, help="JSON scene for testing.")
    parser.add_argument("--qff_type", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--min_freq", type=int, default=0)
    parser.add_argument("--n_quants", type=int, default=80)
    parser.add_argument("--max_freq", type=int, default=3)
    parser.add_argument("--num_freq", type=int, default=4)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=4, help="Rank of the QFF encoding. For QFF3, this is always 1.")
    parser.add_argument("--n_features", type=int, default=4, help="Number of features to use in the QFF encoding. For now, only supports 4.")
    parser.add_argument("--n_steps", type=int, default=50000, help="Number of steps to train for before quitting.")
    parser.add_argument("--test_background", type=str, choices=['black', 'white'], default='white', help="Set background color for testing images")

    return parser.parse_args()


def main(args):
    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    scene_json = args.scene_path

    testbed.load_training_data(scene_json)
    default_conf['encoding']['otype'] = f'QFF{args.qff_type}'
    rank = args.rank if args.qff_type != 3 else 1
    default_conf['encoding']['log2_min_freq'] = args.min_freq
    default_conf['encoding']['log2_max_freq'] = args.max_freq
    default_conf['encoding']['n_frequencies'] = int(args.num_freq)
    default_conf['encoding']['n_quants'] = int(args.n_quants)
    default_conf['encoding']['rank'] = rank
    default_conf['encoding']['n_features'] = min(int(args.n_features), 4)
    print(default_conf['encoding'])


    with open(output_path/ f'{args.run_name}.json', 'w') as f:
        json.dump(default_conf, f, indent=4)

    testbed.reload_network_from_file(str(output_path / f'{args.run_name}.json'))

    testbed.nerf.sharpen = 0.0
    testbed.exposure = 0.0
    testbed.shall_train = True
    testbed.nerf.render_with_lens_distortion = True
    testbed.render_near_distance = max(args.near, 0.0)
    testbed.training_batch_size = 2 ** 18
    n_steps = args.n_steps

    default_conf['optimizer']['nested']['decay_start'] = n_steps * 2 // 5
    default_conf['optimizer']['nested']['decay_interval'] = n_steps // 5

    old_training_step = 0

    # training
    tqdm_last_update = 0
    tic = time.monotonic()
    with tqdm(desc="Training", total=n_steps, unit="steps", dynamic_ncols=True) as t:
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
    toc = time.monotonic()
    elapsed = toc - tic

    testbed.save_snapshot(str(output_path / f"{args.run_name}.ingp"), False)
    bake(output_path / f"{args.run_name}.ingp", output_path / f"{args.run_name}.qff")

    totmse = 0
    totpsnr = 0
    totssim = 0
    totlpips_alex = 0
    totlpips_vgg = 0
    totcount = 0

    # Evaluate metrics on black background
    if args.test_background == 'black':
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    else:
        testbed.background_color = [1.0, 1.0, 1.0, 1.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.render_min_transmittance = 1e-4

    testbed.shall_train = False
    testbed.load_training_data(args.test_scene_path)

    # create output directory for images
    (output_path / 'test_renderings').mkdir(exist_ok=True, parents=True)

    with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame", dynamic_ncols=True) as t:
        for i in t:
            resolution = testbed.nerf.training.dataset.metadata[i].resolution
            testbed.render_ground_truth = True
            testbed.set_camera_to_training_view(i)
            ref_image = testbed.render(resolution[0], resolution[1], 1, True)
            testbed.render_ground_truth = False
            image = testbed.render(resolution[0], resolution[1], spp, True)
            write_image(str(output_path / 'test_renderings' / f"{i}.png"), image)

            A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
            R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)

            # compute metrics
            mse = float(compute_error("MSE", A, R))
            ssim = float(compute_error("SSIM", A, R))
            lpips_alex = float(rgb_lpips(A, R, 'alex'))
            lpips_vgg = float(rgb_lpips(A, R, 'vgg'))

            # aggregate metrics
            totssim += ssim
            totmse += mse
            totlpips_alex += lpips_alex
            totlpips_vgg += lpips_vgg

            psnr = mse2psnr(mse)
            totpsnr += psnr
            totcount = totcount+1
            t.set_postfix(psnr = totpsnr/(totcount or 1))

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    lpips_alex = totlpips_alex / (totcount or 1)
    lpips_vgg = totlpips_vgg / (totcount or 1)
    with open(output_path / 'results.json', 'w') as f:
            json.dump({
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips_alex": lpips_alex,
                    "lpips_vgg": lpips_vgg,
                    "elapsed": elapsed,
            }, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
