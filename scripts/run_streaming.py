#!/usr/bin/env python3

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *

from tqdm import tqdm
import websocket
import pyngp as ngp # noqa
from .parse_ingp import parse_ingp_file

def parse_args():
    parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")
    parser.add_argument("--scene", default="", help="Path to folder with transforms.json", required=True)
    parser.add_argument("--network", default="configs/nerf/qff.json")
    parser.add_argument("--output_folder", default="", help="Path to output files for streaming")
    return parser.parse_args()

import multiprocessing as mp
from multiprocessing import Queue, Lock, Value

def nerf_process(scene, network, output_path, lock):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR
    testbed.load_training_data(scene)
    testbed.reload_network_from_file(network)

    testbed.nerf.sharpen = float(0.0)
    testbed.exposure = 0.0
    testbed.shall_train = True
    testbed.nerf.render_with_lens_distortion = True

    step = 0
    while testbed.frame():
        step += 1
        if step % 16:
            lock.acquire()
            testbed.save_snapshot(output_path)
            num.value = 1
            lock.release()

def server_process(output_path):

    return

if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_folder)

    num = Value('i', 0.0)
    lock = Lock()

    nerf_p = mp.Process(target=nerf_process, args=(args.scene, args.network, output_path / 'base.ingp', num, lock))
    nerf_p.start()

    server_p = mp.Process()
    server_p.start()


