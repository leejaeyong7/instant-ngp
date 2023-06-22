import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pathlib import Path
import zipfile
import shutil
from r3d import r3d_to_transforms
from parse_ingp import parse_ingp_file
from tqdm import tqdm
import json

# importing instant-ngp 
from common import *
import pyngp as ngp # noqa

def train_qff(dataset_path, num_iters, checkpoint_path):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    network = 'configs/nerf/qff.json'
    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR
    testbed.load_training_data(str(dataset_path))
    testbed.reload_network_from_file(network)

    testbed.nerf.sharpen = float(0.0)
    testbed.exposure = 0.0
    testbed.shall_train = True
    testbed.nerf.render_with_lens_distortion = True
    testbed.nerf.training.near_distance = 0.0
    testbed.nerf.training.depth_supervision_lambda = 1.0
    testbed.nerf.training.optimize_extra_dims = True

    iterator = tqdm(range(num_iters), total=num_iters, dynamic_ncols=True)
    for iter in iterator:
        iterator.set_description(f'loss: {testbed.loss:.04f}')
        success = testbed.frame()
        if not success:
          break
    testbed.save_snapshot(str(checkpoint_path))

def r3d_to_baked(args):
    r3d_file = Path(args.r3d_file_path)
    output_path = Path(args.output_path)

    if args.decompress_path is not None:
        decompress_path = Path(args.decompress_path)
    else:
        decompress_path = output_path / 'decompressed'
    decompress_path.mkdir(exist_ok=True, parents=True)

    # first mv input.r3d file to input.zip
    with zipfile.ZipFile(r3d_file, 'r') as zip_ref:
        zip_ref.extractall(decompress_path)

    # process r3d output to instant-ngp friendly format
    dataset_path = decompress_path / 'processed'
    r3d_to_transforms(decompress_path, dataset_path, False, args.subsample)

    # run instant-qff for N iterations
    checkpoint_path = decompress_path / 'checkpoint.ingp'
    train_qff(dataset_path, args.num_train_iter, checkpoint_path)

    # bake the output file
    scene_json, files_to_write = parse_ingp_file(checkpoint_path)

    with open(output_path / 'scene.json', 'w') as f:
        json.dump(scene_json, f)

    for file_to_write, data_to_write in files_to_write.items():
        data_to_write.tofile(output_path / file_to_write)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--r3d_file_path', type=str, required=True, help='Path to raw r3d file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output baked files')
    parser.add_argument('--decompress_path', type=str, default=None, help='optional path to specify decompressing files. by default will use the "output path / decompressed"')
    parser.add_argument('--subsample', type=int, default=1, help='optional sampling used for r3d frames')
    parser.add_argument('--num_train_iter', type=int, default=25000, help='optional number of training iterations with instant-qff')
    args = parser.parse_args()
    r3d_to_baked(args)
