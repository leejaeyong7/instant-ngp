#!/usr/bin/env python3
import math
import argparse
from pathlib import Path
import numpy as np
import json
import copy
import cv2
from tqdm import tqdm
import msgpack
import zlib
import gzip

def try_decompress(data):

    try:
        zdata = zlib.decompress(data)
    except zlib.error as ze:
        try:
            gdata = gzip.decompress(data)
        except:
            return data
        return gdata
    return zdata
def morton3D(x, y, z):
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    z = z.astype(np.uint32)
    def expand_bits(v):
        v = (v * int('0x00010001', 16)) & int('0xFF0000FF', 16);
        v = (v * int('0x00000101', 16)) & int('0x0F00F00F', 16);
        v = (v * int('0x00000011', 16)) & int('0xC30C30C3', 16);
        v = (v * int('0x00000005', 16)) & int('0x49249249', 16);
        return v
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)
    return xx | (yy << 1) | (zz << 2)
def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    ingp_file = Path(args.ingp_file)

    with open(ingp_file, 'rb') as f:
        zf = f.read()
    zf = try_decompress(zf)
    try:
        ckpt = msgpack.unpackb(zf)
    except msgpack.ExtraData as e:
        print('Error parsing msgpack')

    # write to output folder
    enc_conf = ckpt['encoding']
    dnet_conf = ckpt['network']
    cnet_conf = ckpt['rgb_network']
    snapshot = ckpt['snapshot']

    C = enc_conf['n_features']
    Q = enc_conf['n_quants']
    F = enc_conf['n_frequencies']
    R = snapshot['density_grid_size']
    poses = [p['start'] for p in ckpt['snapshot']['nerf']['dataset']['xforms']]
    freqs = 2 ** ((np.arange(F).astype(np.float32) / (F - 1.0)) * enc_conf['log2_max_freq'] - enc_conf['log2_min_freq'])
    scene_json = {
        'freqs': freqs.tolist(),
        'num_freqs': enc_conf['n_frequencies'],
        'num_density_layers': dnet_conf['n_hidden_layers'] + 1,
        'num_color_layers': cnet_conf['n_hidden_layers'] + 1,
        'num_feats': C,
        'num_quants': Q,
        'qff_out': F*2*C,
        'density_bias': 0,
        'grid_res': R,
        'poses': [p for p in poses],
        'render_step': snapshot['bounding_radius'] / 1024 * math.sqrt(3),
        'up': snapshot['up_dir'],
    }

    with open(output_path / 'scene.json', 'w') as f:
        json.dump(scene_json, f)

    # write buffer files
    qff_raw_buffers = snapshot['params_binary'][-F*2*C*Q*Q*Q*2:]
    qff_buffers = np.frombuffer(qff_raw_buffers, np.float16).reshape(F, 2, Q, Q, Q, C)#.transpose(0, 1, 4, 3, 2, 5)
    for f in range(F):
        qff_buffers[f, 0].tofile(output_path / f'qff_encoding_{f}_sin.bin')
        qff_buffers[f, 1].tofile(output_path / f'qff_encoding_{f}_cos.bin')
    grid = np.frombuffer(snapshot['density_grid_binary'], np.float16)
    inds = np.mgrid[:R, :R, :R]
    m = morton3D(inds[0], inds[1], inds[2])
    grid_vals = grid.reshape(-1, R*R*R)[:, m.reshape(-1)]
    grid_vals[0].reshape(R, R, R).transpose(2, 1, 0).tofile(output_path / 'grid_texture.bin')

    # write MLP to files
    mlp_raw_buffers = snapshot['params_binary'][:-F*2*C*Q*Q*Q*2]
    mlp_buffers = np.frombuffer(mlp_raw_buffers, np.float16)

    # density network
    offset = 0
    dnet_i = dnet_conf['n_hidden_layers']
    dnet_w = dnet_conf['n_neurons']
    dnet_in = F*2*C
    dnet_out = 16
    for i in range(dnet_i + 1):
        if i == (dnet_i):
            dnet_w = dnet_out
        mlp_buffer = mlp_buffers[offset: offset + (dnet_in * dnet_w)]
        mlp_buffer = mlp_buffer.reshape(dnet_w, dnet_in).T
        mlp_buffer.astype(np.float32).tofile(output_path / f'qff_density_layer_{i}.bin')
        offset += dnet_in * dnet_w
        dnet_in = dnet_w

    # color network
    cnet_i = cnet_conf['n_hidden_layers']
    cnet_w = cnet_conf['n_neurons']
    cnet_in = dnet_out + 16
    cnet_out = 16
    for i in range(cnet_i + 1):
        if i == cnet_i:
            cnet_w = cnet_out

        mlp_buffer = mlp_buffers[offset: offset + (cnet_in * cnet_w)]
        mlp_buffer = mlp_buffer.reshape(cnet_w, cnet_in).T
        crop_in = None
        crop_out = None
        if i == 0:
            crop_in = 16 + 16
        if i == cnet_i:
            crop_out = 4
        mlp_buffer = mlp_buffer[:crop_in, :crop_out]
        mlp_buffer.astype(np.float32).tofile(output_path / f'qff_rgb_layer_{i}.bin')
        offset += cnet_in * cnet_w
        cnet_in = cnet_w
        if i == (cnet_i - 1):
            cnet_w = cnet_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts INGP file into ")
    parser.add_argument("--ingp_file", default="", help="path to the Record3D capture")
    parser.add_argument("--output_path", default="outputs", help="path to the Record3D capture")
    args = parser.parse_args()
    main(args)
