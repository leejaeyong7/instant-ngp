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
import numpy as np
import cbor2

def grid_to_rle(grid):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    d, h, w = grid.shape
    assert h == w and w == d
    G = h
    grid = grid.reshape(-1)

    # Compute change indices
    diff = grid[1:] ^ grid[:-1]
    idx = diff.nonzero()[0]

    # Encode run length
    idx = np.concatenate(
        [
            np.array([0], dtype=idx.dtype),
            idx + 1,
            np.array([G*G*G], dtype=idx.dtype),
        ]
    )
    btw_idxs = idx[1:] - idx[:-1]
    counts = [] if grid[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())
    return counts

def rle_to_grid(rle, G):
    mask = np.empty(G*G*G, dtype=bool)
    idx = 0
    parity = False
    for count in rle:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(G, G, G)
    return mask

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

def get_dir_out(dir_encoding):
    if 'nested' in dir_encoding:
        dir_encodings = dir_encoding['nested']
        for dir_encoding in dir_encodings:
            if ('otype' in dir_encoding) and dir_encoding['otype'] == 'SphericalHarmonics':
                return (dir_encoding['degree']) ** 2
    else:
        if ('otype' in dir_encoding) and dir_encoding['otype'] == 'SphericalHarmonics':
            return (dir_encoding['degree']) ** 2
    raise NotImplementedError

def parse_ingp_file(ingp_file):
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
    dir_conf = ckpt['dir_encoding']
    dir_out = get_dir_out(dir_conf)
    snapshot = ckpt['snapshot']
    otype = enc_conf['otype']
    assert otype.startswith('QFF')
    qff_type = int(otype.replace('QFF', ''))

    C = enc_conf['n_features']
    Q = enc_conf['n_quants']
    F = enc_conf['n_frequencies']
    default_rank = 1 if qff_type == 3 else 4
    R = enc_conf.get('rank', default_rank)
    G = snapshot['density_grid_size']
    poses = [p['start'] for p in ckpt['snapshot']['nerf']['dataset']['xforms']]
    render_step = snapshot['bounding_radius'] / 1024 * math.sqrt(3)

    grid = np.frombuffer(snapshot['density_grid_binary'], np.float16)
    mips = grid.size // (G*G*G)

    inds = np.mgrid[:G, :G, :G]
    finfo = np.finfo(grid.dtype)
    m = morton3D(inds[0], inds[1], inds[2])
    min_alpha = 0.005
    flat_grids = grid.reshape(-1, G*G*G)
    grid_rles = []
    for mip in range(mips):
        grid_vals = flat_grids[mip, m.reshape(-1)]
        grid_texture = grid_vals.reshape(G, G, G).transpose(2, 1, 0)
        grid_texture = np.clip(grid_texture, finfo.min, finfo.max)
        grid_thres = -math.log(1 - min_alpha) / render_step
        grid_mask = grid_texture > (0.1 / (2 ** mip))
        # print(grid_mask.shape)
        grid_rle = grid_to_rle(grid_mask)
        grid_rles.append(grid_rle)

    log2_max_freq = enc_conf['log2_max_freq']
    log2_min_freq = enc_conf['log2_min_freq']

    freqs = 2 ** (np.arange(F).astype(np.float32) / (F - 1.0) * (log2_max_freq - log2_min_freq) + log2_min_freq)

    data_to_write = {
            'freqs': freqs.tolist(),
            'n_density_layers': [], # output layer sizes
            'n_color_layers': [], # output layer sizes
            'qff_type': qff_type,
            'n_freqs': F,
            'n_feats': C,
            'n_quants': Q,
            'rank': R,
            'grid_res': G,
            'grid_mip': mips,
            'grid_thres': grid_thres,
            'grid_rles': grid_rles,
            'poses': poses,
            'render_step': render_step,
            'up': snapshot['up_dir'],
            }

    buffer_sizes = {
            1: F*2*3*C*Q*R,
            2: F*2*3*C*Q*Q*R,
            3: F*2*C*Q*Q*Q,
            }
    qff_buffer_size = buffer_sizes[qff_type]
    qff_raw_buffers = snapshot['params_binary'][-qff_buffer_size*2:]
    if qff_type == 1:
        # Fx2x3xCxQxR => 3xFx2xQxRxC
        qff_buffers = np.frombuffer(qff_raw_buffers, np.float16).reshape(F, 2, 3, C, Q, R).transpose(2, 0, 1, 4, 5, 3)
    elif qff_type == 2:
        # Fx2x3xCxQxR => 3xFx2xQxQxRxC
        qff_buffers = np.frombuffer(qff_raw_buffers, np.float16).reshape(F, 2, 3, C, Q, Q, R).transpose(2, 0, 1, 4, 5, 6, 3)
    elif qff_type == 3:
        # Fx2xCxQxQxQ => Fx2xQxQxQxC
        qff_buffers = np.frombuffer(qff_raw_buffers, np.float16).reshape(F, 2, Q, Q, Q, C)
    else:
        raise NotImplementedError


    data_to_write['qff_buffer'] = qff_buffers.tobytes()

    # write MLP to files
    mlp_raw_buffers = snapshot['params_binary'][:-qff_buffer_size*2]
    mlp_buffers = np.frombuffer(mlp_raw_buffers, np.float16)

    # for now, all QFF buffer types have the same MLP structure
    # density = (Fx2xC) x 16 (density + features)
    # color = 16 + 16 (spherical harmonics) + n_extra_learnable_dims (for image-based lighting)

    # density network
    offset = 0
    dnet_i = dnet_conf['n_hidden_layers']
    dnet_w = dnet_conf['n_neurons']
    dnet_in = F*2*C
    dnet_out = 16

    # converts MLP buffer data to chunk of 4x4 matrices.
    def js_mlp_buffer(mlp_buffer):
        input_size, output_size = mlp_buffer.shape
        return mlp_buffer.reshape(input_size// 4, 4, output_size// 4, 4).transpose(0, 2, 3, 1).reshape(-1, 16).astype(np.float32).tobytes()

    for i in range(dnet_i + 1):
        if i == (dnet_i):
            dnet_w = dnet_out
        mlp_buffer = mlp_buffers[offset: offset + (dnet_in * dnet_w)]
        # in x out => in/4 x 4 x out/4 x 4 => in/4 x out/4 x (4 x 4) =>
        mlp_buffer = mlp_buffer.reshape(dnet_w, dnet_in).T
        data_to_write[f'qff_density_layer_{i}'] = js_mlp_buffer(mlp_buffer)
        data_to_write[f'n_density_layers'].append(dnet_w)

        offset += dnet_in * dnet_w
        dnet_in = dnet_w

    # color network
    cnet_i = cnet_conf['n_hidden_layers']
    cnet_w = cnet_conf['n_neurons']
    n_extra_learnable_dims = snapshot['nerf']['dataset']['n_extra_learnable_dims']
    cnet_in = dnet_out + dir_out + n_extra_learnable_dims
    cnet_out = 16

    #
    if n_extra_learnable_dims > 0:
        extra_opts = []
        for extra_dims_opt in ckpt['snapshot']['nerf']['extra_dims_opt']:
            extra_opts.append(extra_dims_opt['variable'])
        # Ex1
        mean_extra_opts = np.stack(extra_opts).mean(0).reshape(-1, 1)

    for i in range(cnet_i + 1):
        if i == cnet_i:
            cnet_w = cnet_out

        mlp_buffer = mlp_buffers[offset: offset + (cnet_in * cnet_w)]
        mlp_buffer = mlp_buffer.reshape(cnet_w, cnet_in).T


        # handle extra dimensions (image-based lighting)
        crop_in = None
        crop_out = None
        if i == 0:
            crop_in = dnet_out + dir_out

            # compute 'bias' for rgb. By default, it should be 0, but if we have extra learnable dims, we need to compute the mean
            if n_extra_learnable_dims > 0:
                extra_buffer = mlp_buffer[dnet_out + dir_out :dnet_out + dir_out + n_extra_learnable_dims, :crop_out]
                mean_extra_bias = (extra_buffer * mean_extra_opts).sum(0)
                data_to_write[f'image_bias'].append(mean_extra_bias.reshape(-1).astype(np.float32).tobytes())

        if i == cnet_i:
            crop_out = 4

        mlp_buffer = mlp_buffer[:crop_in, :crop_out]

        data_to_write[f'qff_color_layer_{i}'] = js_mlp_buffer(mlp_buffer)

        if i == cnet_i:
            cnet_w = 4
        data_to_write[f'n_color_layers'].append(cnet_w)

        offset += cnet_in * cnet_w
        cnet_in = cnet_w

    return data_to_write

def bake(ingp_file, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    ingp_file = Path(ingp_file)
    output_data = parse_ingp_file(ingp_file)

    with open(output_file, 'wb') as f:
        f.write(cbor2.dumps(output_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts INGP file into QFF file")
    parser.add_argument("--ingp_file", required=True, help="path to the INGP file")
    parser.add_argument("--output_file", required=True, help="path to the output QFF file")
    args = parser.parse_args()
    bake(args.ingp_file, args.output_file)
