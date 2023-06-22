#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path

import numpy as np
import json
import copy
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm
from PIL import Image
import liblzfse  # https://pypi.org/project/pyliblzfse/

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def load_depth(filepath, dtype=np.float32):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=dtype)
    num_points = depth_img.size
    if num_points == (640 * 480):
        depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    elif num_points == (256 * 192):
        depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
    else:
        raise NotImplementedError()
    return depth_img

def rotate_camera(c2w, degree=90):
    rad = np.deg2rad(degree)
    R = Quaternion(axis=[0, 0, -1], angle=rad)
    T = R.transformation_matrix
    return c2w @ T

def swap_axes(c2w):
    rad = np.pi / 2
    R = Quaternion(axis=[1, 0, 0], angle=rad)
    T = R.transformation_matrix
    return T @ c2w

# Automatic rescale & offset the poses.
def find_rotation_and_apply(raw_transforms):
    frames = raw_transforms['frames']
    for frame in frames:
        frame['transform_matrix'] = np.array(frame['transform_matrix'])

    up = np.zeros(3)
    for f in tqdm(frames):
        up += f["transform_matrix"][0:3,1]

    up = up / np.linalg.norm(up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for frame in frames:
        frame['transform_matrix'] = (R @ frame['transform_matrix']).tolist()

def find_transforms_center_and_scale(raw_transforms):
    fx = raw_transforms['fl_x']
    fy = raw_transforms['fl_y']
    cx = raw_transforms['cx']
    cy = raw_transforms['cy']
    w = raw_transforms['w']
    h = raw_transforms['h']
    yz_neg = np.array([
        1, 0, 0,
        0, -1, 0,
        0, 0, -1,
    ]).reshape(3, 3)
    K = np.array([
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
    ]).reshape(3, 3) @ yz_neg

    image_corners = np.array([
        0.5, 0.5, 1,
        w-0.5, 0.5, 1,
        0.5, h-0.5, 1,
        w-0.5, h-0.5, 1,
    ]).reshape(-1, 3)

    corner_points = image_corners @ np.linalg.inv(K).T


    frames = raw_transforms['frames']
    world_points = []
    for frame in tqdm(frames, desc="Computing Optimal AABB"):
        pose = np.array(frame['transform_matrix'])
        far = frame['far']

        proj_corner_points = corner_points * far
        proj_corner_hpoints = np.concatenate([
            proj_corner_points,
            np.ones_like(proj_corner_points[:, :1])
        ], 1)
        world_point = proj_corner_hpoints @ pose.T
        world_points.append(world_point[:, :3])
    # compute aabb 
    world_points = np.concatenate(world_points)
    mins = world_points.min(0)
    maxs = world_points.max(0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins).max()
    return center, 1 / scale



def normalize_transforms(transforms, translation, scale):
    normalized_transforms = copy.deepcopy(transforms)
    ids = normalized_transforms['integer_depth_scale']
    normalized_transforms['integer_depth_scale'] = ids * scale
    for f in normalized_transforms["frames"]:
        f["transform_matrix"] = np.asarray(f["transform_matrix"])
        f["transform_matrix"][0:3,3] -= translation
        f["transform_matrix"][0:3,3] *= scale
        f["transform_matrix"] = f["transform_matrix"].tolist()
    return normalized_transforms

def main(args):
    dataset_dir = Path(args.scene)
    output_dir = Path(args.output)
    r3d_to_transforms(dataset_dir, output_dir, args.rotate, args.subsample)

def r3d_to_transforms(r3d_path, output_path, rotate=False, subsample=1):
    (output_path / 'images').mkdir(exist_ok=True, parents=True)
    (output_path / 'depths').mkdir(exist_ok=True, parents=True)

    with open(r3d_path / 'metadata') as f:
        metadata = json.load(f)

    poses = np.array(metadata['poses'])
    n_images = len(poses)

    if not rotate:
        h = metadata['h']
        w = metadata['w']
        K = np.array(metadata['K']).reshape([3, 3]).T
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    else:
        h = metadata['w']
        w = metadata['h']
        K = np.array(metadata['K']).reshape([3, 3]).T
        fx = K[1, 1]
        fy = K[0, 0]
        cx = K[1, 2]
        cy = h - K[0, 2]


    max_depth = 0
    for idx in tqdm(list(range(0, n_images, subsample)), desc="Computing Max Depth"):
        # dh x dw float32 
        depth_path = r3d_path / 'rgbd' / f'{idx}.depth'
        depth = load_depth(depth_path)
        max_depth = max(depth.max(), max_depth)

    frames = []
    for idx in tqdm(list(range(0, n_images, subsample)), desc="Processing Images"):
        # Link the image.
        img_path = r3d_path / 'rgbd' / f'{idx}.jpg'
        depth_path = r3d_path / 'rgbd' / f'{idx}.depth'
        conf_path = r3d_path / 'rgbd' / f'{idx}.conf'

        out_image_path = output_path / 'images' / f'{idx}.png'
        out_depth_path = output_path / 'depths' / f'{idx}.depth.png'

        # copy image
        image = Image.open(img_path)
        if rotate:
            image = image.rotate(90, expand=1)
        image.save(out_image_path)

        # extract depth
        depth = load_depth(depth_path)
        conf = load_depth(conf_path, dtype=np.uint8)

        # save depth
        min_d = depth[conf == 2].min()
        max_d = depth[conf == 2].max()

        # quantize depth into 16 bits
        depth = (depth*65535 /float(max_depth)).astype(np.uint16)
        depth[conf != 2] = 0

        if rotate:
            depth = np.rot90(depth)
        depth = cv2.resize(depth, dsize=(image.width, image.height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_depth_path), depth)

        # Extract c2w.
        """ Each `pose` is a 7-element tuple which contains quaternion + world position.  [qx, qy, qz, qw, tx, ty, tz] """
        pose = poses[idx]
        q = Quaternion(x=pose[0], y=pose[1], z=pose[2], w=pose[3])
        c2w = np.eye(4)
        c2w[:3, :3] = q.rotation_matrix
        c2w[:3, -1] = [pose[4], pose[5], pose[6]]
        if rotate:
            c2w = rotate_camera(c2w)
            c2w = swap_axes(c2w)

        frame = {
            "file_path": f"./images/{idx}.png",
            "depth_path": f"./depths/{idx}.depth.png",
            "transform_matrix": c2w.tolist(),
            'near': min_d * 0.9,
            'far': max_d * 1.4
        }
        frames.append(frame)

    # write out transforms
    transforms = {}
    transforms['fl_x'] = fx
    transforms['fl_y'] = fy
    transforms['cx'] = cx
    transforms['cy'] = cy
    transforms['w'] = w
    transforms['h'] = h
    transforms['aabb_scale'] = 1
    transforms['scale'] = 1.0
    transforms['camera_angle_x'] = 2 * np.arctan(transforms['w'] / (2 * transforms['fl_x']))
    transforms['camera_angle_y'] = 2 * np.arctan(transforms['h'] / (2 * transforms['fl_y']))
    transforms['frames'] = frames
    transforms["integer_depth_scale"] = float(max_depth) /65535.0


    # Normalize the poses.
    find_rotation_and_apply(transforms)

    translation, scale = find_transforms_center_and_scale(transforms)
    normalized_transforms = normalize_transforms(transforms, translation, scale)

    output_path = output_path / 'transforms.json'
    with open(output_path, "w") as outfile:
        json.dump(normalized_transforms, outfile, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert a Record3D capture to nerf format transforms.json")
    parser.add_argument("--scene", default="", help="path to the Record3D capture")
    parser.add_argument("--output", default="outputs", help="path to output")
    parser.add_argument("--rotate", action="store_true", help="rotate the dataset")
    parser.add_argument("--subsample", default=1, type=int, help="step size of subsampling")
    args = parser.parse_args()
    main(args)
