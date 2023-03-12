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
		print(depth_img.size, depth_img.shape)
	return depth_img

def rotate_img(img_path, degree=90):
	img = Image.open(img_path)
	img = img.rotate(degree, expand=1)
	img.save(img_path, quality=100, subsampling=0)

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
	print("computing center of attention...")
	frames = raw_transforms['frames']
	for frame in frames:
		frame['transform_matrix'] = np.array(frame['transform_matrix'])

	rays_o = []
	rays_d = []
	for f in tqdm(frames):
		mf = f["transform_matrix"][0:3,:]
		rays_o.append(mf[:3,3:])
		rays_d.append(mf[:3,2:3])
	rays_o = np.asarray(rays_o)[..., 0]
	rays_d = np.asarray(rays_d)[..., 0]

	radius=3.141592
	buffer=1.1

	m = np.zeros((3, 3))
	b = np.zeros(3)
	for o, d in zip(rays_o, rays_d):
		d2 = (d ** 2).sum() 
		da = (o * d).sum()
		for ii in range(3):
			m[ii] += d[ii] * d
			m[ii, ii] -= d2
			b[ii] += d[ii] * da - o[ii] * d2
	p = np.linalg.solve(m, b)
	rmax = np.linalg.norm(p - rays_o, ord=2, axis=-1).max()
	s = (2 * rmax * buffer) / radius
	for frame in frames:
		frame['transform_matrix'] = frame['transform_matrix'].tolist()
	return p, s


def normalize_transforms(transforms, translation, scale):
	normalized_transforms = copy.deepcopy(transforms)
	ids = normalized_transforms['integer_depth_scale']
	normalized_transforms['integer_depth_scale'] = ids / scale
	for f in normalized_transforms["frames"]:
		f["transform_matrix"] = np.asarray(f["transform_matrix"])
		f["transform_matrix"][0:3,3] -= translation
		f["transform_matrix"][0:3,3] *= scale
		f["transform_matrix"] = f["transform_matrix"].tolist()
	return normalized_transforms

def parse_args():
	parser = argparse.ArgumentParser(description="convert a Record3D capture to nerf format transforms.json")
	parser.add_argument("--scene", default="", help="path to the Record3D capture")
	parser.add_argument("--output", default="outputs", help="path to output")
	parser.add_argument("--rotate", action="store_true", help="rotate the dataset")
	parser.add_argument("--subsample", default=1, type=int, help="step size of subsampling")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	dataset_dir = Path(args.scene)
	output_dir = Path(args.output)
	(output_dir / 'images').mkdir(exist_ok=True, parents=True)
	(output_dir / 'depths').mkdir(exist_ok=True, parents=True)

	with open(dataset_dir / 'metadata') as f:
		metadata = json.load(f)

	frames = []
	n_images = len(list((dataset_dir / 'rgbd').glob('*.jpg')))
	poses = np.array(metadata['poses'])
	for idx in tqdm(range(n_images)):
		# Link the image.
		img_name = f'{idx}.jpg'
		img_path = dataset_dir / 'rgbd' / img_name
		depth_path = dataset_dir / 'rgbd' / f'{idx}.depth'
		conf_path = dataset_dir / 'rgbd' / f'{idx}.conf'

		out_image_path = output_dir / 'images' / f'{idx}.png'
		# copy image
		image = Image.open(img_path)
		image.save(out_image_path)


		# Rotate the image.
		if args.rotate:
			# TODO: parallelize this step with joblib.
			rotate_img(out_image_path)

		# Extract c2w.
		""" Each `pose` is a 7-element tuple which contains quaternion + world position.
			[qx, qy, qz, qw, tx, ty, tz]
		"""
		pose = poses[idx]
		q = Quaternion(x=pose[0], y=pose[1], z=pose[2], w=pose[3])
		c2w = np.eye(4)
		c2w[:3, :3] = q.rotation_matrix
		c2w[:3, -1] = [pose[4], pose[5], pose[6]]
		if args.rotate:
			c2w = rotate_camera(c2w)
			c2w = swap_axes(c2w)

		frames.append(
			{
				"file_path": f"./images/{idx}.png",
				"depth_path": f"./depths/{idx}.depth.png",
				"transform_matrix": c2w.tolist(),
			}
		)

	if not args.rotate:
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
	for idx in tqdm(range(n_images)):
		# dh x dw float32 
		depth_path = dataset_dir / 'rgbd' / f'{idx}.depth'
		depth = load_depth(depth_path)
		max_depth = max(depth.max(), max_depth)

	for idx in tqdm(range(n_images)):
		depth_path = dataset_dir / 'rgbd' / f'{idx}.depth'
		conf_path = dataset_dir / 'rgbd' / f'{idx}.conf'
		depth = load_depth(depth_path)
		conf = load_depth(conf_path, dtype=np.uint8)
		depth = (depth*65535/float(max_depth)).astype(np.uint16)
		depth[conf != 2] = 0
		depth = cv2.resize(depth, dsize=(image.width, image.height), interpolation=cv2.INTER_NEAREST)
		if args.rotate:
			depth = np.rot90(depth)
		cv2.imwrite(str(output_dir / 'depths' / f'{idx}.depth.png'), depth)


	transforms = {}
	transforms['fl_x'] = fx
	transforms['fl_y'] = fy
	transforms['cx'] = cx
	transforms['cy'] = cy
	transforms['w'] = w
	transforms['h'] = h
	transforms['aabb_scale'] = 4
	transforms['scale'] = 1.0
	transforms['camera_angle_x'] = 2 * np.arctan(transforms['w'] / (2 * transforms['fl_x']))
	transforms['camera_angle_y'] = 2 * np.arctan(transforms['h'] / (2 * transforms['fl_y']))
	transforms['frames'] = frames
	transforms["integer_depth_scale"] = float(max_depth) /65535.0


	# Normalize the poses.
	transforms['frames'] = transforms['frames'][::args.subsample]
	find_rotation_and_apply(transforms)
	translation, scale = find_transforms_center_and_scale(transforms)
	normalized_transforms = normalize_transforms(transforms, translation, scale)

	output_path = output_dir / 'transforms.json'
	with open(output_path, "w") as outfile:
		json.dump(normalized_transforms, outfile, indent=2)
