import torch.nn.functional as NF
import torch
import cv2
import numpy as np
import json
from tqdm import tqdm
import argparse
import open3d as o3d
from pathlib import Path

def main(args):
    path = Path(args.path)
    with open(path / 'transforms.json', 'r') as f:
        trans = json.load(f)

    # setup intrinsics
    fx = trans['fl_x']
    fy = trans['fl_x']
    cx = trans['cx']
    cy = trans['cy']
    W = trans['w']
    H = trans['h']
    K = torch.tensor([
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
    ]).reshape(3, 3)


    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=1.0 / 1024.0,
        sdf_trunc=0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


    # read frames 
    scale = trans['integer_depth_scale']
    for frame in tqdm(trans['frames']):
        pose = np.array(frame['transform_matrix'])
        pose[:, 1:3] *= -1
        extrinsic = np.linalg.inv(pose)
        image_path= path / frame['file_path']
        depth_path= path / frame['depth_path']

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) * scale
        colors = cv2.imread(str(image_path))[..., [2, 1, 0]]
        rgbi = o3d.geometry.Image(np.ascontiguousarray(colors))
        depthi = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbi, depthi, depth_scale=1, convert_rgb_to_intensity=False)
        volume.integrate(rgbd_image, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizes transforms with depth")
    parser.add_argument("--path", default="", help="Path containing transforms.json")
    args = parser.parse_args()

    main(args)