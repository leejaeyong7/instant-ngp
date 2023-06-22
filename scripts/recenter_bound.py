import json
import torch

def find_nearest(poses):
    ray_o = poses[:, :3, 3]
    ray_d = poses[:, :3, 2]
    m = torch.zeros((3, 3))
    b = torch.zeros(3)
    for o, d in zip(ray_o, ray_d):
        d2 = (d ** 2).sum() 
        da = (o * d).sum()
        for ii in range(3):
            m[ii] += d[ii] * d
            m[ii, ii] -= d2
            b[ii] += d[ii] * da - o[ii] * d2
    p = torch.linalg.solve(m, b)
    rmax = (p - ray_o).norm(p=2, dim=-1).max()
    s = (2 * rmax * 1.1) / 3
    return p, s

def main(args):
    with open(args.json_path, 'r') as f:
        transforms = json.load(f)
    frames = transforms['frames']
    poses = []
    for frame in frames:
        pose = torch.Tensor(frame['transform_matrix']).view(4, 4)
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        poses.append(pose)

    all_poses = torch.stack(poses)

    center, scale = find_nearest(all_poses)
    offset = torch.zeros(3)
    offset[0] = 3
    offset[1] = 0
    offset[2] = 0

    for frame in frames:
        pose = torch.Tensor(frame['transform_matrix']).view(4, 4)
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        pose[:3, 3] -= center
        pose[:3, 3:] /= scale
        pose[:3, 3] += offset
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        frame['transform_matrix'] = pose.numpy().tolist()
    
    # go through each pose and update 
    with open(args.out_json_path, 'w') as f:
        json.dump(transforms, f)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--json_path', '-j', type=str)
    parser.add_argument('--out_json_path', '-o', type=str)
    args = parser.parse_args()
    main(args)

