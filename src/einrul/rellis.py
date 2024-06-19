from glob import glob 
import os
import numpy as np
import struct
import open3d
import argparse
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation as R
from natsort import natsorted

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    return pcd


def main(args):
    rellis_runs = glob(os.path.join(args.rellis, "*"))
    
    pcd_list = []

    for n, run in enumerate(rellis_runs):
        pose_list = []
        output_dir = os.path.join(args.output, str(n)
                                  )

        # load pose.txt
        pose_txt = os.path.join(run, 'poses.txt')
        calib_txt = os.path.join(run, 'calib.txt')
        
        calib = {}
        for line in open(calib_txt, 'r').readlines():
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose
        
        Tr = calib['Tr']
        Tr_inv = np.linalg.inv(Tr)
        
        for line in open(pose_txt, 'r').readlines():
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            
            transform_matrix = np.matmul(Tr_inv, np.matmul(pose, Tr))
            rotation_matrix = transform_matrix[:3, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat()
            translation = transform_matrix[:3, 3]

            q_dict = {'x': quaternion[0], 'y': quaternion[1], 'z': quaternion[2], 'w': quaternion[3]}
            t_dict = {'x': translation[0], 'y': translation[1], 'z': translation[2]}

            pose_list.append({'rotation': q_dict, 'position': t_dict})
            # pose_list.append({"rotation": quaternion, "position": translation})
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'poses.pkl'), 'wb') as f:
            pickle.dump(pose_list, f)

        # load bins and transform to pcd
        bins = natsorted(glob(os.path.join(run, 'os1_cloud_node_kitti_bin/*.bin')))
        output_pcd_dir = os.path.join(output_dir, 'pointcloud')
        os.makedirs(output_pcd_dir, exist_ok=True)

        for i, bin in tqdm(enumerate(bins), total=len(bins)):
            pcd = convert_kitti_bin_to_pcd(bin)
            
            pcd_path = os.path.join(output_pcd_dir, f'{i}.pcd')
            open3d.io.write_point_cloud(pcd_path, pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rellis"
                        , type=str, default="/data/dataset/rellis-3d/Rellis-3D")
    parser.add_argument("--output", type=str, default="/data/dataset/rellis-3d/EINRUL")

    args = parser.parse_args()

    main(args)
