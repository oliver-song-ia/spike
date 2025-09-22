#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d

def visualize_point_cloud(npz_path):
    data = np.load(npz_path)
    # 保存时用 np.savez，只存了一个数组 => 取第一个
    arr = data[data.files[0]]

    if arr.size == 0:
        print(f"{npz_path} is empty!")
        return

    print(f"Loaded {arr.shape[0]} points from {npz_path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    pcd.paint_uniform_color([0.2, 0.7, 0.9])  # 淡蓝色
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed ITOP point cloud (.npz).")
    parser.add_argument("npz_file", type=str, help="Path to the .npz file to visualize")
    args = parser.parse_args()

    visualize_point_cloud(args.npz_file)
