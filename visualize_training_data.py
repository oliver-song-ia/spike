"""
3D visualization script for converted SPiKE training data
Randomly visualizes several frames from the training dataset
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Import skeleton joint definitions
import sys
sys.path.append('/home/oliver/Documents/SPiKE')
from const import skeleton_joints


def load_training_data(train_dir, labels_file, num_samples=5):
    """
    Load random samples from training data

    Args:
        train_dir: Path to directory with .npz point cloud files
        labels_file: Path to train_labels.h5 file
        num_samples: Number of random samples to load

    Returns:
        List of (point_cloud, joints, identifier) tuples
    """
    samples = []

    # Load labels
    with h5py.File(labels_file, 'r') as f:
        identifiers = f['id'][:]
        joint_coords = f['real_world_coordinates'][:]
        is_valid = f['is_valid'][:]

    # Get indices of valid frames
    valid_indices = [i for i, v in enumerate(is_valid) if v]

    # Randomly sample indices
    if len(valid_indices) > num_samples:
        selected_indices = random.sample(valid_indices, num_samples)
    else:
        selected_indices = valid_indices[:num_samples]

    # Load corresponding point clouds
    for idx in selected_indices:
        identifier = identifiers[idx].decode('utf-8')
        joints = joint_coords[idx]

        # Load point cloud file
        # Extract frame number from identifier (format: "XX_YYYYY")
        frame_num = int(identifier.split('_')[-1])
        pc_file = os.path.join(train_dir, f"{frame_num}.npz")

        if os.path.exists(pc_file):
            pc_data = np.load(pc_file)
            point_cloud = pc_data['arr_0']
            samples.append((point_cloud, joints, identifier))
            print(f"Loaded sample {identifier}: {point_cloud.shape[0]} points, joints shape {joints.shape}")
        else:
            print(f"Warning: Point cloud file not found for {identifier}")

    return samples


def visualize_frame(ax, point_cloud, joints, identifier, downsample_points=2000):
    """
    Visualize a single frame with point cloud and skeleton

    Args:
        ax: Matplotlib 3D axis
        point_cloud: Nx3 array of point coordinates
        joints: 15x3 array of joint coordinates
        identifier: Frame identifier string
        downsample_points: Number of points to display for performance
    """
    ax.clear()

    # Downsample point cloud for visualization
    if point_cloud.shape[0] > downsample_points:
        indices = np.random.choice(point_cloud.shape[0], downsample_points, replace=False)
        pc_vis = point_cloud[indices]
    else:
        pc_vis = point_cloud

    # Plot point cloud
    ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
              c='blue', s=1, alpha=0.3, label='Point Cloud')

    # Plot skeleton joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
              c='red', s=100, alpha=1.0, label='Joints')

    # Draw skeleton connections
    for connection in skeleton_joints.joint_connections:
        start_idx, end_idx, color = connection
        start_pos = joints[start_idx]
        end_pos = joints[end_idx]
        ax.plot3D([start_pos[0], end_pos[0]],
                 [start_pos[1], end_pos[1]],
                 [start_pos[2], end_pos[2]],
                 color=color, linewidth=2)

    # Add joint labels
    for idx, joint_name in skeleton_joints.joint_indices.items():
        pos = joints[idx]
        ax.text(pos[0], pos[1], pos[2], joint_name, fontsize=8)

    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame: {identifier}')

    # Set equal aspect ratio
    max_range = np.array([
        pc_vis[:, 0].max() - pc_vis[:, 0].min(),
        pc_vis[:, 1].max() - pc_vis[:, 1].min(),
        pc_vis[:, 2].max() - pc_vis[:, 2].min()
    ]).max() / 2.0

    mid_x = (pc_vis[:, 0].max() + pc_vis[:, 0].min()) * 0.5
    mid_y = (pc_vis[:, 1].max() + pc_vis[:, 1].min()) * 0.5
    mid_z = (pc_vis[:, 2].max() + pc_vis[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc='upper right')


def create_multi_frame_visualization(samples):
    """
    Create a figure with multiple subplots for each sample

    Args:
        samples: List of (point_cloud, joints, identifier) tuples
    """
    n_samples = len(samples)

    # Create figure with subplots
    if n_samples <= 2:
        fig = plt.figure(figsize=(16, 8))
        rows, cols = 1, n_samples
    elif n_samples <= 4:
        fig = plt.figure(figsize=(16, 16))
        rows, cols = 2, 2
    elif n_samples <= 6:
        fig = plt.figure(figsize=(24, 16))
        rows, cols = 2, 3
    else:
        fig = plt.figure(figsize=(24, 24))
        rows, cols = 3, 3
        n_samples = min(n_samples, 9)  # Limit to 9 subplots

    for i, (point_cloud, joints, identifier) in enumerate(samples[:n_samples]):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        visualize_frame(ax, point_cloud, joints, identifier)

    plt.suptitle('SPiKE Training Data Samples', fontsize=16)
    plt.tight_layout()


def print_data_statistics(samples):
    """
    Print statistics about the loaded samples

    Args:
        samples: List of (point_cloud, joints, identifier) tuples
    """
    print("\n" + "="*50)
    print("Training Data Statistics")
    print("="*50)

    for i, (point_cloud, joints, identifier) in enumerate(samples):
        print(f"\nSample {i+1} (ID: {identifier}):")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Points range: X[{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}], "
              f"Y[{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}], "
              f"Z[{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")
        print(f"  Joints shape: {joints.shape}")
        print(f"  Joints range: X[{joints[:, 0].min():.2f}, {joints[:, 0].max():.2f}], "
              f"Y[{joints[:, 1].min():.2f}, {joints[:, 1].max():.2f}], "
              f"Z[{joints[:, 2].min():.2f}, {joints[:, 2].max():.2f}]")

        # Check if joints are within point cloud bounds
        pc_bounds = [
            [point_cloud[:, 0].min(), point_cloud[:, 0].max()],
            [point_cloud[:, 1].min(), point_cloud[:, 1].max()],
            [point_cloud[:, 2].min(), point_cloud[:, 2].max()]
        ]

        joints_in_bounds = True
        for j in range(3):
            if joints[:, j].min() < pc_bounds[j][0] or joints[:, j].max() > pc_bounds[j][1]:
                joints_in_bounds = False
                break

        print(f"  Joints within point cloud bounds: {joints_in_bounds}")


def main():
    # Paths to training data
    train_dir = "/home/oliver/Documents/data/Mocap/train"
    labels_file = "/home/oliver/Documents/data/Mocap/train_labels.h5"

    # Number of random samples to visualize
    num_samples = 6

    print(f"Loading {num_samples} random samples from training data...")
    print(f"Train directory: {train_dir}")
    print(f"Labels file: {labels_file}")

    # Load random samples
    samples = load_training_data(train_dir, labels_file, num_samples)

    if len(samples) == 0:
        print("No valid samples found!")
        return

    print(f"\nSuccessfully loaded {len(samples)} samples")

    # Print statistics
    print_data_statistics(samples)

    # Create visualization
    create_multi_frame_visualization(samples)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()