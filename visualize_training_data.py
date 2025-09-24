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


def load_training_data(train_dir, labels_file, arm_labels_file, num_samples=5):
    """
    Load random samples from training data including arm coordinates

    Args:
        train_dir: Path to directory with .npz point cloud files
        labels_file: Path to train_labels.h5 file
        arm_labels_file: Path to arm_labels.h5 file
        num_samples: Number of random samples to load

    Returns:
        List of (point_cloud, joints, arm_data, identifier) tuples
    """
    samples = []

    # Load main labels
    with h5py.File(labels_file, 'r') as f:
        identifiers = f['id'][:]
        joint_coords = f['real_world_coordinates'][:]
        is_valid = f['is_valid'][:]

    # Load arm labels
    arm_data = {}
    if os.path.exists(arm_labels_file):
        with h5py.File(arm_labels_file, 'r') as f:
            arm_identifiers = f['id'][:]
            if 'left_arm_coords' in f:
                arm_data['left_arm_coords'] = f['left_arm_coords'][:]
            if 'right_arm_coords' in f:
                arm_data['right_arm_coords'] = f['right_arm_coords'][:]
        print(f"Loaded arm data: {list(arm_data.keys())}")
    else:
        print("Warning: arm_labels.h5 not found, arm visualization will be skipped")

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

        # Get corresponding arm data for this frame
        frame_arm_data = {}
        for arm_key, arm_coords in arm_data.items():
            if idx < len(arm_coords):
                frame_arm_data[arm_key] = arm_coords[idx]

        # Load point cloud file
        # Extract frame number from identifier (format: "XX_YYYYY")
        frame_num = int(identifier.split('_')[-1])
        pc_file = os.path.join(train_dir, f"{frame_num}.npz")

        if os.path.exists(pc_file):
            pc_data = np.load(pc_file)
            point_cloud = pc_data['arr_0']
            samples.append((point_cloud, joints, frame_arm_data, identifier))
            arm_info = f", arms: {list(frame_arm_data.keys())}" if frame_arm_data else ""
            print(f"Loaded sample {identifier}: {point_cloud.shape[0]} points, joints shape {joints.shape}{arm_info}")
        else:
            print(f"Warning: Point cloud file not found for {identifier}")

    return samples


def visualize_frame(ax, point_cloud, joints, arm_data, identifier, downsample_points=2000):
    """
    Visualize a single frame with point cloud, skeleton, and arm coordinates

    Args:
        ax: Matplotlib 3D axis
        point_cloud: Nx3 array of point coordinates
        joints: 15x3 array of joint coordinates
        arm_data: Dictionary with arm coordinate data
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
              c='red', s=100, alpha=1.0, label='Body Joints')

    # Draw skeleton connections
    for connection in skeleton_joints.joint_connections:
        start_idx, end_idx, color = connection
        start_pos = joints[start_idx]
        end_pos = joints[end_idx]
        ax.plot3D([start_pos[0], end_pos[0]],
                 [start_pos[1], end_pos[1]],
                 [start_pos[2], end_pos[2]],
                 color=color, linewidth=2)

    # Plot arm coordinates if available
    if 'left_arm_coords' in arm_data:
        left_arm = arm_data['left_arm_coords']  # Shape: (2, 3)
        ax.scatter(left_arm[:, 0], left_arm[:, 1], left_arm[:, 2],
                  c='green', s=150, alpha=1.0, marker='^', label='Left Arm')
        # Connect left arm points
        if len(left_arm) == 2:
            ax.plot3D([left_arm[0, 0], left_arm[1, 0]],
                     [left_arm[0, 1], left_arm[1, 1]],
                     [left_arm[0, 2], left_arm[1, 2]],
                     color='green', linewidth=3, alpha=0.8)

    if 'right_arm_coords' in arm_data:
        right_arm = arm_data['right_arm_coords']  # Shape: (2, 3)
        ax.scatter(right_arm[:, 0], right_arm[:, 1], right_arm[:, 2],
                  c='orange', s=150, alpha=1.0, marker='v', label='Right Arm')
        # Connect right arm points
        if len(right_arm) == 2:
            ax.plot3D([right_arm[0, 0], right_arm[1, 0]],
                     [right_arm[0, 1], right_arm[1, 1]],
                     [right_arm[0, 2], right_arm[1, 2]],
                     color='orange', linewidth=3, alpha=0.8)

    # Add joint labels
    for idx, joint_name in skeleton_joints.joint_indices.items():
        pos = joints[idx]
        ax.text(pos[0], pos[1], pos[2], joint_name, fontsize=6)

    # Add arm labels if available
    if 'left_arm_coords' in arm_data:
        for i, pos in enumerate(arm_data['left_arm_coords']):
            ax.text(pos[0], pos[1], pos[2], f'L{i+1}', fontsize=8, color='green')

    if 'right_arm_coords' in arm_data:
        for i, pos in enumerate(arm_data['right_arm_coords']):
            ax.text(pos[0], pos[1], pos[2], f'R{i+1}', fontsize=8, color='orange')

    # Set axis properties
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
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
        samples: List of (point_cloud, joints, arm_data, identifier) tuples
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

    for i, (point_cloud, joints, arm_data, identifier) in enumerate(samples[:n_samples]):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        visualize_frame(ax, point_cloud, joints, arm_data, identifier)

    plt.suptitle('SPiKE Training Data Samples (with Arm GT)', fontsize=16)
    plt.tight_layout()


def print_data_statistics(samples):
    """
    Print statistics about the loaded samples

    Args:
        samples: List of (point_cloud, joints, arm_data, identifier) tuples
    """
    print("\n" + "="*50)
    print("Training Data Statistics (with Arm GT)")
    print("="*50)

    for i, (point_cloud, joints, arm_data, identifier) in enumerate(samples):
        print(f"\nSample {i+1} (ID: {identifier}):")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Points range: X[{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}], "
              f"Y[{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}], "
              f"Z[{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")
        print(f"  Joints shape: {joints.shape}")
        print(f"  Joints range: X[{joints[:, 0].min():.2f}, {joints[:, 0].max():.2f}], "
              f"Y[{joints[:, 1].min():.2f}, {joints[:, 1].max():.2f}], "
              f"Z[{joints[:, 2].min():.2f}, {joints[:, 2].max():.2f}]")

        # Print arm data statistics
        if arm_data:
            print(f"  Arm data available: {list(arm_data.keys())}")
            for arm_key, arm_coords in arm_data.items():
                if len(arm_coords) > 0:
                    print(f"  {arm_key} shape: {arm_coords.shape}")
                    print(f"  {arm_key} range: X[{arm_coords[:, 0].min():.2f}, {arm_coords[:, 0].max():.2f}], "
                          f"Y[{arm_coords[:, 1].min():.2f}, {arm_coords[:, 1].max():.2f}], "
                          f"Z[{arm_coords[:, 2].min():.2f}, {arm_coords[:, 2].max():.2f}]")
        else:
            print(f"  No arm data available")

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

        # Check if arm data is within point cloud bounds
        if arm_data:
            arms_in_bounds = True
            for arm_key, arm_coords in arm_data.items():
                if len(arm_coords) > 0:
                    for j in range(3):
                        if arm_coords[:, j].min() < pc_bounds[j][0] or arm_coords[:, j].max() > pc_bounds[j][1]:
                            arms_in_bounds = False
                            break
            print(f"  Arms within point cloud bounds: {arms_in_bounds}")


def main():
    # Paths to training data
    train_dir = "/home/oliver/Documents/data/Mocap/train"
    labels_file = "/home/oliver/Documents/data/Mocap/train_labels.h5"
    arm_labels_file = "/home/oliver/Documents/data/Mocap/arm_labels.h5"

    # Number of random samples to visualize
    num_samples = 6

    print(f"Loading {num_samples} random samples from training data...")
    print(f"Train directory: {train_dir}")
    print(f"Labels file: {labels_file}")
    print(f"Arm labels file: {arm_labels_file}")

    # Load random samples
    samples = load_training_data(train_dir, labels_file, arm_labels_file, num_samples)

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