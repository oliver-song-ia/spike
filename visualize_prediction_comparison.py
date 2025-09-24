"""
Visualization script to compare point cloud, predictions, and ground truth
- Shows point clouds with predicted and ground truth poses
- Focuses on upper body joints (0-8) since lower body predictions are invalid
- Uses world coordinate system data from inference_labels_world.h5
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


def load_prediction_data(train_dir, train_labels_file, arm_labels_file, inference_file, num_samples=5):
    """
    Load random samples from prediction results including point clouds

    Args:
        train_dir: Path to directory with .npz point cloud files
        train_labels_file: Path to train_labels.h5 file (original GT)
        arm_labels_file: Path to arm_labels.h5 file (arm coordinates)
        inference_file: Path to inference_labels_world.h5 file (predictions)
        num_samples: Number of random samples to load

    Returns:
        List of (point_cloud, gt_joints, pred_joints, arm_data, identifier) tuples
    """
    samples = []

    # Load original training labels (GT)
    with h5py.File(train_labels_file, 'r') as f:
        train_identifiers = f['id'][:]
        train_gt_coords = f['real_world_coordinates'][:]
        train_is_valid = f['is_valid'][:]

    # Create mapping from identifier to GT coordinates
    gt_mapping = {}
    for i, identifier in enumerate(train_identifiers):
        id_str = identifier.decode('utf-8')
        gt_mapping[id_str] = train_gt_coords[i]

    # Load arm labels
    arm_data_mapping = {}
    if os.path.exists(arm_labels_file):
        with h5py.File(arm_labels_file, 'r') as f:
            arm_identifiers = f['id'][:]
            arm_data = {}
            if 'left_arm_coords' in f:
                arm_data['left_arm_coords'] = f['left_arm_coords'][:]
            if 'right_arm_coords' in f:
                arm_data['right_arm_coords'] = f['right_arm_coords'][:]

        # Create mapping from identifier to arm coordinates
        for i, identifier in enumerate(arm_identifiers):
            id_str = identifier.decode('utf-8')
            frame_arm_data = {}
            for arm_key, arm_coords in arm_data.items():
                if i < len(arm_coords):
                    frame_arm_data[arm_key] = arm_coords[i]
            arm_data_mapping[id_str] = frame_arm_data
        print(f"Loaded arm data: {list(arm_data.keys())}")
    else:
        print("Warning: arm_labels.h5 not found, arm visualization will be skipped")

    # Load inference results (predictions)
    with h5py.File(inference_file, 'r') as f:
        identifiers = f['id'][:]
        pred_coords = f['predicted_coordinates'][:]
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
        pred_joints = pred_coords[idx]

        # Get original GT from training data
        if identifier in gt_mapping:
            gt_joints = gt_mapping[identifier]
        else:
            print(f"Warning: GT not found for {identifier}")
            continue

        # Get arm data for this frame
        frame_arm_data = arm_data_mapping.get(identifier, {})

        # Load point cloud file
        # Extract frame number from identifier (format: "XX_YYYYY")
        frame_num = int(identifier.split('_')[-1])
        pc_file = os.path.join(train_dir, f"{frame_num}.npz")

        if os.path.exists(pc_file):
            pc_data = np.load(pc_file)
            point_cloud = pc_data['arr_0']
            samples.append((point_cloud, gt_joints, pred_joints, frame_arm_data, identifier))
            arm_info = f", arms: {list(frame_arm_data.keys())}" if frame_arm_data else ""
            print(f"Loaded sample {identifier}: {point_cloud.shape[0]} points, "
                  f"GT shape {gt_joints.shape}, Pred shape {pred_joints.shape}{arm_info}")
        else:
            print(f"Warning: Point cloud file not found for {identifier}")

    return samples


def calculate_joint_errors(gt_joints, pred_joints):
    """Calculate per-joint errors"""
    errors = np.linalg.norm(pred_joints - gt_joints, axis=1)
    return errors


def visualize_comparison(ax, point_cloud, gt_joints, pred_joints, arm_data, identifier,
                        upper_body_only=True, downsample_points=2000):
    """
    Visualize point cloud with GT and predicted joints, including arm coordinates

    Args:
        ax: Matplotlib 3D axis
        point_cloud: Nx3 array of point coordinates
        gt_joints: 15x3 array of ground truth joint coordinates
        pred_joints: 15x3 array of predicted joint coordinates
        arm_data: Dictionary with arm coordinate data
        identifier: Frame identifier string
        upper_body_only: If True, only show upper body joints (0-8)
        downsample_points: Number of points to display for performance
    """
    ax.clear()

    # Define upper body joint indices (0-8)
    if upper_body_only:
        joint_indices = list(range(9))  # 0-8: upper body
        title_suffix = " (Upper Body Only)"
    else:
        joint_indices = list(range(15))  # 0-14: all joints
        title_suffix = " (All Joints)"

    # Downsample point cloud for visualization
    if point_cloud.shape[0] > downsample_points:
        indices = np.random.choice(point_cloud.shape[0], downsample_points, replace=False)
        pc_vis = point_cloud[indices]
    else:
        pc_vis = point_cloud

    # Plot point cloud
    ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
              c='lightblue', s=0.5, alpha=0.4, label='Point Cloud')

    # Plot ground truth joints
    gt_subset = gt_joints[joint_indices]
    ax.scatter(gt_subset[:, 0], gt_subset[:, 1], gt_subset[:, 2],
              c='green', s=100, alpha=0.9, marker='o', label='Ground Truth')

    # Plot predicted joints
    pred_subset = pred_joints[joint_indices]
    ax.scatter(pred_subset[:, 0], pred_subset[:, 1], pred_subset[:, 2],
              c='red', s=100, alpha=0.9, marker='^', label='Prediction')

    # Draw GT skeleton connections (green)
    for connection in skeleton_joints.joint_connections:
        start_idx, end_idx, color = connection
        if start_idx in joint_indices and end_idx in joint_indices:
            start_pos = gt_joints[start_idx]
            end_pos = gt_joints[end_idx]
            ax.plot3D([start_pos[0], end_pos[0]],
                     [start_pos[1], end_pos[1]],
                     [start_pos[2], end_pos[2]],
                     color='green', linewidth=2, alpha=0.7)

    # Draw predicted skeleton connections (red, dashed)
    for connection in skeleton_joints.joint_connections:
        start_idx, end_idx, color = connection
        if start_idx in joint_indices and end_idx in joint_indices:
            start_pos = pred_joints[start_idx]
            end_pos = pred_joints[end_idx]
            ax.plot3D([start_pos[0], end_pos[0]],
                     [start_pos[1], end_pos[1]],
                     [start_pos[2], end_pos[2]],
                     color='red', linewidth=2, alpha=0.7, linestyle='--')

    # Draw error lines between GT and predictions
    for i in joint_indices:
        gt_pos = gt_joints[i]
        pred_pos = pred_joints[i]
        ax.plot3D([gt_pos[0], pred_pos[0]],
                 [gt_pos[1], pred_pos[1]],
                 [gt_pos[2], pred_pos[2]],
                 color='orange', linewidth=1, alpha=0.6)

    # Plot arm coordinates if available
    if 'left_arm_coords' in arm_data:
        left_arm = arm_data['left_arm_coords']  # Shape: (2, 3)
        ax.scatter(left_arm[:, 0], left_arm[:, 1], left_arm[:, 2],
                  c='cyan', s=150, alpha=1.0, marker='^', label='Left Arm')
        # Connect left arm points
        if len(left_arm) == 2:
            ax.plot3D([left_arm[0, 0], left_arm[1, 0]],
                     [left_arm[0, 1], left_arm[1, 1]],
                     [left_arm[0, 2], left_arm[1, 2]],
                     color='cyan', linewidth=3, alpha=0.8)

    if 'right_arm_coords' in arm_data:
        right_arm = arm_data['right_arm_coords']  # Shape: (2, 3)
        ax.scatter(right_arm[:, 0], right_arm[:, 1], right_arm[:, 2],
                  c='magenta', s=150, alpha=1.0, marker='v', label='Right Arm')
        # Connect right arm points
        if len(right_arm) == 2:
            ax.plot3D([right_arm[0, 0], right_arm[1, 0]],
                     [right_arm[0, 1], right_arm[1, 1]],
                     [right_arm[0, 2], right_arm[1, 2]],
                     color='magenta', linewidth=3, alpha=0.8)

    # Add joint labels for key joints only
    key_joints = [0, 1, 8, 2, 3, 6, 7] if upper_body_only else [0, 1, 8, 2, 3, 6, 7, 9, 10, 13, 14]
    key_joints = [j for j in key_joints if j in joint_indices]

    for i in key_joints:
        joint_name = skeleton_joints.joint_indices.get(i, f"Joint{i}")
        error = np.linalg.norm(pred_joints[i] - gt_joints[i])

        # Position label near GT joint
        gt_pos = gt_joints[i]
        ax.text(gt_pos[0], gt_pos[1], gt_pos[2],
               f'{i}:{joint_name}\n{error:.1f}mm',
               fontsize=7, alpha=0.9,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))

    # Add arm labels if available
    if 'left_arm_coords' in arm_data:
        for i, pos in enumerate(arm_data['left_arm_coords']):
            ax.text(pos[0], pos[1], pos[2], f'L{i+1}', fontsize=8, color='cyan')

    if 'right_arm_coords' in arm_data:
        for i, pos in enumerate(arm_data['right_arm_coords']):
            ax.text(pos[0], pos[1], pos[2], f'R{i+1}', fontsize=8, color='magenta')

    # Set axis properties
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Frame: {identifier}{title_suffix}')

    # Set equal aspect ratio using all displayed joints
    all_joints = np.vstack([gt_subset, pred_subset])
    max_range = np.array([
        all_joints[:, 0].max() - all_joints[:, 0].min(),
        all_joints[:, 1].max() - all_joints[:, 1].min(),
        all_joints[:, 2].max() - all_joints[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_joints[:, 0].max() + all_joints[:, 0].min()) * 0.5
    mid_y = (all_joints[:, 1].max() + all_joints[:, 1].min()) * 0.5
    mid_z = (all_joints[:, 2].max() + all_joints[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc='upper right')


def create_comparison_visualization(samples, upper_body_only=True):
    """
    Create a figure with multiple subplots for each sample

    Args:
        samples: List of (point_cloud, gt_joints, pred_joints, identifier) tuples
        upper_body_only: If True, only show upper body joints
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

    for i, (point_cloud, gt_joints, pred_joints, arm_data, identifier) in enumerate(samples[:n_samples]):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        visualize_comparison(ax, point_cloud, gt_joints, pred_joints, arm_data, identifier, upper_body_only)

    joint_type = "Upper Body" if upper_body_only else "All"
    plt.suptitle(f'SPiKE Prediction vs Ground Truth Comparison ({joint_type} Joints)', fontsize=16)
    plt.tight_layout()


def print_error_statistics(samples, upper_body_only=True):
    """
    Print error statistics for the loaded samples

    Args:
        samples: List of (point_cloud, gt_joints, pred_joints, identifier) tuples
        upper_body_only: If True, only analyze upper body joints
    """
    print("\n" + "="*60)
    print("Prediction Error Statistics")
    print("="*60)

    joint_indices = list(range(9)) if upper_body_only else list(range(15))
    joint_names = ['Head', 'Neck', 'R Shoulder', 'L Shoulder', 'R Elbow', 'L Elbow',
                   'R Hand', 'L Hand', 'Torso', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Foot', 'L Foot']

    all_errors = []

    for i, (point_cloud, gt_joints, pred_joints, arm_data, identifier) in enumerate(samples):
        print(f"\nSample {i+1} (ID: {identifier}):")
        errors = calculate_joint_errors(gt_joints, pred_joints)

        for j in joint_indices:
            joint_name = joint_names[j]
            error = errors[j]
            print(f"  {j:2d} {joint_name:12s}: {error:6.1f} mm")
            all_errors.append(error)

    # Overall statistics
    all_errors = np.array(all_errors)
    print(f"\nOverall Statistics ({len(joint_indices)} joints, {len(samples)} samples):")
    print(f"  Mean error: {all_errors.mean():.2f} mm")
    print(f"  Std error:  {all_errors.std():.2f} mm")
    print(f"  Min error:  {all_errors.min():.2f} mm")
    print(f"  Max error:  {all_errors.max():.2f} mm")


def main():
    # Paths to data
    train_dir = "/home/oliver/Documents/data/Mocap/train"
    train_labels_file = "/home/oliver/Documents/data/Mocap/train_labels.h5"
    arm_labels_file = "/home/oliver/Documents/data/Mocap/arm_labels.h5"
    inference_file = "/home/oliver/Documents/SPiKE/experiments/Custom/1/log/inference_labels_world.h5"

    # Number of random samples to visualize
    num_samples = 6

    print(f"Loading {num_samples} random samples for prediction comparison...")
    print(f"Train directory: {train_dir}")
    print(f"Train labels file: {train_labels_file}")
    print(f"Arm labels file: {arm_labels_file}")
    print(f"Inference file: {inference_file}")

    # Load random samples
    samples = load_prediction_data(train_dir, train_labels_file, arm_labels_file, inference_file, num_samples)

    if len(samples) == 0:
        print("No valid samples found!")
        return

    print(f"\nSuccessfully loaded {len(samples)} samples")

    # Print error statistics (upper body only)
    print_error_statistics(samples, upper_body_only=True)

    # Create visualization (upper body only)
    create_comparison_visualization(samples, upper_body_only=True)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()