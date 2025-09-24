#!/usr/bin/env python3
"""
Independent script to convert ITOP format data directly to training format
No intermediate files, direct memory-based conversion
"""

import os
import numpy as np
import h5py
from tqdm import tqdm


def rotate_x_minus_90(points):
    """
    Rotate points by -90 degrees around X-axis

    Rotation matrix for -90 degrees around X-axis:
    [1  0  0]
    [0  0  1]
    [0 -1  0]

    Args:
        points: numpy array of shape (N, 3) or (15, 3)

    Returns:
        rotated_points: numpy array of same shape
    """
    rotation_matrix = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)

    return np.dot(points, rotation_matrix.T)


def load_session_data(session_path):
    """
    Load all data from a single ITOP session

    Args:
        session_path: Path to session directory

    Returns:
        tuple: (point_clouds, joints_coords, frame_ids, is_valid, arm_data)
    """
    pointclouds_dir = os.path.join(session_path, "pointclouds")
    labels_file = os.path.join(session_path, "labels.h5")
    arm_file = os.path.join(session_path, "arm_coordinates.h5")

    if not os.path.exists(pointclouds_dir) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing required files in session: {session_path}")

    # Load labels
    with h5py.File(labels_file, 'r') as f:
        joints_coords = f['real_world_coordinates'][:]
        frame_ids = f['id'][:]
        is_valid = f['is_valid'][:]

    # Load point clouds
    pc_files = sorted([f for f in os.listdir(pointclouds_dir) if f.endswith('.npz')])
    point_clouds = []

    for i, pc_file in enumerate(pc_files[:len(joints_coords)]):
        pc_path = os.path.join(pointclouds_dir, pc_file)
        pc_data = np.load(pc_path)
        point_cloud = pc_data['arr_0']  # Shape: (N, 3)

        # Apply rotation transformation
        rotated_pc = rotate_x_minus_90(point_cloud)
        point_clouds.append(rotated_pc)

    # Apply rotation to joint coordinates
    rotated_joints = []
    for joints in joints_coords:
        rotated_joints.append(rotate_x_minus_90(joints))
    rotated_joints = np.array(rotated_joints)

    # Load arm coordinates if available
    arm_data = None
    if os.path.exists(arm_file):
        try:
            arm_info = {}
            with h5py.File(arm_file, 'r') as f:
                if 'left_arm_coords' in f:
                    left_coords = f['left_arm_coords'][:]
                    # Apply rotation to arm coordinates
                    rotated_left = []
                    for coords in left_coords:
                        rotated_left.append(rotate_x_minus_90(coords))
                    arm_info['left_arm_coords'] = np.array(rotated_left)

                if 'right_arm_coords' in f:
                    right_coords = f['right_arm_coords'][:]
                    # Apply rotation to arm coordinates
                    rotated_right = []
                    for coords in right_coords:
                        rotated_right.append(rotate_x_minus_90(coords))
                    arm_info['right_arm_coords'] = np.array(rotated_right)

                if 'id' in f:
                    arm_info['id'] = f['id'][:]

            if arm_info:
                arm_data = arm_info

        except Exception as e:
            print(f"Warning: Could not load arm coordinates from {arm_file}: {e}")

    print(f"Loaded session {os.path.basename(session_path)}: {len(point_clouds)} frames")
    return point_clouds, rotated_joints, frame_ids, is_valid, arm_data


def convert_itop_to_training(itop_dir, output_train_dir, output_labels_file):
    """
    Convert ITOP format data directly to training format in memory

    Args:
        itop_dir: Input ITOP format directory
        output_train_dir: Output training directory for .npz files
        output_labels_file: Output labels .h5 file (joint data only, consistent with original format)
    """
    print("=== Converting ITOP format to training format ===")

    # Get all session directories
    session_dirs = [d for d in os.listdir(itop_dir)
                   if os.path.isdir(os.path.join(itop_dir, d))]
    session_dirs.sort()

    print(f"Found {len(session_dirs)} sessions to process")
    print(f"Input directory: {itop_dir}")
    print(f"Output train directory: {output_train_dir}")
    print(f"Output labels file: {output_labels_file}")

    # Create output directories
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_file), exist_ok=True)

    # Collect all data
    all_joints = []
    all_frame_ids = []
    all_is_valid = []
    arm_data_found = False

    frame_counter = 0

    # Process each session
    for session_dir in tqdm(session_dirs, desc="Processing sessions"):
        session_path = os.path.join(itop_dir, session_dir)

        try:
            point_clouds, joints_coords, frame_ids, is_valid, arm_data = load_session_data(session_path)

            # Save point clouds as numbered .npz files
            for pc in point_clouds:
                pc_file = os.path.join(output_train_dir, f"{frame_counter}.npz")
                np.savez_compressed(pc_file, pc)
                frame_counter += 1

            # Collect label data
            all_joints.extend(joints_coords)

            # Generate new frame IDs in format "00_XXXXX"
            for i in range(len(joints_coords)):
                new_frame_id = f"00_{frame_counter - len(joints_coords) + i:05d}"
                all_frame_ids.append(new_frame_id.encode('utf-8'))

            all_is_valid.extend(is_valid)

            # Check if arm data exists
            if arm_data and ('left_arm_coords' in arm_data or 'right_arm_coords' in arm_data):
                arm_data_found = True

        except Exception as e:
            print(f"Error processing session {session_dir}: {e}")
            continue

    print(f"Processed {len(all_joints)} total frames")

    # Save labels file (only joint data, consistent with original format)
    with h5py.File(output_labels_file, 'w') as f:
        f.create_dataset('real_world_coordinates', data=np.array(all_joints))
        f.create_dataset('id', data=all_frame_ids)
        f.create_dataset('is_valid', data=np.array(all_is_valid))

    print(f"Labels saved to: {output_labels_file}")

    # Print summary
    datasets_info = [
        f"  - real_world_coordinates: {len(all_joints)} frames × 15 joints × 3 coords",
        f"  - id: {len(all_frame_ids)} frame IDs",
        f"  - is_valid: {len(all_is_valid)} validity flags"
    ]

    print("Datasets saved:")
    for info in datasets_info:
        print(info)

    if arm_data_found:
        print("Note: Arm coordinate data found but not saved (use separate arm processing if needed)")
    else:
        print("Note: No arm coordinate data found in sessions")

    print(f"Training point clouds saved to: {output_train_dir} ({frame_counter} files)")

    return len(all_joints)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert ITOP format directly to training format")
    parser.add_argument('--itop-dir', type=str, required=True,
                       help='Input ITOP format directory')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Output training directory for .npz files')
    parser.add_argument('--labels-file', type=str,
                       help='Output labels file path (default: train_labels.h5 in parent of train-dir)')

    args = parser.parse_args()

    # Set up output paths
    output_train_dir = args.train_dir

    if args.labels_file:
        output_labels_file = args.labels_file
    else:
        # Default: put labels file in parent directory of train dir
        parent_dir = os.path.dirname(args.train_dir)
        output_labels_file = os.path.join(parent_dir, 'train_labels.h5')

    # Validate input directory
    if not os.path.exists(args.itop_dir):
        print(f"Error: Input directory does not exist: {args.itop_dir}")
        return 1

    # Create output directories
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_file), exist_ok=True)

    try:
        # Run conversion
        total_frames = convert_itop_to_training(
            args.itop_dir,
            output_train_dir,
            output_labels_file
        )

        print(f"\n=== Conversion Complete ===")
        print(f"Successfully processed {total_frames} frames")
        print(f"Training data ready for SPiKE model training!")

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())