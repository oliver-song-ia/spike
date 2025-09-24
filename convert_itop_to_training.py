#!/usr/bin/env python3
"""
Independent script to convert ITOP format data directly to training format
No intermediate files, direct memory-based conversion
"""

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

# Add project root to path for config utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config


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


def convert_itop_to_training(itop_dir, output_train_dir, output_labels_file, arm_labels_file):
    """
    Convert ITOP format data directly to training format in memory

    Args:
        itop_dir: Input ITOP format directory
        output_train_dir: Output training directory for .npz files
        output_labels_file: Output labels .h5 file (joint data only, consistent with original format)
        arm_labels_file: Output arm labels .h5 file for robot planning
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
    all_arm_data = {}  # Dictionary to store concatenated arm data

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

            # Collect arm data following the reference pattern
            if arm_data:
                for arm_key, arm_coords in arm_data.items():
                    if arm_key == 'id':
                        continue  # Skip ID, we'll generate our own

                    if arm_key not in all_arm_data:
                        all_arm_data[arm_key] = []

                    # Ensure arm data matches the number of frames
                    if len(arm_coords) == len(joints_coords):
                        all_arm_data[arm_key].extend(arm_coords)
                    else:
                        print(f"    Warning: arm data {arm_key} length mismatch: {len(arm_coords)} vs {len(joints_coords)}")
                        # Pad or truncate to match
                        if len(arm_coords) < len(joints_coords):
                            # Pad with zeros
                            padding_shape = list(arm_coords[0].shape) if len(arm_coords) > 0 else [2, 3]
                            padding = np.zeros((len(joints_coords) - len(arm_coords), *padding_shape))
                            padded_coords = np.concatenate([arm_coords, padding], axis=0)
                            all_arm_data[arm_key].extend(padded_coords)
                        else:
                            # Truncate
                            all_arm_data[arm_key].extend(arm_coords[:len(joints_coords)])

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

    # Save arm data separately if available (following reference pattern)
    if all_arm_data:
        print("\nSaving arm coordinates to separate file...")
        with h5py.File(arm_labels_file, 'w') as f:
            # Use same ID format for consistency
            f.create_dataset('id', data=np.array(all_frame_ids, dtype='S'))

            # Save each arm dataset, avoiding conflicts with reserved names
            reserved_names = {'id', 'is_valid', 'real_world_coordinates'}
            for arm_key, arm_coords in all_arm_data.items():
                if len(arm_coords) == len(all_frame_ids):
                    # Check for name conflicts and rename if necessary
                    dataset_name = arm_key
                    if dataset_name in reserved_names:
                        dataset_name = f"arm_{arm_key}"
                        print(f"  Warning: Renamed {arm_key} to {dataset_name} to avoid conflict")

                    arm_array = np.array(arm_coords)
                    f.create_dataset(dataset_name, data=arm_array)
                    print(f"  {dataset_name}: shape={arm_array.shape}, dtype={arm_array.dtype}")
                else:
                    print(f"  Warning: Skipping {arm_key} due to length mismatch: {len(arm_coords)} vs {len(all_frame_ids)}")

        print(f"Arm labels saved to: {arm_labels_file}")
    else:
        print("No arm data found in any session - no arm file created")

    # Print summary
    datasets_info = [
        f"  - real_world_coordinates: {len(all_joints)} frames × 15 joints × 3 coords",
        f"  - id: {len(all_frame_ids)} frame IDs",
        f"  - is_valid: {len(all_is_valid)} validity flags"
    ]

    print("Joint datasets saved:")
    for info in datasets_info:
        print(info)

    if all_arm_data:
        print("Arm datasets saved:")
        for arm_key in all_arm_data.keys():
            if len(all_arm_data[arm_key]) == len(all_frame_ids):
                arm_array = np.array(all_arm_data[arm_key])
                print(f"  - {arm_key}: {arm_array.shape}")
    else:
        print("No arm datasets created")

    print(f"Training point clouds saved to: {output_train_dir} ({frame_counter} files)")

    return len(all_joints)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert ITOP format directly to training format")
    parser.add_argument('--config', type=str, default='experiments/Custom/1',
                       help='Config file path')

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Get paths from config
    itop_dir = config.get('dataset_path')
    data_output_path = config.get('data_output_path')

    # Set up output paths
    train_dir = os.path.join(data_output_path, 'train')
    labels_file = os.path.join(data_output_path, 'train_labels.h5')
    arm_labels_file = os.path.join(data_output_path, 'arm_labels.h5')

    # Validate paths from config
    if not itop_dir:
        print("Error: 'dataset_path' not specified in config file")
        return 1
    if not data_output_path:
        print("Error: 'data_output_path' not specified in config file")
        return 1

    # Validate input directory
    if not os.path.exists(itop_dir):
        print(f"Error: Input directory does not exist: {itop_dir}")
        print(f"Check 'dataset_path' in config file: {args.config}")
        return 1

    print(f"Input ITOP directory: {itop_dir}")
    print(f"Output training directory: {train_dir}")
    print(f"Output labels file: {labels_file}")
    print(f"Output arm labels file: {arm_labels_file}")

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)

    try:
        # Run conversion
        total_frames = convert_itop_to_training(
            itop_dir,
            train_dir,
            labels_file,
            arm_labels_file
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