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


# ----------------------------- Utils -----------------------------
_ROT_X_M90 = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0,-1, 0]], dtype=np.float32)

def rotate_x_minus_90(points: np.ndarray) -> np.ndarray:
    """Rotate by -90° around X-axis; supports (N,3) / (15,3) / batched."""
    pts = np.asarray(points)
    if pts.ndim == 2 and pts.shape[-1] == 3:
        return pts @ _ROT_X_M90.T
    # vectorized for (..., 3)
    flat = pts.reshape(-1, 3) @ _ROT_X_M90.T
    return flat.reshape(pts.shape)


def _ensure_bytes_id(s: str) -> bytes:
    return s.encode("utf-8")


def _gen_new_ids(start_frame: int, count: int):
    """Generate bytes IDs: 00_XXXXX for frames [start_frame-count, start_frame)."""
    base = start_frame - count
    return [_ensure_bytes_id(f"00_{base + i:05d}") for i in range(count)]


# ----------------------------- Core -----------------------------
def load_session_data(session_path):
    """
    Load all data from a single ITOP session

    Returns:
        tuple: (point_clouds, joints_coords, frame_ids, is_valid, arm_data)
    """
    pointclouds_dir = os.path.join(session_path, "pointclouds")
    labels_file     = os.path.join(session_path, "labels.h5")
    arm_file        = os.path.join(session_path, "arm_coordinates.h5")

    if not os.path.exists(pointclouds_dir) or not os.path.exists(labels_file) or not os.path.exists(arm_file):
        raise FileNotFoundError(f"Missing required files in session: {session_path}")

    # labels
    with h5py.File(labels_file, 'r') as f:
        joints_coords = f['real_world_coordinates'][:]
        frame_ids     = f['id'][:]
        is_valid      = f['is_valid'][:]

    # point clouds (take first len(joints) files, aligned)
    pc_files = sorted([f for f in os.listdir(pointclouds_dir) if f.endswith('.npz')])
    pcs = []
    for pc_file in pc_files[:len(joints_coords)]:
        pc = np.load(os.path.join(pointclouds_dir, pc_file))['arr_0']
        pcs.append(rotate_x_minus_90(pc))

    # joints
    rotated_joints = rotate_x_minus_90(joints_coords)

    # arm (assumed to exist)
    arm_info = {}
    with h5py.File(arm_file, 'r') as f:
        left  = f['left_arm_coords'][:]
        right = f['right_arm_coords'][:]
        arm_info['left_arm_coords']  = rotate_x_minus_90(left)
        arm_info['right_arm_coords'] = rotate_x_minus_90(right)
        arm_info['id'] = f['id'][:]

    print(f"Loaded session {os.path.basename(session_path)}: {len(pcs)} frames")
    return pcs, rotated_joints, frame_ids, is_valid, arm_info


def convert_itop_to_training(itop_dir, output_train_dir, output_labels_file, arm_labels_file):
    """
    Convert ITOP format data directly to training format in memory
    """
    print("=== Converting ITOP format to training format ===")

    session_dirs = sorted([d for d in os.listdir(itop_dir)
                           if os.path.isdir(os.path.join(itop_dir, d))])
    print(f"Found {len(session_dirs)} sessions to process")
    print(f"Input directory: {itop_dir}")
    print(f"Output train directory: {output_train_dir}")
    print(f"Output labels file: {output_labels_file}")

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_file), exist_ok=True)

    all_joints, all_frame_ids, all_is_valid = [], [], []
    all_arm_data = {'left_arm_coords': [], 'right_arm_coords': []}
    frame_counter = 0

    for session_dir in tqdm(session_dirs, desc="Processing sessions"):
        session_path = os.path.join(itop_dir, session_dir)
        try:
            pcs, joints_coords, frame_ids, is_valid, arm_data = load_session_data(session_path)

            # save pcs
            for pc in pcs:
                np.savez_compressed(os.path.join(output_train_dir, f"{frame_counter}.npz"), pc)
                frame_counter += 1

            # collect joints/ids/valid
            all_joints.extend(joints_coords)
            new_ids = _gen_new_ids(frame_counter, len(joints_coords))
            all_frame_ids.extend(new_ids)
            all_is_valid.extend(is_valid)

            # arm collect with pad/truncate to match joints length
            for arm_key in ('left_arm_coords', 'right_arm_coords'):
                arr = arm_data[arm_key]
                if len(arr) == len(joints_coords):
                    all_arm_data[arm_key].extend(arr)
                elif len(arr) < len(joints_coords):
                    pad_shape = arr[0].shape if len(arr) > 0 else (2, 3)
                    missing = len(joints_coords) - len(arr)
                    pad = np.zeros((missing, *pad_shape), dtype=arr.dtype if len(arr) > 0 else np.float32)
                    all_arm_data[arm_key].extend(np.concatenate([arr, pad], axis=0))
                    print(f"    Warning: arm data {arm_key} length mismatch: {len(arr)} vs {len(joints_coords)} (padded)")
                else:
                    all_arm_data[arm_key].extend(arr[:len(joints_coords)])
                    print(f"    Warning: arm data {arm_key} length mismatch: {len(arr)} vs {len(joints_coords)} (truncated)")

        except Exception as e:
            print(f"Error processing session {session_dir}: {e}")
            continue

    print(f"Processed {len(all_joints)} total frames")

    # save joint labels
    with h5py.File(output_labels_file, 'w') as f:
        f.create_dataset('real_world_coordinates', data=np.array(all_joints))
        f.create_dataset('id', data=all_frame_ids)  # bytes array
        f.create_dataset('is_valid', data=np.array(all_is_valid))
    print(f"Labels saved to: {output_labels_file}")

    # save arm labels (always present by assumption)
    print("\nSaving arm coordinates to separate file...")
    with h5py.File(arm_labels_file, 'w') as f:
        f.create_dataset('id', data=np.array(all_frame_ids, dtype='S'))
        reserved = {'id', 'is_valid', 'real_world_coordinates'}
        for k, seq in all_arm_data.items():
            name = k if k not in reserved else f"arm_{k}"
            arr = np.array(seq)
            f.create_dataset(name, data=arr)
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"Arm labels saved to: {arm_labels_file}")

    # summary prints (kept identical in spirit)
    datasets_info = [
        f"  - real_world_coordinates: {len(all_joints)} frames × 15 joints × 3 coords",
        f"  - id: {len(all_frame_ids)} frame IDs",
        f"  - is_valid: {len(all_is_valid)} validity flags"
    ]
    print("Joint datasets saved:")
    for info in datasets_info:
        print(info)

    print("Arm datasets saved:")
    for k in all_arm_data.keys():
        arr = np.array(all_arm_data[k])
        print(f"  - {k}: {arr.shape}")

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
