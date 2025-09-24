"""
Convert Mocap data from spike_format to original training format
- Point clouds: /home/oliver/Documents/data/Mocap/train/*.npz (numbered 0.npz, 1.npz, etc.)
- Labels: /home/oliver/Documents/data/Mocap/train_labels.h5 with same structure as ITOP
"""

import os
import numpy as np
import h5py
import glob
from tqdm import tqdm

def load_arm_coordinates(session_path):
    """
    Load arm coordinates data if available

    Args:
        session_path: Session directory path

    Returns:
        dict: Dictionary with arm coordinate data, or None if not found
    """
    arm_file = os.path.join(session_path, "arm_coordinates.h5")

    if not os.path.exists(arm_file):
        return None

    try:
        arm_data = {}
        with h5py.File(arm_file, 'r') as f:
            for key in f.keys():
                arm_data[key] = f[key][:]

        print(f"    Found arm data: {list(arm_data.keys())}")
        return arm_data
    except Exception as e:
        print(f"    Error loading arm coordinates: {e}")
        return None

def load_mocap_session_data(session_path):
    """
    Load single Mocap session data including arm coordinates

    Returns:
        tuple: (point_clouds_list, joints_coords_array, frame_ids_array, is_valid_array, arm_data)
    """
    pointclouds_dir = os.path.join(session_path, "pointclouds")
    labels_file = os.path.join(session_path, "labels.h5")

    if not os.path.exists(pointclouds_dir) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing required files: {session_path}")

    # Load labels
    with h5py.File(labels_file, 'r') as f:
        joints_coords = f['real_world_coordinates'][:]
        frame_ids = f['id'][:]
        is_valid = f['is_valid'][:]

    # Load arm coordinates if available
    arm_data = load_arm_coordinates(session_path)

    # Load point clouds
    pc_files = sorted([f for f in os.listdir(pointclouds_dir) if f.endswith('.npz')])
    point_clouds = []

    for i, pc_file in enumerate(pc_files[:len(joints_coords)]):
        pc_path = os.path.join(pointclouds_dir, pc_file)
        pc_data = np.load(pc_path)
        point_clouds.append(pc_data['arr_0'])

    print(f"Loaded session {os.path.basename(session_path)}: {len(point_clouds)} frames")
    return point_clouds, joints_coords, frame_ids, is_valid, arm_data

def convert_mocap_to_train_format(input_dir, output_train_dir, output_labels_file, output_arm_file):
    """
    Convert all Mocap sessions to training format with separate arm data file

    Args:
        input_dir: Path to spike_format directory with sessions
        output_train_dir: Path to output train directory
        output_labels_file: Path to output train_labels.h5 file
        output_arm_file: Path to output arm_labels.h5 file
    """
    # Get all session directories
    session_dirs = [d for d in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, d))]
    session_dirs.sort()

    print(f"Found {len(session_dirs)} sessions to convert")

    # Collect all data
    all_point_clouds = []
    all_joints = []
    all_frame_ids = []
    all_is_valid = []
    all_arm_data = {}  # Dictionary to store concatenated arm data
    global_frame_idx = 0

    for session_name in tqdm(session_dirs, desc="Processing sessions"):
        session_path = os.path.join(input_dir, session_name)

        try:
            point_clouds, joints_coords, frame_ids, is_valid, arm_data = load_mocap_session_data(session_path)

            # Add session data to global arrays
            all_point_clouds.extend(point_clouds)
            all_joints.extend(joints_coords)
            all_is_valid.extend(is_valid)

            # Collect arm data
            if arm_data:
                for arm_key, arm_coords in arm_data.items():
                    if arm_key not in all_arm_data:
                        all_arm_data[arm_key] = []

                    # Ensure arm data matches the number of frames
                    if len(arm_coords) == len(point_clouds):
                        all_arm_data[arm_key].extend(arm_coords)
                    else:
                        print(f"    Warning: arm data {arm_key} length mismatch: {len(arm_coords)} vs {len(point_clouds)}")
                        # Pad or truncate to match
                        if len(arm_coords) < len(point_clouds):
                            # Pad with zeros
                            padding_shape = list(arm_coords[0].shape) if len(arm_coords) > 0 else [3]
                            padding = np.zeros((len(point_clouds) - len(arm_coords), *padding_shape))
                            padded_coords = np.concatenate([arm_coords, padding], axis=0)
                            all_arm_data[arm_key].extend(padded_coords)
                        else:
                            # Truncate
                            all_arm_data[arm_key].extend(arm_coords[:len(point_clouds)])

            # Create global frame identifiers in ITOP format: "PersonID_FrameID"
            # ITOP uses format "XX_YYYYY" where XX is person ID and YYYYY is frame
            # Use simple incremental numbering for consistency
            for local_frame_idx in range(len(point_clouds)):
                # Format: "00_XXXXX" where XXXXX is the global frame index
                identifier = f"00_{global_frame_idx:05d}"
                all_frame_ids.append(identifier)
                global_frame_idx += 1

        except Exception as e:
            print(f"Error processing session {session_name}: {e}")
            continue

    print(f"Total frames collected: {len(all_point_clouds)}")

    # Save point clouds as numbered .npz files matching frame IDs
    print("Saving point clouds...")
    for i, pc in enumerate(tqdm(all_point_clouds, desc="Saving point clouds")):
        # Use frame ID number directly to ensure consistency
        frame_num = int(all_frame_ids[i].split('_')[-1])
        pc_path = os.path.join(output_train_dir, f"{frame_num}.npz")
        np.savez_compressed(pc_path, pc.astype(np.float16))

    # Save labels in H5 format matching ITOP structure (UNCHANGED)
    print("Saving main labels...")
    with h5py.File(output_labels_file, 'w') as f:
        # Convert string identifiers to bytes for H5 compatibility
        id_bytes = [id_str.encode('utf-8') for id_str in all_frame_ids]

        # Create datasets matching ITOP format - EXACTLY as before
        f.create_dataset('id', data=id_bytes)
        f.create_dataset('real_world_coordinates', data=np.array(all_joints))
        f.create_dataset('is_valid', data=np.array(all_is_valid))

        print(f"Saved labels: {len(all_frame_ids)} frames")
        print(f"Joint coordinates shape: {np.array(all_joints).shape}")
        print(f"Is valid shape: {np.array(all_is_valid).shape}")

    # Save arm data separately if available
    if all_arm_data:
        print("\nSaving arm coordinates to separate file...")
        with h5py.File(output_arm_file, 'w') as f:
            # Use same ID format for consistency
            id_bytes = [id_str.encode('utf-8') for id_str in all_frame_ids]
            f.create_dataset('id', data=id_bytes)

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

            # Create arm metadata using actual dataset names created
            created_datasets = []
            for arm_key in all_arm_data.keys():
                if len(all_arm_data[arm_key]) == len(all_frame_ids):
                    dataset_name = arm_key if arm_key not in reserved_names else f"arm_{arm_key}"
                    created_datasets.append(dataset_name)

            if created_datasets:
                f.attrs['arm_data_keys'] = [key.encode('utf-8') for key in created_datasets]
                f.attrs['num_arm_datasets'] = len(created_datasets)
                f.attrs['num_frames'] = len(all_frame_ids)
                print(f"Arm data metadata: {len(created_datasets)} datasets, {len(all_frame_ids)} frames")

        print(f"Arm data saved to: {output_arm_file}")
    else:
        print("No arm data found in any session - no arm file created")

def main():
    input_dir = "/home/oliver/Documents/data/Mocap/spike_format"
    output_train_dir = "/home/oliver/Documents/data/Mocap/train"
    output_labels_file = "/home/oliver/Documents/data/Mocap/train_labels.h5"
    output_arm_file = "/home/oliver/Documents/data/Mocap/arm_labels.h5"

    # Ensure output directory exists
    os.makedirs(output_train_dir, exist_ok=True)

    # Convert data
    convert_mocap_to_train_format(input_dir, output_train_dir, output_labels_file, output_arm_file)

    print(f"\nConversion complete!")
    print(f"Point clouds saved to: {output_train_dir}")
    print(f"Main labels saved to: {output_labels_file}")

    # Check if arm file was created
    if os.path.exists(output_arm_file):
        print(f"Arm labels saved to: {output_arm_file}")
    else:
        print("No arm data available - arm labels file not created")

    # Verify output structure
    pc_count = len([f for f in os.listdir(output_train_dir) if f.endswith('.npz')])
    print(f"Final verification: {pc_count} point cloud files created")

if __name__ == "__main__":
    main()