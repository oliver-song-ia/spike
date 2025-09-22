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

def load_mocap_session_data(session_path):
    """
    Load single Mocap session data

    Returns:
        tuple: (point_clouds_list, joints_coords_array, frame_ids_array, is_valid_array)
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

    # Load point clouds
    pc_files = sorted([f for f in os.listdir(pointclouds_dir) if f.endswith('.npz')])
    point_clouds = []

    for i, pc_file in enumerate(pc_files[:len(joints_coords)]):
        pc_path = os.path.join(pointclouds_dir, pc_file)
        pc_data = np.load(pc_path)
        point_clouds.append(pc_data['arr_0'])

    print(f"Loaded session {os.path.basename(session_path)}: {len(point_clouds)} frames")
    return point_clouds, joints_coords, frame_ids, is_valid

def convert_mocap_to_train_format(input_dir, output_train_dir, output_labels_file):
    """
    Convert all Mocap sessions to training format

    Args:
        input_dir: Path to spike_format directory with sessions
        output_train_dir: Path to output train directory
        output_labels_file: Path to output train_labels.h5 file
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
    global_frame_idx = 0

    for session_name in tqdm(session_dirs, desc="Processing sessions"):
        session_path = os.path.join(input_dir, session_name)

        try:
            point_clouds, joints_coords, frame_ids, is_valid = load_mocap_session_data(session_path)

            # Add session data to global arrays
            all_point_clouds.extend(point_clouds)
            all_joints.extend(joints_coords)
            all_is_valid.extend(is_valid)

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

    # Save labels in H5 format matching ITOP structure
    print("Saving labels...")
    with h5py.File(output_labels_file, 'w') as f:
        # Convert string identifiers to bytes for H5 compatibility
        id_bytes = [id_str.encode('utf-8') for id_str in all_frame_ids]

        # Create datasets matching ITOP format
        f.create_dataset('id', data=id_bytes)
        f.create_dataset('real_world_coordinates', data=np.array(all_joints))
        f.create_dataset('is_valid', data=np.array(all_is_valid))

        print(f"Saved labels: {len(all_frame_ids)} frames")
        print(f"Joint coordinates shape: {np.array(all_joints).shape}")
        print(f"Is valid shape: {np.array(all_is_valid).shape}")

def main():
    input_dir = "/home/oliver/Documents/data/Mocap/spike_format"
    output_train_dir = "/home/oliver/Documents/data/Mocap/train"
    output_labels_file = "/home/oliver/Documents/data/Mocap/train_labels.h5"

    # Ensure output directory exists
    os.makedirs(output_train_dir, exist_ok=True)

    # Convert data
    convert_mocap_to_train_format(input_dir, output_train_dir, output_labels_file)

    print(f"✅ Conversion complete!")
    print(f"Point clouds saved to: {output_train_dir}")
    print(f"Labels saved to: {output_labels_file}")

    # Verify output structure
    pc_count = len([f for f in os.listdir(output_train_dir) if f.endswith('.npz')])
    print(f"Final verification: {pc_count} point cloud files created")

if __name__ == "__main__":
    main()