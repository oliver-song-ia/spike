"""
Convert ITOP format data to SPIKE format with -90 degree rotation around X-axis
- Point clouds: rotate all points by -90 degrees around X-axis
- Joint coordinates: apply same rotation transformation
- Save to spike_format directory with same structure as itop_format
- Also convert arm_coordinates.h5 if present
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

    # Handle both (N, 3) and (15, 3) shapes
    if points.ndim == 2 and points.shape[1] == 3:
        return points @ rotation_matrix.T
    else:
        raise ValueError(f"Expected shape (N, 3), got {points.shape}")

def convert_arm_coordinates(input_session_path, output_session_path):
    """
    Convert arm_coordinates.h5 file if it exists

    Args:
        input_session_path: Path to input session directory
        output_session_path: Path to output session directory

    Returns:
        bool: True if file was converted, False if not found
    """
    input_arm_file = os.path.join(input_session_path, "arm_coordinates.h5")
    output_arm_file = os.path.join(output_session_path, "arm_coordinates.h5")

    if not os.path.exists(input_arm_file):
        return False

    print(f"  Converting arm_coordinates.h5")

    with h5py.File(input_arm_file, 'r') as input_f:
        with h5py.File(output_arm_file, 'w') as output_f:
            # Copy all datasets, rotating coordinate data
            for key in input_f.keys():
                data = input_f[key][:]

                # Check if this is coordinate data that needs rotation
                is_coordinate_data = False

                # Check for common coordinate field names
                if key in ['real_world_coordinates', 'coordinates']:
                    is_coordinate_data = True
                # Check for arm-related fields with 3D coordinates
                elif 'arm' in key.lower() and data.ndim >= 2 and data.shape[-1] == 3:
                    is_coordinate_data = True
                # Check for any field ending with 'coordinates'
                elif key.lower().endswith('coordinates') and data.ndim >= 2 and data.shape[-1] == 3:
                    is_coordinate_data = True
                # General check for 3D coordinate arrays
                elif data.ndim >= 2 and data.shape[-1] == 3 and data.dtype in [np.float32, np.float64]:
                    is_coordinate_data = True

                if is_coordinate_data:
                    # Rotate coordinate data
                    if data.ndim == 3:  # (N, joints, 3)
                        rotated_data = np.zeros_like(data)
                        for i in range(len(data)):
                            rotated_data[i] = rotate_x_minus_90(data[i])
                    elif data.ndim == 2:  # (joints, 3)
                        rotated_data = rotate_x_minus_90(data)
                    elif data.ndim == 1 and len(data) >= 3 and len(data) % 3 == 0:
                        # Handle flattened coordinates (x1,y1,z1,x2,y2,z2,...)
                        reshaped = data.reshape(-1, 3)
                        rotated_reshaped = rotate_x_minus_90(reshaped)
                        rotated_data = rotated_reshaped.reshape(data.shape)
                    else:
                        rotated_data = data

                    output_f.create_dataset(key, data=rotated_data)
                    print(f"    Rotated {key}: {data.shape} -> {rotated_data.shape}")
                else:
                    # Copy non-coordinate data as-is
                    output_f.create_dataset(key, data=data)
                    print(f"    Copied {key}: {data.shape}")

    return True

def convert_session_itop2spike(input_session_path, output_session_path):
    """
    Convert single session from itop_format to spike_format with X-axis -90 degree rotation

    Args:
        input_session_path: Path to input session directory
        output_session_path: Path to output session directory

    Returns:
        int: Number of point cloud files converted
    """
    # Create output directories
    os.makedirs(output_session_path, exist_ok=True)
    output_pointclouds_dir = os.path.join(output_session_path, "pointclouds")
    os.makedirs(output_pointclouds_dir, exist_ok=True)

    # Load labels
    input_labels_file = os.path.join(input_session_path, "labels.h5")
    output_labels_file = os.path.join(output_session_path, "labels.h5")

    if not os.path.exists(input_labels_file):
        raise FileNotFoundError(f"Labels file not found: {input_labels_file}")

    print(f"Converting session: {os.path.basename(input_session_path)}")

    # Process labels with rotation
    with h5py.File(input_labels_file, 'r') as input_f:
        joints_coords = input_f['real_world_coordinates'][:]  # Shape: (N, 15, 3)
        frame_ids = input_f['id'][:]
        is_valid = input_f['is_valid'][:]

        # Rotate joint coordinates
        rotated_joints = np.zeros_like(joints_coords)
        for i in range(len(joints_coords)):
            rotated_joints[i] = rotate_x_minus_90(joints_coords[i])

        # Save rotated labels
        with h5py.File(output_labels_file, 'w') as output_f:
            output_f.create_dataset('id', data=frame_ids)
            output_f.create_dataset('real_world_coordinates', data=rotated_joints)
            output_f.create_dataset('is_valid', data=is_valid)

    # Convert arm_coordinates.h5 if present
    arm_converted = convert_arm_coordinates(input_session_path, output_session_path)
    if arm_converted:
        print(f"  Successfully converted arm_coordinates.h5")

    # Process point clouds with rotation
    input_pointclouds_dir = os.path.join(input_session_path, "pointclouds")
    pc_files = sorted([f for f in os.listdir(input_pointclouds_dir) if f.endswith('.npz')])

    for pc_file in tqdm(pc_files[:len(joints_coords)], desc="Rotating point clouds"):
        input_pc_path = os.path.join(input_pointclouds_dir, pc_file)
        output_pc_path = os.path.join(output_pointclouds_dir, pc_file)

        # Load, rotate, and save point cloud
        pc_data = np.load(input_pc_path)
        original_pc = pc_data['arr_0']  # Shape: (N, 3)

        # Apply rotation
        rotated_pc = rotate_x_minus_90(original_pc)

        # Save rotated point cloud
        np.savez_compressed(output_pc_path, rotated_pc.astype(original_pc.dtype))

    print(f"  Converted {len(pc_files)} point cloud files")
    return len(pc_files)

def convert_all_itop2spike(input_dir, output_dir):
    """
    Convert all sessions from itop_format to spike_format

    Args:
        input_dir: Path to itop_format directory
        output_dir: Path to spike_format directory
    """
    # Get all session directories
    session_dirs = [d for d in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, d))]
    session_dirs.sort()

    print(f"Found {len(session_dirs)} sessions to convert")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    total_frames = 0
    total_arm_files = 0

    # Convert each session
    for session_name in tqdm(session_dirs, desc="Converting sessions"):
        input_session_path = os.path.join(input_dir, session_name)
        output_session_path = os.path.join(output_dir, session_name)

        try:
            frames_converted = convert_session_itop2spike(input_session_path, output_session_path)
            total_frames += frames_converted

            # Check if arm coordinates were converted
            if os.path.exists(os.path.join(output_session_path, "arm_coordinates.h5")):
                total_arm_files += 1

        except Exception as e:
            print(f"Error converting session {session_name}: {e}")
            continue

    print(f"\nConversion complete!")
    print(f"Total sessions converted: {len(session_dirs)}")
    print(f"Total frames converted: {total_frames}")
    print(f"Total arm coordinate files converted: {total_arm_files}")
    print(f"Output saved to: {output_dir}")

def verify_rotation():
    """
    Verify rotation transformation with sample data

    Returns:
        bool: True if verification passed, False otherwise
    """
    print("Verifying rotation transformation...")

    # Test point: (1, 0, 0) -> should become (1, 0, 0) (X unchanged)
    # Test point: (0, 1, 0) -> should become (0, 0, -1) (Y -> -Z)
    # Test point: (0, 0, 1) -> should become (0, 1, 0) (Z -> Y)

    test_points = np.array([
        [1, 0, 0],  # X-axis
        [0, 1, 0],  # Y-axis
        [0, 0, 1],  # Z-axis
    ])

    expected_result = np.array([
        [1, 0, 0],   # X unchanged
        [0, 0, -1],  # Y -> -Z
        [0, 1, 0],   # Z -> Y
    ])

    rotated = rotate_x_minus_90(test_points)

    print("Rotation verification:")
    print("Original -> Rotated (Expected)")
    for i, (orig, rot, exp) in enumerate(zip(test_points, rotated, expected_result)):
        print(f"  {orig} -> {rot} ({exp})")
        if not np.allclose(rot, exp, atol=1e-6):
            print(f"  ERROR: Mismatch at point {i}")
            return False

    print("  SUCCESS: Rotation transformation verified")
    return True

def main():
    """Main function"""
    # Verify rotation first
    if not verify_rotation():
        print("ERROR: Rotation verification failed!")
        return

    input_dir = "/home/oliver/Documents/data/Mocap/itop_format"
    output_dir = "/home/oliver/Documents/data/Mocap/spike_format"

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Convert all sessions
    convert_all_itop2spike(input_dir, output_dir)

    # Verify output structure
    if os.path.exists(output_dir):
        session_count = len([d for d in os.listdir(output_dir)
                           if os.path.isdir(os.path.join(output_dir, d))])
        print(f"Final verification: {session_count} session directories created")

if __name__ == "__main__":
    main()