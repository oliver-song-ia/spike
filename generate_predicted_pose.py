"""
Generate predicted poses using trained model on entire dataset
- Runs inference on all samples in test dataset
- Saves predictions to experiments/Custom/1/log/inference_labels.h5
- Maintains same format as original labels for easy comparison
"""

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

from model import model_builder
from trainer_itop import load_data
from utils.config_utils import load_config, set_random_seed


def load_original_centroids(train_dir, train_labels_file):
    """
    Load original point clouds and calculate their centroids for world coordinate conversion

    Args:
        train_dir: Path to directory with original .npz point cloud files
        train_labels_file: Path to train_labels.h5 file

    Returns:
        dict: Mapping from frame_id to original centroid (3,)
    """
    print("Loading original point clouds to calculate true centroids...")

    # Load frame IDs from training labels
    with h5py.File(train_labels_file, 'r') as f:
        identifiers = f['id'][:]

    centroids = {}
    frame_ids = []

    # Extract frame numbers and calculate centroids
    for identifier in tqdm(identifiers, desc="Computing original centroids"):
        # Handle both string and numeric identifiers
        if isinstance(identifier, (bytes, np.bytes_)):
            frame_id_str = identifier.decode('utf-8')
        elif isinstance(identifier, (int, np.integer)):
            frame_id_str = f"00_{identifier:05d}"
        else:
            frame_id_str = str(identifier)
        frame_num = int(frame_id_str.split('_')[-1])  # Extract frame number from "XX_YYYYY"
        frame_ids.append(frame_num)

        # Load original point cloud
        pc_file = os.path.join(train_dir, f"{frame_num}.npz")
        if os.path.exists(pc_file):
            pc_data = np.load(pc_file)
            original_pc = pc_data['arr_0']  # Shape: (N, 3)

            # Calculate true centroid from original world coordinates
            true_centroid = np.mean(original_pc, axis=0)  # Shape: (3,)
            centroids[frame_num] = true_centroid
        else:
            print(f"Warning: Original point cloud not found for frame {frame_num}")
            centroids[frame_num] = np.zeros(3)  # Fallback

    print(f"Loaded centroids for {len(centroids)} frames")
    return centroids


def load_session_info(itop_format_dir):
    """
    Load session information to map frame IDs to trajectory IDs

    Args:
        itop_format_dir: Path to itop_format directory with session folders

    Returns:
        dict: Mapping from frame_id to (traj_id, frame_in_session)
    """
    print("Loading session information...")

    session_dirs = [d for d in os.listdir(itop_format_dir)
                   if os.path.isdir(os.path.join(itop_format_dir, d)) and d.startswith('2025-')]
    session_dirs.sort()

    frame_to_session_map = {}
    current_frame_id = 0

    for traj_id, session_dir in enumerate(session_dirs):
        session_path = os.path.join(itop_format_dir, session_dir)
        labels_file = os.path.join(session_path, 'labels.h5')

        if os.path.exists(labels_file):
            try:
                with h5py.File(labels_file, 'r') as f:
                    session_frame_count = len(f['id'][:])

                # Map each frame in this session to the trajectory ID
                for frame_in_session in range(session_frame_count):
                    # Frame ID format: "00_XXXXX"
                    frame_id = f"00_{current_frame_id:05d}"
                    frame_to_session_map[frame_id] = (traj_id, frame_in_session)
                    current_frame_id += 1

                print(f"  Session {traj_id} ({session_dir}): {session_frame_count} frames")
            except Exception as e:
                print(f"  Warning: Could not read {session_dir}: {e}")

    print(f"Loaded {len(session_dirs)} sessions, {len(frame_to_session_map)} total frames")
    return frame_to_session_map


def process_arm_coordinates(arm_coords):
    """
    Process arm coordinates to keep only 40cm length from L2/R2 end
    Cut from L1/R1 side and preserve 40cm towards L2/R2

    Args:
        arm_coords: Array of shape (2, 3) with [P1, P2] coordinates

    Returns:
        tuple: (new_p1, p2) where new_p1 is 40cm away from p2 towards p1
    """
    if len(arm_coords) != 2:
        return arm_coords[0], arm_coords[1]

    p1, p2 = arm_coords[0], arm_coords[1]

    # Calculate direction vector from P2 to P1 (reverse direction)
    direction = p1 - p2
    direction_length = np.linalg.norm(direction)

    if direction_length <= 400:  # If 40cm or less, return original
        return p1, p2

    # Normalize direction and place new_p1 at 40cm (400mm) from P2 towards P1
    direction_normalized = direction / direction_length
    new_p1 = p2 + 400 * direction_normalized

    return new_p1, p2


def _to_K3(x: np.ndarray) -> np.ndarray:
    """Force array into shape (K,3)."""
    arr = np.asarray(x)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"Cannot reshape 1D of size {arr.size} to (K,3)")
        return arr.reshape(-1, 3)
    if arr.shape[-1] == 3:
        return arr.reshape(-1, 3)
    raise ValueError(f"Cannot convert array of shape {x.shape} to (K,3)")


def generate_predictions(model, data_loader, device, centroids_dict, target_frame="last"):
    """
    Generate predictions for entire dataset and convert to world coordinates

    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device to run inference on
        centroids_dict: Dictionary mapping frame_id to original centroid
        target_frame: Which frame to use for centroid calculation ("last", "first", etc.)

    Returns:
        predictions_world: List of predicted joint coordinates in world coordinates (N, 15, 3)
        ground_truths_world: List of ground truth joint coordinates in world coordinates (N, 15, 3)
        video_ids: List of video identifiers
    """
    model.eval()

    predictions_world = []
    ground_truths_world = []
    video_ids = []
    print(f"Generating predictions for {len(data_loader.dataset)} samples...")
    print(f"Converting predictions from centroid coordinates to world coordinates using original centroids...")

    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(data_loader, desc="Inference"):
            for clip, target, video_id in zip(batch_clips, batch_targets, batch_video_ids):
                # Add batch dimension
                clip_batch = clip.unsqueeze(0).to(device, non_blocking=True)
                target_batch = target.unsqueeze(0).to(device, non_blocking=True)

                # Forward pass
                output = model(clip_batch).reshape(target_batch.shape)

                # Convert to numpy for processing
                pred_np = output.squeeze(0).cpu().numpy()
                target_np = target.numpy()
                clip_np = clip.numpy()

                # Convert predictions and GT to world coordinates using original centroid
                pred_joints = _to_K3(pred_np)  # Shape: (15, 3) - centroid coordinates
                gt_joints = _to_K3(target_np)   # Shape: (15, 3) - centroid coordinates

                # Extract frame number from video_id to get original centroid
                vid_id_array = video_id.cpu().numpy()
                if isinstance(vid_id_array, np.ndarray):
                    if vid_id_array.size == 1:
                        frame_num = vid_id_array.item()
                    else:
                        # For format [[0, frame_num]], take the second element (frame number)
                        vid_id_flat = vid_id_array.flatten()
                        if len(vid_id_flat) >= 2:
                            frame_num = vid_id_flat[1]  # Take frame number (second element)
                        else:
                            frame_num = vid_id_flat[0] if len(vid_id_flat) > 0 else 0
                else:
                    frame_num = vid_id_array

                # Get original centroid for this frame
                if frame_num in centroids_dict:
                    original_centroid = centroids_dict[frame_num]  # Shape: (3,)
                else:
                    print(f"Warning: Original centroid not found for frame {frame_num}, using zeros")
                    original_centroid = np.zeros(3)

                # Convert to world coordinates by adding back the original centroid
                pred_joints_world = pred_joints + original_centroid.reshape(1, 3)  # Broadcast centroid
                gt_joints_world = gt_joints + original_centroid.reshape(1, 3)

                predictions_world.append(pred_joints_world)
                ground_truths_world.append(gt_joints_world)
                video_ids.append(frame_num)

    return predictions_world, ground_truths_world, video_ids


def create_trajectory_csv(predictions, video_ids, session_map, arm_labels_file, output_csv_file):
    """
    Create CSV file with trajectory data including predictions and processed arm coordinates

    Args:
        predictions: List of predicted joint coordinates (N, 15, 3)
        video_ids: List of video/frame identifiers
        session_map: Mapping from frame_id to (traj_id, frame_in_session)
        arm_labels_file: Path to arm_labels.h5 file
        output_csv_file: Path to output CSV file
    """
    print("Creating CSV file with trajectory data...")

    # Define upper body joint names (0-8)
    joint_names = [
        'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
        'R_Hand', 'L_Hand', 'Torso'
    ]

    # Load arm data
    arm_data = {}
    if os.path.exists(arm_labels_file):
        with h5py.File(arm_labels_file, 'r') as f:
            arm_identifiers = f['id'][:]
            if 'left_arm_coords' in f:
                arm_data['left_arm_coords'] = f['left_arm_coords'][:]
            if 'right_arm_coords' in f:
                arm_data['right_arm_coords'] = f['right_arm_coords'][:]

        # Create mapping from frame_id to arm data
        arm_mapping = {}
        for i, identifier in enumerate(arm_identifiers):
            # Handle both string and numeric identifiers
            if isinstance(identifier, (bytes, np.bytes_)):
                frame_id = identifier.decode('utf-8')
            elif isinstance(identifier, (int, np.integer)):
                frame_id = f"00_{identifier:05d}"
            else:
                frame_id = str(identifier)
            frame_arm_data = {}

            if 'left_arm_coords' in arm_data and i < len(arm_data['left_arm_coords']):
                left_arm = arm_data['left_arm_coords'][i]
                left_l1, left_l2 = process_arm_coordinates(left_arm)
                frame_arm_data.update({'left_l1': left_l1, 'left_l2': left_l2})

            if 'right_arm_coords' in arm_data and i < len(arm_data['right_arm_coords']):
                right_arm = arm_data['right_arm_coords'][i]
                right_r1, right_r2 = process_arm_coordinates(right_arm)
                frame_arm_data.update({'right_r1': right_r1, 'right_r2': right_r2})

            if frame_arm_data:
                arm_mapping[frame_id] = frame_arm_data
    else:
        print(f"Warning: Arm labels file not found: {arm_labels_file}")
        arm_mapping = {}

    # Prepare CSV data
    csv_data = []

    for i, (pred_joints, frame_id) in enumerate(zip(predictions, video_ids)):
        # Convert frame_id to string if it's numeric
        if isinstance(frame_id, (int, np.integer)):
            frame_id_str = f"00_{frame_id:05d}"
        else:
            frame_id_str = str(frame_id)

        # Get trajectory ID and frame position
        if frame_id_str in session_map:
            traj_id, frame_in_session = session_map[frame_id_str]
        else:
            print(f"Warning: Frame {frame_id_str} not found in session map")
            traj_id = -1
            frame_in_session = -1

        row = {
            'traj_id': traj_id,
            'frame_id': frame_id_str,
            'frame_in_session': frame_in_session
        }

        # Add upper body joint predictions (0-8)
        pred_joints_k3 = _to_K3(pred_joints)
        for j, joint_name in enumerate(joint_names):
            if j < len(pred_joints_k3):
                joint_coord = pred_joints_k3[j]
                row[f'{joint_name}_pred_x'] = joint_coord[0]
                row[f'{joint_name}_pred_y'] = joint_coord[1]
                row[f'{joint_name}_pred_z'] = joint_coord[2]
            else:
                row[f'{joint_name}_pred_x'] = np.nan
                row[f'{joint_name}_pred_y'] = np.nan
                row[f'{joint_name}_pred_z'] = np.nan

        # Add processed arm coordinates
        if frame_id_str in arm_mapping:
            arm_info = arm_mapping[frame_id_str]

            # Left arm
            if 'left_l1' in arm_info:
                left_l1 = arm_info['left_l1']
                row['Left_L1_x'] = left_l1[0]
                row['Left_L1_y'] = left_l1[1]
                row['Left_L1_z'] = left_l1[2]
            else:
                row['Left_L1_x'] = np.nan
                row['Left_L1_y'] = np.nan
                row['Left_L1_z'] = np.nan

            if 'left_l2' in arm_info:
                left_l2 = arm_info['left_l2']
                row['Left_L2_x'] = left_l2[0]
                row['Left_L2_y'] = left_l2[1]
                row['Left_L2_z'] = left_l2[2]
            else:
                row['Left_L2_x'] = np.nan
                row['Left_L2_y'] = np.nan
                row['Left_L2_z'] = np.nan

            # Right arm
            if 'right_r1' in arm_info:
                right_r1 = arm_info['right_r1']
                row['Right_R1_x'] = right_r1[0]
                row['Right_R1_y'] = right_r1[1]
                row['Right_R1_z'] = right_r1[2]
            else:
                row['Right_R1_x'] = np.nan
                row['Right_R1_y'] = np.nan
                row['Right_R1_z'] = np.nan

            if 'right_r2' in arm_info:
                right_r2 = arm_info['right_r2']
                row['Right_R2_x'] = right_r2[0]
                row['Right_R2_y'] = right_r2[1]
                row['Right_R2_z'] = right_r2[2]
            else:
                row['Right_R2_x'] = np.nan
                row['Right_R2_y'] = np.nan
                row['Right_R2_z'] = np.nan
        else:
            # No arm data available for this frame
            for arm_side in ['Left', 'Right']:
                for point in ['L1', 'L2'] if arm_side == 'Left' else ['R1', 'R2']:
                    for coord in ['x', 'y', 'z']:
                        row[f'{arm_side}_{point}_{coord}'] = np.nan

        csv_data.append(row)

    # Create DataFrame and clean invalid characters
    df = pd.DataFrame(csv_data)

    # Clean any invalid characters that might cause issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).replace({r'[^\x20-\x7E]': ''}, regex=True)

    df.to_csv(output_csv_file, index=False, float_format='%.2f')

    print(f"CSV file saved to: {output_csv_file}")
    print(f"CSV contains {len(df)} frames with the following columns:")
    print(f"  - Basic info: traj_id, frame_id, frame_in_session")
    print(f"  - Predicted joints (9): {joint_names}")
    print(f"  - Processed arm coordinates: L1/L2 (left), R1/R2 (right)")
    print(f"  - Total columns: {len(df.columns)}")

    # Print trajectory statistics
    traj_stats = df['traj_id'].value_counts().sort_index()
    print(f"  - Trajectory statistics:")
    for traj_id, count in traj_stats.items():
        if traj_id >= 0:  # Valid trajectory
            print(f"    Traj {traj_id}: {count} frames")

    return df


def create_trajectory_csv_gt(ground_truths, video_ids, session_map, arm_labels_file, output_csv_file):
    """
    Create CSV file with trajectory data including ground truth joint coordinates and processed arm coordinates

    Args:
        ground_truths: List of ground truth joint coordinates (N, 15, 3)
        video_ids: List of video/frame identifiers
        session_map: Mapping from frame_id to (traj_id, frame_in_session)
        arm_labels_file: Path to arm_labels.h5 file
        output_csv_file: Path to output CSV file
    """
    print("Creating CSV file with ground truth trajectory data...")

    # Define all joint names (0-14)
    joint_names = [
        'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
        'R_Hand', 'L_Hand', 'Torso', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee',
        'R_Foot', 'L_Foot'
    ]

    # Load arm data
    arm_data = {}
    if os.path.exists(arm_labels_file):
        with h5py.File(arm_labels_file, 'r') as f:
            arm_identifiers = f['id'][:]
            if 'left_arm_coords' in f:
                arm_data['left_arm_coords'] = f['left_arm_coords'][:]
            if 'right_arm_coords' in f:
                arm_data['right_arm_coords'] = f['right_arm_coords'][:]

        # Create mapping from frame_id to arm data
        arm_mapping = {}
        for i, identifier in enumerate(arm_identifiers):
            # Handle both string and numeric identifiers
            if isinstance(identifier, (bytes, np.bytes_)):
                frame_id = identifier.decode('utf-8')
            elif isinstance(identifier, (int, np.integer)):
                frame_id = f"00_{identifier:05d}"
            else:
                frame_id = str(identifier)
            frame_arm_data = {}

            if 'left_arm_coords' in arm_data and i < len(arm_data['left_arm_coords']):
                left_arm = arm_data['left_arm_coords'][i]
                left_l1, left_l2 = process_arm_coordinates(left_arm)
                frame_arm_data.update({'left_l1': left_l1, 'left_l2': left_l2})

            if 'right_arm_coords' in arm_data and i < len(arm_data['right_arm_coords']):
                right_arm = arm_data['right_arm_coords'][i]
                right_r1, right_r2 = process_arm_coordinates(right_arm)
                frame_arm_data.update({'right_r1': right_r1, 'right_r2': right_r2})

            if frame_arm_data:
                arm_mapping[frame_id] = frame_arm_data
    else:
        print(f"Warning: Arm labels file not found: {arm_labels_file}")
        arm_mapping = {}

    # Prepare CSV data
    csv_data = []

    for i, (gt_joints, frame_id) in enumerate(zip(ground_truths, video_ids)):
        # Convert frame_id to string if it's numeric
        if isinstance(frame_id, (int, np.integer)):
            frame_id_str = f"00_{frame_id:05d}"
        else:
            frame_id_str = str(frame_id)

        # Get trajectory ID and frame position
        if frame_id_str in session_map:
            traj_id, frame_in_session = session_map[frame_id_str]
        else:
            print(f"Warning: Frame {frame_id_str} not found in session map")
            traj_id = -1
            frame_in_session = -1

        row = {
            'traj_id': traj_id,
            'frame_id': frame_id_str,
            'frame_in_session': frame_in_session
        }

        # Add all joint ground truth values (0-14)
        gt_joints_k3 = _to_K3(gt_joints)
        for j, joint_name in enumerate(joint_names):
            if j < len(gt_joints_k3):
                joint_coord = gt_joints_k3[j]
                row[f'{joint_name}_gt_x'] = joint_coord[0]
                row[f'{joint_name}_gt_y'] = joint_coord[1]
                row[f'{joint_name}_gt_z'] = joint_coord[2]
            else:
                row[f'{joint_name}_gt_x'] = np.nan
                row[f'{joint_name}_gt_y'] = np.nan
                row[f'{joint_name}_gt_z'] = np.nan

        # Add processed arm coordinates
        if frame_id_str in arm_mapping:
            arm_info = arm_mapping[frame_id_str]

            # Left arm
            if 'left_l1' in arm_info:
                left_l1 = arm_info['left_l1']
                row['Left_L1_x'] = left_l1[0]
                row['Left_L1_y'] = left_l1[1]
                row['Left_L1_z'] = left_l1[2]
            else:
                row['Left_L1_x'] = np.nan
                row['Left_L1_y'] = np.nan
                row['Left_L1_z'] = np.nan

            if 'left_l2' in arm_info:
                left_l2 = arm_info['left_l2']
                row['Left_L2_x'] = left_l2[0]
                row['Left_L2_y'] = left_l2[1]
                row['Left_L2_z'] = left_l2[2]
            else:
                row['Left_L2_x'] = np.nan
                row['Left_L2_y'] = np.nan
                row['Left_L2_z'] = np.nan

            # Right arm
            if 'right_r1' in arm_info:
                right_r1 = arm_info['right_r1']
                row['Right_R1_x'] = right_r1[0]
                row['Right_R1_y'] = right_r1[1]
                row['Right_R1_z'] = right_r1[2]
            else:
                row['Right_R1_x'] = np.nan
                row['Right_R1_y'] = np.nan
                row['Right_R1_z'] = np.nan

            if 'right_r2' in arm_info:
                right_r2 = arm_info['right_r2']
                row['Right_R2_x'] = right_r2[0]
                row['Right_R2_y'] = right_r2[1]
                row['Right_R2_z'] = right_r2[2]
            else:
                row['Right_R2_x'] = np.nan
                row['Right_R2_y'] = np.nan
                row['Right_R2_z'] = np.nan
        else:
            # No arm data available for this frame
            for arm_side in ['Left', 'Right']:
                for point in ['L1', 'L2'] if arm_side == 'Left' else ['R1', 'R2']:
                    for coord in ['x', 'y', 'z']:
                        row[f'{arm_side}_{point}_{coord}'] = np.nan

        csv_data.append(row)

    # Create DataFrame and clean invalid characters
    df = pd.DataFrame(csv_data)

    # Clean any invalid characters that might cause issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).replace({r'[^\x20-\x7E]': ''}, regex=True)

    df.to_csv(output_csv_file, index=False, float_format='%.2f')

    print(f"Ground truth CSV file saved to: {output_csv_file}")
    print(f"CSV contains {len(df)} frames with the following columns:")
    print(f"  - Basic info: traj_id, frame_id, frame_in_session")
    print(f"  - Ground truth joints (15): {joint_names}")
    print(f"  - Processed arm coordinates: L1/L2 (left), R1/R2 (right)")
    print(f"  - Total columns: {len(df.columns)}")

    # Print trajectory statistics
    traj_stats = df['traj_id'].value_counts().sort_index()
    print(f"  - Trajectory statistics:")
    for traj_id, count in traj_stats.items():
        if traj_id >= 0:  # Valid trajectory
            print(f"    Traj {traj_id}: {count} frames")

    return df


def save_predictions_h5(predictions, ground_truths, video_ids, output_file):
    """
    Save predictions to H5 file in same format as original labels

    Args:
        predictions: List of predicted joint coordinates
        ground_truths: List of ground truth joint coordinates
        video_ids: List of video identifiers
        output_file: Output H5 file path
    """
    print(f"Saving {len(predictions)} predictions to {output_file}")

    # Convert lists to numpy arrays
    pred_array = np.array(predictions, dtype=np.float64)  # Shape: (N, 15, 3)
    gt_array = np.array(ground_truths, dtype=np.float64)   # Shape: (N, 15, 3)

    # Convert video_ids to string format matching original training data
    ids_list = []
    for vid_id in video_ids:
        # Ensure we have a scalar value
        if isinstance(vid_id, np.ndarray):
            vid_id = vid_id.item()

        if isinstance(vid_id, (int, np.integer)):
            # Convert integer ID to string format like "00_XXXXX"
            ids_list.append(f"00_{int(vid_id):05d}")
        elif isinstance(vid_id, str):
            # If already string, use as-is
            ids_list.append(vid_id)
        else:
            # Convert other types to string with proper format
            ids_list.append(f"00_{int(vid_id):05d}")

    print(f"Sample IDs: {ids_list[:5]}...")  # Debug: show first 5 IDs

    # Create is_valid array (assume all predictions are valid)
    is_valid_array = np.ones(len(predictions), dtype=bool)

    print(f"Prediction array shape: {pred_array.shape}")
    print(f"Ground truth array shape: {gt_array.shape}")
    print(f"IDs array shape: {len(ids_list)}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to H5 file in original format
    with h5py.File(output_file, 'w') as f:
        # Match original format exactly
        f.create_dataset('id', data=ids_list)
        f.create_dataset('is_valid', data=is_valid_array)
        f.create_dataset('real_world_coordinates', data=gt_array)

        # Add prediction data as additional dataset
        f.create_dataset('predicted_coordinates', data=pred_array)

        print(f"Saved datasets (converted to world coordinates):")
        print(f"  - id: {len(ids_list)} (object)")
        print(f"  - is_valid: {is_valid_array.shape} (bool)")
        print(f"  - real_world_coordinates: {gt_array.shape} (float64) [world coords]")
        print(f"  - predicted_coordinates: {pred_array.shape} (float64) [world coords]")


def analyze_predictions(predictions, ground_truths):
    """
    Analyze prediction statistics
    """
    pred_array = np.array(predictions)
    gt_array = np.array(ground_truths)

    # Calculate per-joint errors
    errors = np.linalg.norm(pred_array - gt_array, axis=2)  # Shape: (N, 15)
    mean_errors = errors.mean(axis=0)  # Shape: (15,)
    std_errors = errors.std(axis=0)    # Shape: (15,)

    print(f"\nPrediction Analysis:")
    print(f"Dataset size: {len(predictions)} samples")
    print(f"Prediction range: [{pred_array.min():.2f}, {pred_array.max():.2f}]")
    print(f"Ground truth range: [{gt_array.min():.2f}, {gt_array.max():.2f}]")
    print(f"Overall mean error: {errors.mean():.2f} mm")
    print(f"Overall std error: {errors.std():.2f} mm")

    print(f"\nPer-joint errors (mean | std):")
    from const import skeleton_joints
    for i in range(15):
        joint_name = skeleton_joints.joint_indices.get(i, f"Joint{i}")
        print(f"  {i:2d} {joint_name:12s}: {mean_errors[i]:6.2f} | {std_errors[i]:6.2f} mm")


def main(arguments):
    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device(0)

    set_random_seed(config["seed"])

    # Load test data
    print(f"Loading test data...")
    data_loader_test, num_coord_joints = load_data(config, mode="test")
    print(f"Test dataset size: {len(data_loader_test.dataset)}")

    # Load model
    print(f"Loading model from {arguments.model}")
    model = model_builder.create_model(config, num_coord_joints)
    model.to(device)

    checkpoint = torch.load(arguments.model, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"Model loaded successfully")

    # Load original centroids for world coordinate conversion
    data_output_path = config.get('data_output_path', config.get('experiments_path', '.'))
    train_dir = os.path.join(data_output_path, 'train')
    train_labels_file = os.path.join(data_output_path, 'train_labels.h5')
    centroids_dict = load_original_centroids(train_dir, train_labels_file)

    # Generate predictions (convert to world coordinates)
    target_frame = str(config.get("target_frame", "last"))
    predictions, ground_truths, video_ids = generate_predictions(model, data_loader_test, device, centroids_dict, target_frame)

    # Analyze predictions
    analyze_predictions(predictions, ground_truths)

    # Save to H5 file
    output_file = arguments.output
    save_predictions_h5(predictions, ground_truths, video_ids, output_file)

    # Load session information and create CSV files
    itop_format_dir = config.get('dataset_path', '')  # Points to itop_format directory
    arm_labels_file = os.path.join(data_output_path, 'arm_labels.h5')
    csv_output_file = output_file.replace('.h5', '_trajectory.csv')
    csv_gt_output_file = output_file.replace('.h5', '_trajectory_gt.csv')

    session_map = load_session_info(itop_format_dir)

    # Create CSV with predicted joint coordinates
    create_trajectory_csv(predictions, video_ids, session_map, arm_labels_file, csv_output_file)

    # Create CSV with ground truth joint coordinates
    create_trajectory_csv_gt(ground_truths, video_ids, session_map, arm_labels_file, csv_gt_output_file)

    print(f"\nInference complete!")
    print(f"Predictions saved to: {output_file}")
    print(f"CSV trajectory data (predictions) saved to: {csv_output_file}")
    print(f"CSV trajectory data (ground truth) saved to: {csv_gt_output_file}")

    # Verify saved file
    print(f"\nVerifying saved file...")
    with h5py.File(output_file, 'r') as f:
        print(f"File contents:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape} {f[key].dtype}")
        print(f"Attributes: {dict(f.attrs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predicted poses using trained SPiKE model")
    parser.add_argument("--config", type=str, default="experiments/Custom/1",
                        help="Path to the YAML config directory or file")

    args = parser.parse_args()

    # Load config and set all paths from config
    config = load_config(args.config)

    # Set paths from config
    args.model = os.path.join(config.get("output_dir", "."), "best_model.pth")
    args.output = os.path.join(config.get("output_dir", "."), "inference_labels.h5")

    main(args)