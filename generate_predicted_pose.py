#!/usr/bin/env python3
"""
Run inference on ITOP test set and export trajectories to CSV (world coordinates).
- Loads trained model and test loader
- Restores world coordinates via original point-cloud centroids
- Writes two CSVs: predictions (upper-body 9 joints) and GT (all 15 joints)
- Optionally merges 40 cm truncated arm endpoints from arm_labels.h5 if present
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import torch

from model import model_builder
from trainer_itop import load_data
from utils.config_utils import load_config, set_random_seed


# ----------------------------- Utilities -----------------------------
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


def load_original_centroids(train_dir, train_labels_file):
    """
    Read training IDs and compute each frame's true centroid from original train/*.npz.
    Returns: dict[int frame_num] -> (3,)
    """
    print("Loading original centroids...")
    with h5py.File(train_labels_file, 'r') as f:
        identifiers = f['id'][:]

    centroids = {}
    for identifier in tqdm(identifiers, desc="Centroids"):
        if isinstance(identifier, (bytes, np.bytes_)):
            frame_id_str = identifier.decode('utf-8')
        elif isinstance(identifier, (int, np.integer)):
            frame_id_str = f"00_{identifier:05d}"
        else:
            frame_id_str = str(identifier)
        frame_num = int(frame_id_str.split('_')[-1])

        pc_file = os.path.join(train_dir, f"{frame_num}.npz")
        if os.path.exists(pc_file):
            pc = np.load(pc_file)['arr_0']
            centroids[frame_num] = np.mean(pc, axis=0)
        else:
            centroids[frame_num] = np.zeros(3)  # fallback with warning is noisy; keep quiet here
    return centroids


def load_session_info(labels_file):
    """
    Build frame_id -> (traj_id_str, frame_in_session) mapping from traj_id field in labels file.
    Also returns frame_num -> original_frame_id mapping.
    """
    print("Loading session map...")
    with h5py.File(labels_file, 'r') as f:
        identifiers = f['id'][:]
        if 'traj_id' not in f:
            print("Warning: 'traj_id' field not found in labels file. All traj_id will be -1.")
            return {}, {}
        traj_ids = f['traj_id'][:]

    # Decode identifiers and traj_ids
    identifiers = [id.decode('utf-8') if isinstance(id, bytes) else str(id) for id in identifiers]
    traj_ids_decoded = [tid.decode('utf-8') if isinstance(tid, bytes) else str(tid) for tid in traj_ids]

    # Track frame_in_session counter for each traj_id
    session_frame_counters = {}
    frame_to_session_map = {}
    frame_num_to_id_map = {}

    for idx, (frame_id, traj_id_str) in enumerate(zip(identifiers, traj_ids_decoded)):
        if traj_id_str not in session_frame_counters:
            session_frame_counters[traj_id_str] = 0

        frame_in_session = session_frame_counters[traj_id_str]
        frame_to_session_map[frame_id] = (traj_id_str, frame_in_session)
        session_frame_counters[traj_id_str] += 1

        # Extract frame number from frame_id (e.g., "00_00123" -> 123)
        frame_num = int(frame_id.split('_')[-1])
        frame_num_to_id_map[frame_num] = frame_id

    print(f"Found {len(set(traj_ids_decoded))} unique trajectories")
    print(f"Loaded {len(frame_num_to_id_map)} frame ID mappings")
    return frame_to_session_map, frame_num_to_id_map


def process_arm_coordinates(arm_coords):
    """
    Keep 40cm length from second point (P2=L2/R2) toward first (P1=L1/R1).
    arm_coords: (2,3) -> returns (new_p1, p2)
    """
    if len(arm_coords) != 2:
        return arm_coords[0], arm_coords[1]
    p1, p2 = arm_coords[0], arm_coords[1]
    v = p1 - p2
    d = np.linalg.norm(v)
    if d <= 400:
        return p1, p2
    new_p1 = p2 + 400 * (v / d)
    return new_p1, p2


# ----------------------------- Core steps -----------------------------
def generate_predictions(model, data_loader, device, centroids_dict):
    """
    Run inference; convert pred/gt from centroid-coords to world coords by adding original centroid.
    Returns: predictions_world, ground_truths_world, video_ids(list of frame_num)
    """
    model.eval()
    preds_w, gts_w, vids = [], [], []
    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(data_loader, desc="Inference"):
            for clip, target, video_id in zip(batch_clips, batch_targets, batch_video_ids):
                clip_batch = clip.unsqueeze(0).to(device, non_blocking=True)
                output = model(clip_batch).reshape(target.unsqueeze(0).shape)

                pred_np = output.squeeze(0).cpu().numpy()
                gt_np = target.numpy()

                # frame number extraction (supports [[0, f]], [f], or scalar)
                vid_id_array = batch_video_ids.new_tensor(video_id).cpu().numpy() if hasattr(batch_video_ids, 'new_tensor') else np.array(video_id)
                if isinstance(vid_id_array, np.ndarray):
                    vid_flat = vid_id_array.flatten()
                    frame_num = int(vid_flat[1] if len(vid_flat) >= 2 else vid_flat[0])
                else:
                    frame_num = int(vid_id_array)

                centroid = centroids_dict.get(frame_num, np.zeros(3))
                pred_w = _to_K3(pred_np) + centroid.reshape(1, 3)
                gt_w = _to_K3(gt_np) + centroid.reshape(1, 3)

                preds_w.append(pred_w)
                gts_w.append(gt_w)
                vids.append(frame_num)
    return preds_w, gts_w, vids


def _load_arm_mapping(arm_labels_file):
    """
    Returns: dict[str frame_id] -> {'left_l1','left_l2','right_r1','right_r2'} (each (3,))
    If arm file missing, returns {}.
    """
    if not os.path.exists(arm_labels_file):
        return {}

    arm_mapping = {}
    with h5py.File(arm_labels_file, 'r') as f:
        arm_identifiers = f['id'][:]
        left_arr = f['left_arm_coords'][:] if 'left_arm_coords' in f else None
        right_arr = f['right_arm_coords'][:] if 'right_arm_coords' in f else None

    for i, identifier in enumerate(arm_identifiers):
        if isinstance(identifier, (bytes, np.bytes_)):
            frame_id = identifier.decode('utf-8')
        elif isinstance(identifier, (int, np.integer)):
            frame_id = f"00_{identifier:05d}"
        else:
            frame_id = str(identifier)

        entry = {}
        if left_arr is not None and i < len(left_arr):
            l1, l2 = process_arm_coordinates(left_arr[i])
            entry['left_l1'], entry['left_l2'] = l1, l2
        if right_arr is not None and i < len(right_arr):
            r1, r2 = process_arm_coordinates(right_arr[i])
            entry['right_r1'], entry['right_r2'] = r1, r2
        if entry:
            arm_mapping[frame_id] = entry
    return arm_mapping


def create_trajectory_csv(predictions, video_ids, session_map, frame_num_to_id_map, arm_labels_file, output_csv_file):
    """
    CSV with predicted joints (upper body 9 joints) + (optional) processed arm endpoints.
    """
    joint_names = [
        'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
        'R_Hand', 'L_Hand', 'Torso'
    ]
    arm_mapping = _load_arm_mapping(arm_labels_file)

    rows = []
    for pred_joints, frame_num in zip(predictions, video_ids):
        # Get original frame_id from labels file
        frame_id_str = frame_num_to_id_map.get(int(frame_num), f"UNKNOWN_{int(frame_num):05d}")
        traj_id_str, frame_in_session = session_map.get(frame_id_str, ("UNKNOWN", -1))

        row = {'traj_id': traj_id_str, 'frame_id': frame_id_str, 'frame_in_session': frame_in_session}
        pj = _to_K3(pred_joints)
        for j, name in enumerate(joint_names):
            if j < len(pj):
                row[f'{name}_pred_x'], row[f'{name}_pred_y'], row[f'{name}_pred_z'] = pj[j]
            else:
                row[f'{name}_pred_x'] = row[f'{name}_pred_y'] = row[f'{name}_pred_z'] = np.nan

        arm = arm_mapping.get(frame_id_str, {})
        # left
        ll1 = arm.get('left_l1'); ll2 = arm.get('left_l2')
        row['Left_L1_x'], row['Left_L1_y'], row['Left_L1_z'] = (ll1 if ll1 is not None else (np.nan, np.nan, np.nan))
        row['Left_L2_x'], row['Left_L2_y'], row['Left_L2_z'] = (ll2 if ll2 is not None else (np.nan, np.nan, np.nan))
        # right
        rr1 = arm.get('right_r1'); rr2 = arm.get('right_r2')
        row['Right_R1_x'], row['Right_R1_y'], row['Right_R1_z'] = (rr1 if rr1 is not None else (np.nan, np.nan, np.nan))
        row['Right_R2_x'], row['Right_R2_y'], row['Right_R2_z'] = (rr2 if rr2 is not None else (np.nan, np.nan, np.nan))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_file, index=False, float_format='%.2f')
    print(f"CSV (pred) saved: {output_csv_file} | frames: {len(df)}")


def create_trajectory_csv_gt(ground_truths, video_ids, session_map, frame_num_to_id_map, arm_labels_file, output_csv_file):
    """
    CSV with ground-truth joints (all 15) + (optional) processed arm endpoints.
    """
    joint_names = [
        'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
        'R_Hand', 'L_Hand', 'Torso', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee',
        'R_Foot', 'L_Foot'
    ]
    arm_mapping = _load_arm_mapping(arm_labels_file)

    rows = []
    for gt_joints, frame_num in zip(ground_truths, video_ids):
        # Get original frame_id from labels file
        frame_id_str = frame_num_to_id_map.get(int(frame_num), f"UNKNOWN_{int(frame_num):05d}")
        traj_id_str, frame_in_session = session_map.get(frame_id_str, ("UNKNOWN", -1))

        row = {'traj_id': traj_id_str, 'frame_id': frame_id_str, 'frame_in_session': frame_in_session}
        gj = _to_K3(gt_joints)
        for j, name in enumerate(joint_names):
            if j < len(gj):
                row[f'{name}_gt_x'], row[f'{name}_gt_y'], row[f'{name}_gt_z'] = gj[j]
            else:
                row[f'{name}_gt_x'] = row[f'{name}_gt_y'] = row[f'{name}_gt_z'] = np.nan

        arm = arm_mapping.get(frame_id_str, {})
        ll1 = arm.get('left_l1'); ll2 = arm.get('left_l2')
        row['Left_L1_x'], row['Left_L1_y'], row['Left_L1_z'] = (ll1 if ll1 is not None else (np.nan, np.nan, np.nan))
        row['Left_L2_x'], row['Left_L2_y'], row['Left_L2_z'] = (ll2 if ll2 is not None else (np.nan, np.nan, np.nan))
        rr1 = arm.get('right_r1'); rr2 = arm.get('right_r2')
        row['Right_R1_x'], row['Right_R1_y'], row['Right_R1_z'] = (rr1 if rr1 is not None else (np.nan, np.nan, np.nan))
        row['Right_R2_x'], row['Right_R2_y'], row['Right_R2_z'] = (rr2 if rr2 is not None else (np.nan, np.nan, np.nan))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_file, index=False, float_format='%.2f')
    print(f"CSV (gt) saved: {output_csv_file} | frames: {len(df)}")


# ----------------------------- Main -----------------------------
def main(arguments):
    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("device_args", 0))
    device = torch.device(0)
    set_random_seed(config.get("seed", 0))

    # Determine mode (train or test)
    mode = arguments.mode
    print(f"Running inference on {mode} data...")

    # data - create DataLoader without shuffle for inference
    from datasets.itop import ITOP

    dataset_params = {
        "root": config.get("data_output_path", config["dataset_path"]),
        "frames_per_clip": config["frames_per_clip"],
        "num_points": config["num_points"],
        "use_valid_only": config["use_valid_only"],
        "target_frame": config["target_frame"],
    }

    # Create dataset based on mode
    # IMPORTANT: Always use TEST augmentation (no rotation/mirror) for inference to get consistent results
    is_train = (mode == "train")
    aug_list = config["PREPROCESS_TEST"]  # Use test preprocessing for both train and test
    dataset = ITOP(train=is_train, aug_list=aug_list, **dataset_params)

    # Create DataLoader WITHOUT shuffle for inference (to preserve frame order)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # Important: no shuffle for inference
        num_workers=config["workers"],
        pin_memory=True
    )

    num_coord_joints = dataset.num_coord_joints
    print(f"{mode.capitalize()} size: {len(data_loader.dataset)}")

    # model
    print(f"Loading model: {arguments.model}")
    model = model_builder.create_model(config, num_coord_joints).to(device)
    checkpoint = torch.load(arguments.model, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    print("Model loaded.")

    # centroids for world restore
    data_output_path = config.get('data_output_path', config.get('experiments_path', '.'))
    data_dir = os.path.join(data_output_path, mode)
    labels_file = os.path.join(data_output_path, f'{mode}_labels.h5')
    centroids_dict = load_original_centroids(data_dir, labels_file)

    # inference (world coords)
    predictions, ground_truths, video_ids = generate_predictions(model, data_loader, device, centroids_dict)

    # session map & CSV export
    arm_labels_file = os.path.join(data_output_path, f'{mode}_arm_labels.h5')
    session_map, frame_num_to_id_map = load_session_info(labels_file)

    out_pred_csv = os.path.join(config.get("output_dir", "."), f"inference_trajectory_{mode}.csv")
    out_gt_csv = os.path.join(config.get("output_dir", "."), f"inference_trajectory_gt_{mode}.csv")

    create_trajectory_csv(predictions, video_ids, session_map, frame_num_to_id_map, arm_labels_file, out_pred_csv)
    create_trajectory_csv_gt(ground_truths, video_ids, session_map, frame_num_to_id_map, arm_labels_file, out_gt_csv)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITOP inference → CSV (world coords)")
    parser.add_argument("--config", type=str, default="experiments/Custom/pretrained-full",
                        help="Path to the YAML config directory or file")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"],
                        help="Run inference on train or test data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.model = os.path.join(cfg.get("output_dir", "."), "best_model.pth")
    main(args)
