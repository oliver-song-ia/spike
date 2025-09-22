"""
Module for testing SPiKE model on consecutive .npz files with preprocessing and coordinate transformation.
Only shows point cloud and prediction (no ground truth).

Usage:
  python single_infer.py \
    --config experiments/ITOP-SIDE/1 \
    --model  experiments/ITOP-SIDE/1/log/best_model.pth \
    --data-folder path/to/npz_dir \
    --start-frame 0
"""

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from model import model_builder
from trainer_itop import load_data
from utils.config_utils import load_config, set_random_seed
from augmentations.aug_pipeline import AugPipeline
import matplotlib.pyplot as plt


def preprocess_point_cloud(point_cloud, target_points=7000):
    """Preprocessing: (1) optional downsample, (2) center and scale, (3) cast to float16."""
    print(f"Original point cloud shape: {point_cloud.shape}")

    # Step 1: Downsample
    if len(point_cloud) > target_points:
        idx = np.random.choice(len(point_cloud), target_points, replace=False)
        point_cloud = point_cloud[idx]
        print(f"Downsampled to: {point_cloud.shape}")

    # Step 2: Center to origin
    current_center = point_cloud.mean(axis=0)
    print(f"Centering from: [{current_center[0]:.3f}, {current_center[1]:.3f}, {current_center[2]:.3f}]")
    point_cloud_centered = point_cloud - current_center

    # Step 3: Scale to target range (uniform scaling to preserve aspect ratio)
    target_ranges = np.array([1.080, 2.129, 0.883])  # ITOP target ranges
    current_ranges = point_cloud_centered.max(axis=0) - point_cloud_centered.min(axis=0)
    scale_factor = min(target_ranges / np.maximum(current_ranges, 1e-6))
    print(f"Uniform scale factor: {scale_factor:.3f} (preserves aspect ratio)")

    point_cloud_scaled = point_cloud_centered * scale_factor

    # Step 4: Translate to target center
    target_center = np.array([0.013, -0.280, 2.953])
    point_cloud_final = point_cloud_scaled + target_center
    print(f"Final center: [{point_cloud_final.mean(axis=0)[0]:.3f}, {point_cloud_final.mean(axis=0)[1]:.3f}, {point_cloud_final.mean(axis=0)[2]:.3f}]")

    return point_cloud_final.astype(np.float16)


def _extract_pointcloud_from_clip(clip_np: np.ndarray, target_frame: str = "last") -> np.ndarray:
    """Extract single frame (N,3) from (T,N,3)/(1,T,N,3)/(N,3) arrays."""
    arr = np.asarray(clip_np)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        if target_frame == "first":
            return arr[0]
        if target_frame == "last":
            return arr[-1]
        try:
            return arr[int(target_frame)]
        except Exception:
            return arr[-1]
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr
    if arr.shape[-1] == 3:
        return arr.reshape(-1, 3)
    raise ValueError(f"Cannot parse clip to (N,3), got shape {clip_np.shape}")


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


def _equal_axis_3d(ax, pts: np.ndarray):
    """Equalize axis limits for 3D plot."""
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2.0
    r = (maxs - mins).max() * 0.55
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def visualize_pc_and_prediction(pc, pred, title="Point Cloud and Prediction"):
    """Scatter the point cloud and predicted keypoints with joint names."""
    from const import skeleton_joints

    print(f"\nVisualization debug:")
    print(f"  Point cloud range: X[{pc[:,0].min():.1f}, {pc[:,0].max():.1f}], "
          f"Y[{pc[:,1].min():.1f}, {pc[:,1].max():.1f}], Z[{pc[:,2].min():.1f}, {pc[:,2].max():.1f}]")
    print(f"  Prediction range: X[{pred[:,0].min():.1f}, {pred[:,0].max():.1f}], "
          f"Y[{pred[:,1].min():.1f}, {pred[:,1].max():.1f}], Z[{pred[:,2].min():.1f}, {pred[:,2].max():.1f}]")
    print(f"  Scale ratio: PC={pc.max()-pc.min():.1f}mm, Pred={pred.max()-pred.min():.1f}mm")

    # Convert point cloud from meters to millimeters for proper scale matching
    pc_mm = pc * 1000
    print(f"  Point cloud (mm): X[{pc_mm[:,0].min():.1f}, {pc_mm[:,0].max():.1f}], "
          f"Y[{pc_mm[:,1].min():.1f}, {pc_mm[:,1].max():.1f}], Z[{pc_mm[:,2].min():.1f}, {pc_mm[:,2].max():.1f}]")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use millimeter-scale point cloud for visualization
    pc_vis = pc_mm if pc_mm.shape[0] <= 20000 else pc_mm[np.random.choice(pc_mm.shape[0], 20000, replace=False)]
    ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
               s=1.0, alpha=0.6, depthshade=False, label="Point Cloud", c='lightblue')
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
               s=100, marker='o', label='Prediction', edgecolors='black', color='red', linewidth=2)

    # Add joint names as text labels
    for i, (x, y, z) in enumerate(pred):
        joint_name = skeleton_joints.joint_indices.get(i, f"Joint{i}")
        ax.text(x, y, z, f'{i}:{joint_name}', fontsize=9, alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Draw skeleton connections
    for j1, j2, color in skeleton_joints.joint_connections:
        if j1 < len(pred) and j2 < len(pred):
            ax.plot([pred[j1, 0], pred[j2, 0]],
                   [pred[j1, 1], pred[j2, 1]],
                   [pred[j1, 2], pred[j2, 2]],
                   color=color, linewidth=2, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.legend()

    all_pts = np.vstack([pc_vis, pred])
    _equal_axis_3d(ax, all_pts)
    plt.tight_layout(); plt.show()


def load_and_preprocess_npz(npz_path):
    """Load one .npz and apply basic preprocessing (downsample + float16) but no coordinate transform."""
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    pc = data['arr_0']
    print(f"Loaded shape: {pc.shape}, dtype: {pc.dtype}")

    # Check if already preprocessed (small size + float16 + in ITOP range)
    if (pc.dtype == np.float16) and (pc.shape[0] <= 7000) and (pc[:, 2].mean() > 2.0):
        print("✅ Already preprocessed; using directly.")
        return pc

    print("⚠️  Applying preprocessing (downsample + center + scale + float16)...")
    return preprocess_point_cloud(pc)


def load_consecutive_point_clouds(data_folder, start_frame=0, frames_per_clip=3):
    """Load `frames_per_clip` consecutive .npz files from a folder, sorted numerically."""
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    print(f"Loading {frames_per_clip} consecutive frames from {data_folder}")
    npz_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]
    npz_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x)

    if len(npz_files) < start_frame + frames_per_clip:
        raise ValueError(f"Not enough files. Found {len(npz_files)}, need >= {start_frame + frames_per_clip}")

    print(f"Found {len(npz_files)} .npz files; using frames {start_frame}..{start_frame + frames_per_clip - 1}")
    pcs = []
    for i in range(start_frame, start_frame + frames_per_clip):
        path = os.path.join(data_folder, npz_files[i])
        print(f"Frame {i}: {npz_files[i]}")
        pcs.append(load_and_preprocess_npz(path))
    return pcs


def _random_sample_pc(p, num_points):
    """Sample to `num_points` with the same policy used by the ITOP loader."""
    if p.shape[0] > num_points:
        r = np.random.choice(p.shape[0], size=num_points, replace=False)
    elif p.shape[0] < num_points:
        repeat, residue = divmod(num_points, p.shape[0])
        r = np.concatenate([np.arange(p.shape[0])] * repeat +
                           [np.random.choice(p.shape[0], size=residue, replace=False)], axis=0)
    else:
        return p
    return p[r, :]


def create_mock_dataset_sample_with_augmentation(input_pcs, config):
    """
    Build a clip tensor (T,N,3) from a list of preprocessed PCs, then apply test-time augmentation.
    `input_pcs` must be a list with length == frames_per_clip.
    """
    frames_per_clip = config["frames_per_clip"]
    num_points = config["num_points"]

    if not isinstance(input_pcs, list) or len(input_pcs) != frames_per_clip:
        raise ValueError(f"Expected {frames_per_clip} point clouds, got {len(input_pcs) if isinstance(input_pcs, list) else 'non-list'}")

    sampled = []
    for i, pc in enumerate(input_pcs):
        spc = _random_sample_pc(pc, num_points)
        sampled.append(spc)
        print(f"Frame {i}: sampled to {num_points} points")

    clip = np.stack(sampled, axis=0)           # (T,N,3)
    clip_tensor = torch.FloatTensor(clip)      # torch (T,N,3)

    # Test-time augmentation (same as dataset test)
    aug = AugPipeline()
    aug.create_pipeline(config["PREPROCESS_TEST"])
    dummy_joints = torch.zeros(1, 15, 3).float()
    print("Applying test-time augmentation...")
    clip_aug, _, _ = aug.augment(clip_tensor, dummy_joints)
    return clip_aug


def main(arguments):
    config = load_config(arguments.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device(0)

    set_random_seed(config["seed"])

    if not arguments.data_folder:
        raise ValueError("--data-folder is required (single-file mode has been removed).")

    print(f"Loading model from {arguments.model}")
    # Use loader to fetch num_coord_joints (matches your existing setup)
    _, num_coord_joints = load_data(config, mode="test")
    model = model_builder.create_model(config, num_coord_joints).to(device)

    ckpt = torch.load(arguments.model, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Build a clip from consecutive frames
    frames_per_clip = config["frames_per_clip"]
    pcs = load_consecutive_point_clouds(arguments.data_folder, arguments.start_frame, frames_per_clip)

    print("\nLoaded consecutive point clouds summary:")
    for i, pc in enumerate(pcs):
        print(f"  Frame {i}: shape={pc.shape} "
              f"X[{pc[:,0].min():.3f},{pc[:,0].max():.3f}] "
              f"Y[{pc[:,1].min():.3f},{pc[:,1].max():.3f}] "
              f"Z[{pc[:,2].min():.3f},{pc[:,2].max():.3f}]")

    clip = create_mock_dataset_sample_with_augmentation(pcs, config)

    # Inference
    print("\nRunning inference...")
    target_frame = str(config.get("target_frame", "last"))
    with torch.no_grad():
        clip_t = clip.unsqueeze(0).to(device, non_blocking=True)  # (1,T,N,3 or model-specific)
        output = model(clip_t).cpu().numpy()

        clip_np = clip.numpy()
        pred_np = output
        pc = _extract_pointcloud_from_clip(clip_np, target_frame=target_frame)
        pred = _to_K3(pred_np)

        print("\nInference results:")
        print(f"  Point cloud shape: {pc.shape}")
        print(f"  Prediction shape : {pred.shape}")
        print(f"  Prediction center: [{pred.mean(0)[0]:.3f}, {pred.mean(0)[1]:.3f}, {pred.mean(0)[2]:.3f}]")

        title = f"SPiKE Inference on {os.path.basename(arguments.data_folder)} " \
                f"(frames {arguments.start_frame}-{arguments.start_frame + frames_per_clip - 1})"
        visualize_pc_and_prediction(pc, pred, title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE inference on consecutive .npz files (no single-file mode)")
    parser.add_argument("--config", type=str, default="experiments/ITOP-SIDE/1",
                        help="Path to the YAML config directory or file")
    parser.add_argument("--model", type=str, default="experiments/ITOP-SIDE/1/log/best_model.pth",
                        help="Path to the model checkpoint")
    parser.add_argument("--data-folder", type=str, required=True,
                        help="Folder containing .npz files to be loaded consecutively")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index within the folder")
    args = parser.parse_args()
    main(args)
