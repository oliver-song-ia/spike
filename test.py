"""
Visualized testing script based on predict_itop.py
- Randomly samples one frame from test dataset
- Runs inference and visualizes point cloud, ground truth, and predicted keypoints
"""

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import model_builder
from trainer_itop import load_data, create_criterion
from utils.config_utils import load_config, set_random_seed
from utils.metrics import joint_accuracy
from const import skeleton_joints


def _extract_pointcloud_from_clip(clip_np: np.ndarray, target_frame: str = "last") -> np.ndarray:
    """Extract single frame (N,3) from (T,N,3) arrays."""
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


def visualize_sample(pc, gt_joints, pred_joints, sample_id, loss, pck, mAP, threshold):
    """
    Visualize point cloud, ground truth joints, and predicted joints

    Args:
        pc: Point cloud (N, 3)
        gt_joints: Ground truth joints (15, 3)
        pred_joints: Predicted joints (15, 3)
        sample_id: Sample identifier
        loss: Loss value
        pck: PCK per joint (15,)
        mAP: Mean average precision
        threshold: PCK threshold
    """
    print(f"\nVisualization for sample {sample_id}:")
    print(f"  Loss: {loss:.4f}")
    print(f"  mAP: {mAP:.2f}%")
    print(f"  PCK (threshold={threshold}mm): {pck}")
    print(f"  Point cloud shape: {pc.shape}")
    print(f"  GT joints shape: {gt_joints.shape}")
    print(f"  Pred joints shape: {pred_joints.shape}")

    # Convert point cloud from meters to millimeters if needed
    pc_scale = pc.max() - pc.min()
    gt_scale = gt_joints.max() - gt_joints.min()

    if pc_scale < 10 and gt_scale > 100:  # PC in meters, GT in mm
        pc_mm = pc * 1000
        print(f"  Converting PC from meters to millimeters")
    else:
        pc_mm = pc

    print(f"  PC range: X[{pc_mm[:,0].min():.1f}, {pc_mm[:,0].max():.1f}] "
          f"Y[{pc_mm[:,1].min():.1f}, {pc_mm[:,1].max():.1f}] "
          f"Z[{pc_mm[:,2].min():.1f}, {pc_mm[:,2].max():.1f}] mm")
    print(f"  GT range: X[{gt_joints[:,0].min():.1f}, {gt_joints[:,0].max():.1f}] "
          f"Y[{gt_joints[:,1].min():.1f}, {gt_joints[:,1].max():.1f}] "
          f"Z[{gt_joints[:,2].min():.1f}, {gt_joints[:,2].max():.1f}] mm")
    print(f"  Pred range: X[{pred_joints[:,0].min():.1f}, {pred_joints[:,0].max():.1f}] "
          f"Y[{pred_joints[:,1].min():.1f}, {pred_joints[:,1].max():.1f}] "
          f"Z[{pred_joints[:,2].min():.1f}, {pred_joints[:,2].max():.1f}] mm")

    # Create single plot with all elements
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Point Cloud (subsample for clarity)
    pc_vis = pc_mm if pc_mm.shape[0] <= 8000 else pc_mm[np.random.choice(pc_mm.shape[0], 8000, replace=False)]
    ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
               s=3.0, alpha=0.7, c='steelblue', edgecolors='darkblue', linewidth=0.1, label='Point Cloud')

    # Plot Ground Truth Joints
    ax.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2],
               s=200, marker='o', color='green', edgecolors='darkgreen', linewidth=2,
               label='Ground Truth', alpha=0.9)

    # Plot Predicted Joints
    ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2],
               s=200, marker='^', color='red', edgecolors='darkred', linewidth=2,
               label='Prediction', alpha=0.9)

    # Draw GT skeleton (green)
    for j1, j2, _ in skeleton_joints.joint_connections:
        if j1 < len(gt_joints) and j2 < len(gt_joints):
            ax.plot([gt_joints[j1, 0], gt_joints[j2, 0]],
                   [gt_joints[j1, 1], gt_joints[j2, 1]],
                   [gt_joints[j1, 2], gt_joints[j2, 2]],
                   color='green', linewidth=3, alpha=0.7)

    # Draw predicted skeleton (red)
    for j1, j2, _ in skeleton_joints.joint_connections:
        if j1 < len(pred_joints) and j2 < len(pred_joints):
            ax.plot([pred_joints[j1, 0], pred_joints[j2, 0]],
                   [pred_joints[j1, 1], pred_joints[j2, 1]],
                   [pred_joints[j1, 2], pred_joints[j2, 2]],
                   color='red', linewidth=3, alpha=0.7, linestyle='--')

    # Add joint labels for key joints only (to avoid clutter)
    key_joints = [0, 1, 8, 2, 3, 6, 7]  # Head, Neck, Torso, Shoulders, Hands
    for i in key_joints:
        if i < len(pred_joints):
            joint_name = skeleton_joints.joint_indices.get(i, f"Joint{i}")
            pck_val = pck[i] if i < len(pck) else 0

            # Position label between GT and prediction
            label_pos = (gt_joints[i] + pred_joints[i]) / 2
            color = "lightgreen" if pck_val > 80 else "yellow" if pck_val > 50 else "lightcoral"

            ax.text(label_pos[0], label_pos[1], label_pos[2],
                   f'{i}:{joint_name}\nPCK:{pck_val:.1f}%',
                   fontsize=8, alpha=0.9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

    # Set equal aspect ratio using all points
    all_joints = np.vstack([gt_joints, pred_joints])
    _equal_axis_3d(ax, all_joints)

    ax.set_title(f'Sample {sample_id} - Loss: {loss:.4f}, mAP: {mAP:.2f}%', fontsize=14)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def test_random_sample(model, criterion, data_loader, device, threshold, target_frame="last"):
    """
    Test on a random sample from the dataset and visualize results
    """
    model.eval()

    # Randomly select a sample
    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    print(f"Testing on random sample index: {random_idx}")

    with torch.no_grad():
        # Get specific sample
        clip, target, video_id = data_loader.dataset[random_idx]

        # Add batch dimension
        clip_batch = clip.unsqueeze(0).to(device, non_blocking=True)
        target_batch = target.unsqueeze(0).to(device, non_blocking=True)

        # Forward pass
        output = model(clip_batch).reshape(target_batch.shape)
        loss = criterion(output, target_batch)

        # Calculate metrics
        pck, mean_ap = joint_accuracy(output, target_batch, threshold)

        # Convert to numpy for visualization
        clip_np = clip.numpy()
        target_np = target.numpy()
        output_np = output.squeeze(0).cpu().numpy()
        pck_np = pck.cpu().numpy()
        map_val = mean_ap.cpu().item()

        # Extract point cloud from clip
        pc = _extract_pointcloud_from_clip(clip_np, target_frame=target_frame)
        gt_joints = _to_K3(target_np)
        pred_joints = _to_K3(output_np)

        # Visualize
        visualize_sample(pc, gt_joints, pred_joints,
                        video_id, loss.item(), pck_np, map_val, threshold)

        return loss.item(), map_val, pck_np


def main(arguments):
    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device(0)

    set_random_seed(config["seed"])

    print(f"Loading test data...")
    data_loader_test, num_coord_joints = load_data(config, mode="test")
    print(f"Test dataset size: {len(data_loader_test.dataset)}")

    # Load model
    model = model_builder.create_model(config, num_coord_joints)
    model.to(device)

    criterion = create_criterion(config)

    print(f"Loading model from {arguments.model}")
    checkpoint = torch.load(arguments.model, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    # Test on random sample
    target_frame = str(config.get("target_frame", "last"))
    threshold = config["threshold"]

    print(f"Testing with threshold: {threshold}")
    print(f"Target frame: {target_frame}")

    loss, mAP, pck = test_random_sample(
        model, criterion, data_loader_test, device, threshold, target_frame
    )

    print(f"\nTest Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  mAP: {mAP:.2f}%")
    print(f"  PCK per joint: {pck}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE Visualized Testing")
    parser.add_argument("--config", type=str, default="experiments/Custom/1",
                        help="Path to the YAML config directory or file")
    parser.add_argument("--model", type=str, default="experiments/Custom/1/log/best_model.pth",
                        help="Path to the model checkpoint")

    args = parser.parse_args()
    main(args)