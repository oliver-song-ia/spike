#!/usr/bin/env python3
"""
Compare inference results between Custom/1 and Custom/2 experiments
"""

import pandas as pd
import numpy as np
import h5py
import os
from pathlib import Path

def load_h5_data(h5_file):
    """Load data from H5 file"""
    with h5py.File(h5_file, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][:]
        print(f"Loaded H5 file {h5_file}:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    return data

def load_csv_data(csv_file):
    """Load CSV data"""
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV file {csv_file}: {df.shape}")
    return df

def compare_csv_files(csv1_path, csv2_path, file_type="predictions"):
    """Compare two CSV files"""
    print(f"\n=== Comparing {file_type} CSV files ===")
    print(f"File 1: {csv1_path}")
    print(f"File 2: {csv2_path}")

    if not os.path.exists(csv1_path):
        print(f"Warning: File 1 does not exist")
        return None
    if not os.path.exists(csv2_path):
        print(f"Warning: File 2 does not exist")
        return None

    df1 = load_csv_data(csv1_path)
    df2 = load_csv_data(csv2_path)

    # Basic comparison
    print(f"\nShape comparison:")
    print(f"  Custom/1: {df1.shape}")
    print(f"  Custom/2: {df2.shape}")

    # Since Custom/2 is a subset of Custom/1, find the intersection
    min_len = min(len(df1), len(df2))
    print(f"  Taking first {min_len} rows for comparison")

    # Truncate both to same length for comparison
    df1_subset = df1.iloc[:min_len].copy()
    df2_subset = df2.iloc[:min_len].copy()

    # Column comparison
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_cols = cols1.intersection(cols2)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1

    print(f"\nColumn comparison:")
    print(f"  Common columns: {len(common_cols)}")
    print(f"  Only in Custom/1: {len(only_in_1)} - {list(only_in_1)[:5]}")
    print(f"  Only in Custom/2: {len(only_in_2)} - {list(only_in_2)[:5]}")

    # Data comparison for common columns
    if len(common_cols) > 0:
        print(f"\nData comparison for common columns (first {min_len} rows):")

        # Compare basic info columns
        info_cols = ['traj_id', 'frame_id', 'frame_in_session']
        for col in info_cols:
            if col in common_cols:
                if col in df1_subset.columns and col in df2_subset.columns:
                    diff = (df1_subset[col] != df2_subset[col]).sum()
                    total = len(df1_subset)
                    print(f"  {col}: {diff}/{total} differences ({diff/total*100:.1f}%)")

        # Compare joint prediction columns
        joint_cols = [col for col in common_cols if '_pred_' in col]
        if len(joint_cols) > 0:
            print(f"  Joint prediction columns: {len(joint_cols)}")
            for joint_type in ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso']:
                x_col = f'{joint_type}_pred_x'
                y_col = f'{joint_type}_pred_y'
                z_col = f'{joint_type}_pred_z'

                if all(col in common_cols for col in [x_col, y_col, z_col]):
                    # Calculate differences using subsets
                    diff_x = np.abs(df1_subset[x_col] - df2_subset[x_col]).mean()
                    diff_y = np.abs(df1_subset[y_col] - df2_subset[y_col]).mean()
                    diff_z = np.abs(df1_subset[z_col] - df2_subset[z_col]).mean()
                    total_diff = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
                    print(f"    {joint_type}: mean diff = {total_diff:.2f}mm (x:{diff_x:.2f}, y:{diff_y:.2f}, z:{diff_z:.2f})")

        # Compare arm coordinate columns
        arm_cols = [col for col in common_cols if col.startswith(('Left_', 'Right_'))]
        if len(arm_cols) > 0:
            print(f"  Arm coordinate columns: {len(arm_cols)}")
            for arm_point in ['Left_L1', 'Left_L2', 'Right_R1', 'Right_R2']:
                x_col = f'{arm_point}_x'
                y_col = f'{arm_point}_y'
                z_col = f'{arm_point}_z'

                if all(col in common_cols for col in [x_col, y_col, z_col]):
                    # Calculate differences using subsets
                    diff_x = np.abs(df1_subset[x_col] - df2_subset[x_col]).mean()
                    diff_y = np.abs(df1_subset[y_col] - df2_subset[y_col]).mean()
                    diff_z = np.abs(df1_subset[z_col] - df2_subset[z_col]).mean()
                    total_diff = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
                    print(f"    {arm_point}: mean diff = {total_diff:.2f}mm (x:{diff_x:.2f}, y:{diff_y:.2f}, z:{diff_z:.2f})")

    return df1_subset, df2_subset

def compare_h5_predictions(h5_file, reference_csv=None):
    """Compare H5 predictions with reference data"""
    print(f"\n=== Analyzing H5 predictions ===")

    if not os.path.exists(h5_file):
        print(f"H5 file does not exist: {h5_file}")
        return None

    h5_data = load_h5_data(h5_file)

    if 'predicted_coordinates' in h5_data and 'real_world_coordinates' in h5_data:
        pred_coords = h5_data['predicted_coordinates']  # Shape: (N, 15, 3)
        gt_coords = h5_data['real_world_coordinates']   # Shape: (N, 15, 3)

        print(f"\nH5 Data Analysis:")
        print(f"  Predictions shape: {pred_coords.shape}")
        print(f"  Ground truth shape: {gt_coords.shape}")

        # Calculate per-joint errors
        joint_names = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
                      'R_Hand', 'L_Hand', 'Torso', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Foot', 'L_Foot']

        print(f"\nPer-joint prediction errors:")
        for i, joint_name in enumerate(joint_names):
            if i < pred_coords.shape[1]:
                # Calculate 3D distance error
                diff = pred_coords[:, i, :] - gt_coords[:, i, :]
                distances = np.sqrt(np.sum(diff**2, axis=1))
                mean_error = np.mean(distances)
                std_error = np.std(distances)
                print(f"  {i:2d} {joint_name:12s}: {mean_error:7.2f} ± {std_error:6.2f} mm")

        # Overall statistics
        all_diff = pred_coords - gt_coords
        all_distances = np.sqrt(np.sum(all_diff**2, axis=2))  # Shape: (N, 15)
        overall_mean = np.mean(all_distances)
        overall_std = np.std(all_distances)

        print(f"\nOverall prediction error: {overall_mean:.2f} ± {overall_std:.2f} mm")

        return h5_data

    return h5_data

def main():
    """Main comparison function"""
    print("=== SPiKE Inference Results Comparison ===")

    # Define file paths
    custom1_base = "/home/oliver/Documents/SPiKE/experiments/Custom/1/log"
    custom2_base = "/home/oliver/Documents/SPiKE/experiments/Custom/2/log"

    # Compare prediction CSV files
    pred_csv1 = os.path.join(custom1_base, "inference_labels_trajectory.csv")
    pred_csv2 = os.path.join(custom2_base, "inference_labels_trajectory.csv")
    compare_csv_files(pred_csv1, pred_csv2, "Predictions")

    # Compare ground truth CSV files
    gt_csv1 = os.path.join(custom1_base, "inference_labels_trajectory_gt.csv")
    gt_csv2 = os.path.join(custom2_base, "inference_labels_trajectory_gt.csv")
    compare_csv_files(gt_csv1, gt_csv2, "Ground Truth")

    # Analyze H5 file from Custom/2 (if available)
    h5_file = os.path.join(custom2_base, "inference_labels.h5")
    compare_h5_predictions(h5_file)

    print("\n=== Comparison Complete ===")

if __name__ == "__main__":
    main()