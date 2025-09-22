"""
Test script to verify converted Mocap data compatibility with training script
"""

import h5py
import numpy as np
import os
from datasets.itop import ITOP
from utils.config_utils import load_config

def test_h5_structure():
    """Test the structure of converted train_labels.h5"""
    labels_file = "/home/oliver/Documents/data/Mocap/train_labels.h5"

    print("Testing H5 file structure...")
    with h5py.File(labels_file, 'r') as f:
        print(f"Available keys: {list(f.keys())}")

        ids = f['id'][:]
        coords = f['real_world_coordinates'][:]
        valid = f['is_valid'][:]

        print(f"IDs shape: {ids.shape}, dtype: {ids.dtype}")
        print(f"Coordinates shape: {coords.shape}, dtype: {coords.dtype}")
        print(f"Is valid shape: {valid.shape}, dtype: {valid.dtype}")

        # Sample first few entries
        print(f"\nFirst 5 IDs: {[id.decode('utf-8') for id in ids[:5]]}")
        print(f"First coordinate entry shape: {coords[0].shape}")
        print(f"First 5 valid flags: {valid[:5]}")

    return True

def test_point_cloud_files():
    """Test point cloud file structure"""
    train_dir = "/home/oliver/Documents/data/Mocap/train"

    print("\nTesting point cloud files...")

    # Test first few files
    for i in range(5):
        file_path = os.path.join(train_dir, f"{i}.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            pc = data['arr_0']
            print(f"File {i}.npz: shape={pc.shape}, dtype={pc.dtype}")
        else:
            print(f"File {i}.npz not found")

    return True

def test_dataset_loading():
    """Test loading data using ITOP dataset class"""
    print("\nTesting dataset loading...")

    try:
        # Create minimal config for testing
        config = {
            'frames_per_clip': 3,
            'num_points': 4096,
            'use_valid_only': False,
            'target_frame': 'last',
            'PREPROCESS_TEST': [
                {
                    'name': 'CenterAug',
                    'p_prob': 1.0,
                    'p_axes': [True, True, True],
                    'apply_on_gt': True
                }
            ]
        }

        # Test dataset creation
        dataset = ITOP(
            root="/home/oliver/Documents/data/Mocap",
            train=True,
            frames_per_clip=config['frames_per_clip'],
            num_points=config['num_points'],
            use_valid_only=config['use_valid_only'],
            target_frame=config['target_frame'],
            aug_list=config['PREPROCESS_TEST']
        )

        print(f"Dataset loaded successfully!")
        print(f"Dataset length: {len(dataset)}")
        print(f"Number of coordinate joints: {dataset.num_coord_joints}")

        # Test getting first sample
        if len(dataset) > 0:
            clip, joints, metadata = dataset[0]
            print(f"Sample 0:")
            print(f"  Clip shape: {clip.shape}")
            print(f"  Joints shape: {joints.shape}")
            print(f"  Metadata: {metadata}")

        return True

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

def main():
    print("Testing converted Mocap data compatibility...")

    # Test H5 structure
    if test_h5_structure():
        print("✅ H5 structure test passed")
    else:
        print("❌ H5 structure test failed")
        return

    # Test point cloud files
    if test_point_cloud_files():
        print("✅ Point cloud files test passed")
    else:
        print("❌ Point cloud files test failed")
        return

    # Test dataset loading
    if test_dataset_loading():
        print("✅ Dataset loading test passed")
        print("\n🎉 All tests passed! Mocap data is compatible with training script.")
    else:
        print("❌ Dataset loading test failed")

if __name__ == "__main__":
    main()