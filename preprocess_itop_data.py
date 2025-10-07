#!/usr/bin/env python3
"""
Preprocess ITOP dataset by combining individual .npz files into a single HDF5 file
This dramatically speeds up data loading from S3/FSx (from hours to seconds)
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from functools import partial


def load_single_pc(pc_name, point_clouds_folder):
    """Load a single point cloud file - for parallel processing"""
    try:
        pc_path = os.path.join(point_clouds_folder, pc_name)
        pc = np.load(pc_path)['arr_0']
        return pc, pc.shape[0]
    except Exception as e:
        return None, 0


def preprocess_point_clouds(data_root, output_file, split='train', num_workers=8):
    """
    Load all point cloud .npz files and save to a single HDF5 file

    Args:
        data_root: Root directory containing the ITOP format data
        output_file: Path to save the HDF5 file
        split: 'train' or 'test'
    """
    # Construct paths - data is directly in train/test folders
    point_clouds_folder = os.path.join(data_root, split)
    labels_file = os.path.join(data_root, f'{split}_labels.h5')

    print(f"Loading {split} data from: {data_root}")
    print(f"Point clouds folder: {point_clouds_folder}")
    print(f"Labels file: {labels_file}")

    # Check if paths exist
    if not os.path.exists(point_clouds_folder):
        raise FileNotFoundError(f"Point clouds folder not found: {point_clouds_folder}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Load labels to get identifiers
    print("Loading labels...")
    with h5py.File(labels_file, 'r') as f:
        identifiers = f['id'][:]
        joints = f['real_world_coordinates'][:]
        is_valid_flags = f['is_valid'][:]

    print(f"Found {len(identifiers)} samples")

    # Get list of point cloud files
    point_cloud_names = sorted(
        os.listdir(point_clouds_folder),
        key=lambda x: int(x.split('.')[0])
    )

    print(f"Found {len(point_cloud_names)} point cloud files")

    # Verify counts match
    if len(point_cloud_names) != len(identifiers):
        print(f"WARNING: Mismatch between point clouds ({len(point_cloud_names)}) and labels ({len(identifiers)})")

    # Create output HDF5 file
    print(f"\nCreating HDF5 file: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # First pass: determine max point cloud size (parallel)
    print("\nScanning point cloud sizes...")
    load_func = partial(load_single_pc, point_clouds_folder=point_clouds_folder)

    with Pool(num_workers) as pool:
        sample_results = list(tqdm(
            pool.imap(load_func, point_cloud_names[:100], chunksize=10),
            total=min(100, len(point_cloud_names)),
            desc="Sampling sizes"
        ))

    pc_sizes = [size for _, size in sample_results if size > 0]
    max_points = max(pc_sizes) if pc_sizes else 10000
    avg_points = int(np.mean(pc_sizes)) if pc_sizes else 5000

    print(f"Average points per cloud (sampled): {avg_points}")
    print(f"Max points per cloud (sampled): {max_points}")

    # Use a reasonable max size (add some buffer)
    max_points = max(max_points, 10000)

    with h5py.File(output_file, 'w') as hf:
        # Create datasets
        num_samples = len(point_cloud_names)

        # Point clouds: variable length, so we'll store as ragged array with size info
        # Reduce compression for faster writing
        pc_dataset = hf.create_dataset(
            'point_clouds',
            shape=(num_samples, max_points, 3),
            dtype='float32',
            compression='gzip',
            compression_opts=1  # Lighter compression for speed
        )

        # Store actual sizes
        sizes_dataset = hf.create_dataset(
            'sizes',
            shape=(num_samples,),
            dtype='int32'
        )

        # Store identifiers (as strings)
        id_dataset = hf.create_dataset(
            'identifiers',
            shape=(num_samples,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        # Store joints
        joints_dataset = hf.create_dataset(
            'joints',
            data=joints,
            dtype='float32'
        )

        # Store is_valid flags
        valid_dataset = hf.create_dataset(
            'is_valid',
            data=is_valid_flags,
            dtype='bool'
        )

        # Process all point clouds (parallel loading)
        print(f"\nProcessing {num_samples} point clouds...")

        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(load_func, point_cloud_names, chunksize=100),
                total=num_samples,
                desc=f"Loading {split}"
            ))

        # Write to HDF5
        print(f"Writing to HDF5...")
        for i, (pc, size) in enumerate(tqdm(results, desc=f"Writing {split}")):
            if pc is not None and size > 0:
                # Store point cloud (pad if necessary)
                actual_size = min(size, max_points)
                pc_dataset[i, :actual_size, :] = pc[:actual_size, :]
                sizes_dataset[i] = actual_size
                id_dataset[i] = identifiers[i].decode('utf-8')
            else:
                # Fill with zeros if error
                sizes_dataset[i] = 0
                id_dataset[i] = f"error_{i}"

        # Store metadata
        hf.attrs['split'] = split
        hf.attrs['num_samples'] = num_samples
        hf.attrs['max_points'] = max_points
        hf.attrs['source_path'] = data_root

    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nDone! Saved to {output_file}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Samples: {num_samples}")


def verify_hdf5(hdf5_file):
    """Verify the preprocessed HDF5 file"""
    print(f"\nVerifying {hdf5_file}...")

    with h5py.File(hdf5_file, 'r') as f:
        print(f"\nDatasets:")
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

        print(f"\nMetadata:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")

        # Show sample data
        print(f"\nSample data:")
        print(f"  First identifier: {f['identifiers'][0]}")
        print(f"  First point cloud size: {f['sizes'][0]}")
        print(f"  First point cloud shape: {f['point_clouds'][0, :f['sizes'][0], :].shape}")

    print("\nVerification complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess ITOP dataset for fast loading')
    parser.add_argument('--data_root', type=str, default='/mnt/fsx/fsx/Mocap_dataset',
                       help='Root directory of Mocap data')
    parser.add_argument('--output_dir', type=str, default='/home/ia-dev/oliver/spike/preprocessed',
                       help='Output directory for HDF5 files')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'both'], default='both',
                       help='Which split to process')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the HDF5 files after creation')

    args = parser.parse_args()

    # Process requested splits
    if args.split in ['train', 'both']:
        output_file = os.path.join(args.output_dir, 'train_data.h5')
        preprocess_point_clouds(args.data_root, output_file, split='train')

        if args.verify:
            verify_hdf5(output_file)

    if args.split in ['test', 'both']:
        output_file = os.path.join(args.output_dir, 'test_data.h5')
        preprocess_point_clouds(args.data_root, output_file, split='test')

        if args.verify:
            verify_hdf5(output_file)

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"HDF5 files saved to: {args.output_dir}")
    print("  - train_data.h5")
    print("  - test_data.h5")
