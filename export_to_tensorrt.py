#!/usr/bin/env python3
"""
Export SPiKE model to TensorRT format.
This script converts a PyTorch model to ONNX and then to TensorRT.
"""

import os
import sys
import torch
import yaml
import argparse
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "model"))

from model.model_builder import create_model


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config):
    """Load the trained model from checkpoint."""
    # Determine number of joints based on dataset
    if "ITOP" in config["dataset"] or config["dataset"] == "CUSTOM":
        num_coord_joints = 15 * 3  # 15 joints, 3 coordinates each
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    # Create model
    model = create_model(config, num_coord_joints)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()

    return model


def export_to_onnx(model, onnx_path, batch_size=1, frames_per_clip=3, num_points=4096, device='cuda'):
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Create dummy input
    # Input shape: [B, L, N, 3] where:
    # B = batch_size, L = frames_per_clip, N = num_points, 3 = xyz coordinates
    dummy_input = torch.randn(batch_size, frames_per_clip, num_points, 3).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['point_cloud'],
        output_names=['joint_coordinates'],
        dynamic_axes={
            'point_cloud': {0: 'batch_size'},
            'joint_coordinates': {0: 'batch_size'}
        }
    )

    print(f"✓ ONNX export successful: {onnx_path}")


def export_to_tensorrt(onnx_path, engine_path, fp16=True, workspace_size=4):
    """Convert ONNX model to TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: TensorRT is not installed.")
        print("Please install TensorRT: pip install tensorrt")
        return False

    print(f"Converting ONNX to TensorRT: {engine_path}")

    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))  # workspace_size GB

    # Enable FP16 if requested and supported
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")

    # Build engine
    print("Building TensorRT engine... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return False

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ TensorRT export successful: {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Export SPiKE model to TensorRT')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/oliver/Documents/SPiKE/experiments/Custom/pretrained-full/log/best_model.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--config', type=str,
                        default='/home/oliver/Documents/SPiKE/experiments/Custom/pretrained-full/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str,
                        default='/home/oliver/Documents/SPiKE/experiments/Custom/pretrained-full/log/best_model',
                        help='Output path (without extension)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision for TensorRT')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--onnx-only', action='store_true',
                        help='Export to ONNX only (skip TensorRT conversion)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for export (cuda or cpu)')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        return

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, config)

    # Get input parameters from config
    frames_per_clip = config.get('frames_per_clip', 3)
    num_points = config.get('num_points', 4096)

    # Set output paths
    onnx_path = f"{args.output}.onnx"
    engine_path = f"{args.output}.engine"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Export to ONNX
    export_to_onnx(model, onnx_path, args.batch_size, frames_per_clip, num_points, args.device)

    # Export to TensorRT
    if not args.onnx_only:
        success = export_to_tensorrt(onnx_path, engine_path, args.fp16, args.workspace)
        if success:
            print("\n" + "="*50)
            print("Export completed successfully!")
            print(f"ONNX model: {onnx_path}")
            print(f"TensorRT engine: {engine_path}")
            print("="*50)
    else:
        print("\n" + "="*50)
        print("ONNX export completed!")
        print(f"ONNX model: {onnx_path}")
        print("="*50)


if __name__ == '__main__':
    main()
