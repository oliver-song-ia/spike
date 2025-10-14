#!/usr/bin/env python3
"""
Convert PyTorch SPiKE model to TensorRT engine for faster inference
"""

import torch
import tensorrt as trt
import numpy as np
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import model_builder
from utils.config_utils import load_config

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file_path, engine_file_path, fp16_mode=True, max_batch_size=1):
    """
    Build TensorRT engine from ONNX file

    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        fp16_mode: Enable FP16 precision
        max_batch_size: Maximum batch size
    """
    print(f"Building TensorRT engine from {onnx_file_path}...")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("Parsing ONNX file...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    # Create builder config
    config = builder.create_builder_config()

    # Set memory pool limit (8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    # Enable FP16 if requested and supported
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    # Save engine
    print(f"Saving engine to {engine_file_path}...")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    print("TensorRT engine built successfully!")
    return engine_file_path


def export_to_onnx(model_path, config_path, onnx_path, device='cuda:0'):
    """
    Export PyTorch model to ONNX format using TorchScript tracing

    Args:
        model_path: Path to PyTorch model checkpoint
        config_path: Path to model config
        onnx_path: Path to save ONNX model
        device: Device to use
    """
    print(f"Loading PyTorch model from {model_path}...")

    # Load config
    config = load_config(config_path)

    # Create model
    num_coord_joints = 45  # 15 joints * 3 coordinates
    model = model_builder.create_model(config, num_coord_joints)

    # Load weights
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Create dummy input (batch_size=1, frames_per_clip=3, num_points=2048, coordinates=3)
    dummy_input = torch.randn(1, 3, 2048, 3, device=device)

    print(f"Exporting via TorchScript tracing first...")

    # First convert to TorchScript via tracing (more compatible than direct ONNX export)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    print(f"Exporting to ONNX: {onnx_path}...")

    # Export TorchScript model to ONNX
    torch.onnx.export(
        traced_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch SPiKE model to TensorRT')
    parser.add_argument('--config', type=str, default='experiments/Custom/pretrained-full',
                        help='Path to model config directory')
    parser.add_argument('--model', type=str, default='experiments/Custom/pretrained-full/log/best_model.pth',
                        help='Path to PyTorch model checkpoint')
    parser.add_argument('--output', type=str, default='experiments/Custom/pretrained-full/log/model.trt',
                        help='Output path for TensorRT engine')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision (default: True)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for conversion')

    args = parser.parse_args()

    # Create intermediate ONNX file
    onnx_path = args.output.replace('.trt', '.onnx')

    try:
        # Step 1: Export PyTorch to ONNX
        export_to_onnx(args.model, args.config, onnx_path, args.device)

        # Step 2: Build TensorRT engine from ONNX
        engine_path = build_engine(onnx_path, args.output, fp16_mode=args.fp16)

        if engine_path:
            print(f"\nConversion complete!")
            print(f"TensorRT engine saved to: {engine_path}")
            print(f"\nTo use the TensorRT engine in pose_detector:")
            print(f"  python pose_detector.py --use_tensorrt=True --tensorrt_engine_path={engine_path}")

            # Optionally remove ONNX file
            # os.remove(onnx_path)
        else:
            print("\nConversion failed!")
            return 1

    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
