#!/usr/bin/env python3
"""
在 Jetson 上将 ONNX 模型转换为 TensorRT 引擎
"""

import os
import sys
import argparse
import tensorrt as trt
import numpy as np


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=2):
    """
    将 ONNX 模型转换为 TensorRT 引擎

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出引擎路径
        fp16: 是否使用 FP16 精度
        workspace_gb: 工作空间大小(GB)
    """
    print("="*60)
    print("Building TensorRT Engine for Jetson")
    print("="*60)
    print(f"ONNX model: {onnx_path}")
    print(f"Output engine: {engine_path}")
    print(f"FP16: {fp16}")
    print(f"Workspace: {workspace_gb} GB")
    print()

    # 创建 logger
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建 builder
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    print("✓ ONNX parsed successfully")

    # 创建配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # 添加优化配置文件 (支持动态 batch size)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 3, 4096, 3), (1, 3, 4096, 3), (4, 3, 4096, 3))
    config.add_optimization_profile(profile)

    # 启用 FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 enabled")

    # 构建引擎
    print("\nBuilding engine (this may take 5-15 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    # 保存引擎
    print(f"\nSaving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"✓ Engine saved ({size_mb:.1f} MB)")

    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engine for Jetson')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Output engine path')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 precision')
    parser.add_argument('--workspace', type=int, default=2, help='Workspace size in GB')

    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX file not found: {args.onnx}")
        sys.exit(1)

    success = build_engine(args.onnx, args.output, args.fp16, args.workspace)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
