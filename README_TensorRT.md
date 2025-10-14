# TensorRT 推理使用指南

##问题说明

由于PyTorch的CUDA context和PyCUDA的context存在冲突,当前的TensorRT推理在某些情况下会失败。

## 解决方案

推荐使用以下独立的TensorRT推理脚本,不与PyTorch混用:

###  1. 简单的TensorRT测试

```bash
python tensorrt_wrapper.py
```

这个脚本可以测试TensorRT引擎是否正常工作,并进行性能基准测试。

### 2. 使用真实数据测试

```bash
python test_real_data.py
```

测试TensorRT引擎在实际数据上的推理效果。

### 3. 多次推理测试

```bash
python test_multiple_inferences.py
```

验证多次连续推理是否稳定。

## 模型文件

- **PyTorch模型**: `experiments/Custom/pretrained-full/log/best_model.pth` (498 MB)
- **ONNX模型**: `experiments/Custom/pretrained-full/log/best_model.onnx` (249 MB)
- **TensorRT引擎**: `experiments/Custom/pretrained-full/log/best_model.engine` (126 MB)

## TensorRT性能

根据基准测试结果:
- **平均推理时间**: 2.66 ± 0.64 ms
- **吞吐量**: 376 FPS

这比PyTorch推理快得多！

## 导出流程

### 1. PyTorch → ONNX

```bash
python export_to_tensorrt.py --onnx-only
```

### 2. ONNX → TensorRT

```bash
python onnx_to_trt_simple.py
```

## 已知问题

PyTorch和PyCUDA的CUDA context冲突导致在同一个Python进程中无法同时使用PyTorch的CUDA操作和TensorRT推理。

**临时解决方案**:
- 方案1: 使用纯TensorRT推理,不使用PyTorch CUDA
- 方案2: 在separate进程中运行TensorRT推理
- 方案3: 使用ONNX Runtime代替TensorRT (更简单,无context冲突)

## 替代方案: ONNX Runtime

ONNX Runtime不会与PyTorch产生CUDA context冲突,建议使用:

```bash
pip install onnxruntime-gpu
```

然后创建ONNX Runtime推理脚本即可。

## 文件说明

- `export_to_tensorrt.py` - 导出PyTorch模型到ONNX/TensorRT
- `onnx_to_trt_simple.py` - 将ONNX转换为TensorRT引擎
- `tensorrt_wrapper.py` - TensorRT推理封装类
- `predict_itop_tensorrt.py` - 使用TensorRT评估模型(有CUDA context冲突问题)
- `test_*.py` - 各种测试脚本

## 输入/输出格式

- **输入**: `[batch_size, frames, points, xyz]` = `[B, 3, 4096, 3]`
- **输出**: `[batch_size, joints*coords]` = `[B, 45]` (15关节 × 3坐标)

输出需要reshape为 `[B, 15, 3]` 来获得关节坐标。
