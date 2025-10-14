# Jetson 部署指南

## 📦 需要的文件

### 从电脑复制到 Jetson:

```bash
# 1. ONNX 模型文件
experiments/Custom/pretrained-full/log/best_model.onnx  (249 MB)

# 2. Python 脚本
tensorrt_wrapper.py              # TensorRT 推理封装
build_jetson_engine.py          # ONNX → TensorRT 转换脚本
jetson_simple_inference.py      # 简单推理示例
```

## 🚀 快速开始

### 方法 1: 自动部署 (推荐)

```bash
# 修改脚本中的 Jetson IP 地址
nano deploy_to_jetson.sh

# 运行部署脚本
./deploy_to_jetson.sh
```

### 方法 2: 手动部署

#### 步骤 1: 复制文件到 Jetson

```bash
# 创建目录
ssh jetson@<IP> "mkdir -p ~/spike_inference"

# 复制文件
scp experiments/Custom/pretrained-full/log/best_model.onnx \
    jetson@<IP>:~/spike_inference/

scp tensorrt_wrapper.py \
    build_jetson_engine.py \
    jetson_simple_inference.py \
    jetson@<IP>:~/spike_inference/
```

#### 步骤 2: SSH 登录到 Jetson

```bash
ssh jetson@<IP>
cd ~/spike_inference
```

#### 步骤 3: 构建 TensorRT 引擎

```bash
# 安装依赖 (如果还没有)
pip3 install numpy pycuda

# 转换 ONNX 到 TensorRT (耗时 5-15 分钟)
python3 build_jetson_engine.py \
    --onnx best_model.onnx \
    --output best_model_jetson.engine \
    --fp16
```

#### 步骤 4: 测试推理

```bash
python3 jetson_simple_inference.py
```

## 💻 在你的代码中使用

```python
from jetson_simple_inference import JetsonPoseEstimator

# 初始化 (只需一次)
estimator = JetsonPoseEstimator("best_model_jetson.engine")

# 实时推理循环
while True:
    # 获取点云数据 [3, 4096, 3] 或 [1, 3, 4096, 3]
    point_cloud = get_point_cloud_from_sensor()

    # 推理 (~2-5ms on Jetson)
    joints = estimator.estimate_pose(point_cloud)

    # joints shape: [15, 3]
    # 15 个关节, 每个 3D 坐标 (x, y, z)

    # 使用结果
    head_pos = joints[0]  # 头部位置
    print(f"Head: {head_pos}")
```

## 📊 预期性能

| Jetson 型号 | 推理时间 | FPS |
|-------------|---------|-----|
| Jetson Orin | ~2-3 ms | 300-400 |
| Jetson Xavier | ~4-6 ms | 150-250 |
| Jetson Nano | ~15-25 ms | 40-60 |

## 🔧 进一步优化 (可选)

### INT8 量化 (可再快 2-4倍)

需要准备校准数据集,修改 `build_jetson_engine.py` 添加 INT8 支持。

### DLA 加速器 (部分 Jetson 支持)

在构建引擎时使用 DLA 可降低功耗。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `best_model.onnx` | ONNX 模型 (通用格式) |
| `best_model_jetson.engine` | Jetson 专用 TensorRT 引擎 |
| `tensorrt_wrapper.py` | TensorRT 推理封装类 |
| `build_jetson_engine.py` | ONNX → TensorRT 转换工具 |
| `jetson_simple_inference.py` | 简单易用的推理接口 |
| `deploy_to_jetson.sh` | 自动部署脚本 |

## ⚠️ 重要提示

1. **不要**直接复制电脑上的 `.engine` 文件到 Jetson
   - TensorRT 引擎是硬件相关的
   - 必须在 Jetson 上重新构建

2. **只需要** `.onnx` 文件是跨平台的
   - 在 Jetson 上用 ONNX 构建专用引擎

3. **内存占用**
   - TensorRT 引擎: ~126 MB
   - 推理时显存: ~200-300 MB

## 🆘 故障排除

### 如果构建引擎失败:

```bash
# 检查 TensorRT 是否安装
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 检查 CUDA
python3 -c "import pycuda.driver; print('CUDA OK')"

# 降低 workspace 大小
python3 build_jetson_engine.py --onnx best_model.onnx \
    --output best_model_jetson.engine --workspace 1
```

### 如果推理失败:

```bash
# 检查引擎文件
ls -lh best_model_jetson.engine

# 重新构建引擎
rm best_model_jetson.engine
python3 build_jetson_engine.py --onnx best_model.onnx \
    --output best_model_jetson.engine
```

## 📞 联系

如有问题，检查 `README_TensorRT.md` 了解更多技术细节。
