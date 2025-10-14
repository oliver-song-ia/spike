#!/bin/bash
# Jetson 自动部署脚本

# 配置 (请修改为你的 Jetson IP)
JETSON_IP="jetson@192.168.1.100"
JETSON_PATH="/home/jetson/spike_inference"

echo "================================================"
echo "SPiKE Model - Jetson Deployment Script"
echo "================================================"

# 1. 创建远程目录
echo ""
echo "[1/4] Creating directory on Jetson..."
ssh $JETSON_IP "mkdir -p $JETSON_PATH"

# 2. 复制文件
echo ""
echo "[2/4] Copying files to Jetson..."
scp experiments/Custom/pretrained-full/log/best_model.onnx $JETSON_IP:$JETSON_PATH/
scp build_jetson_engine.py $JETSON_IP:$JETSON_PATH/
scp tensorrt_wrapper.py $JETSON_IP:$JETSON_PATH/
scp jetson_simple_inference.py $JETSON_IP:$JETSON_PATH/

# 3. 在 Jetson 上构建引擎
echo ""
echo "[3/4] Building TensorRT engine on Jetson..."
echo "This will take 5-15 minutes..."
ssh $JETSON_IP << 'ENDSSH'
cd spike_inference

python3 build_jetson_engine.py \
    --onnx best_model.onnx \
    --output best_model_jetson.engine \
    --workspace 2

if [ $? -ne 0 ]; then
    echo "✗ Failed to build TensorRT engine"
    exit 1
fi
ENDSSH

if [ $? -ne 0 ]; then
    echo ""
    echo "Deployment failed!"
    exit 1
fi

# 4. 测试推理
echo ""
echo "[4/4] Testing inference..."
ssh $JETSON_IP << 'ENDSSH2'
cd spike_inference
python3 jetson_simple_inference.py
ENDSSH2

echo ""
echo "================================================"
echo "✓ Deployment Complete!"
echo "================================================"
echo ""
echo "To run inference on Jetson:"
echo "  ssh $JETSON_IP"
echo "  cd $JETSON_PATH"
echo "  python3 jetson_simple_inference.py"
echo ""
