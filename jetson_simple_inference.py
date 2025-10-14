#!/usr/bin/env python3
"""
简单的 Jetson 实时推理示例
适用于方案1：纯 TensorRT 推理
"""

from tensorrt_wrapper import TensorRTModel
import numpy as np
import time

class JetsonPoseEstimator:
    """Jetson 姿态估计器"""

    def __init__(self, engine_path="best_model_jetson.engine"):
        """
        初始化姿态估计器

        Args:
            engine_path: TensorRT 引擎路径
        """
        print("Initializing Jetson Pose Estimator...")
        self.model = TensorRTModel(engine_path)

        # 预热
        print("Warming up...")
        dummy_input = np.random.randn(1, 3, 4096, 3).astype(np.float32)
        for _ in range(10):
            self.model.infer(dummy_input)

        print("✓ Ready for inference!")

    def estimate_pose(self, point_cloud):
        """
        从点云估计姿态

        Args:
            point_cloud: numpy array, shape [3, 4096, 3] 或 [1, 3, 4096, 3]
                        3帧 × 4096点 × xyz坐标

        Returns:
            joints: numpy array, shape [15, 3]
                   15个关节 × xyz坐标

        关节顺序：
            0: Head (头)
            1: Neck (颈)
            2: R Shoulder (右肩)
            3: L Shoulder (左肩)
            4: R Elbow (右肘)
            5: L Elbow (左肘)
            6: R Hand (右手)
            7: L Hand (左手)
            8: Torso (躯干)
            9-14: 下半身关节（如果不需要可以忽略）
        """
        # 确保输入是 [1, 3, 4096, 3]
        if point_cloud.ndim == 3:
            point_cloud = point_cloud[np.newaxis, ...]

        # 推理
        output = self.model.infer(point_cloud)

        # 转换为关节坐标
        joints = output.reshape(15, 3)

        return joints

    def benchmark(self, num_iterations=100):
        """
        性能基准测试

        Args:
            num_iterations: 测试迭代次数
        """
        print(f"\nRunning benchmark ({num_iterations} iterations)...")

        dummy_input = np.random.randn(1, 3, 4096, 3).astype(np.float32)
        times = []

        for i in range(num_iterations):
            start = time.time()
            _ = self.model.infer(dummy_input)
            times.append((time.time() - start) * 1000)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000.0 / avg_time

        print("\n" + "="*60)
        print("Benchmark Results:")
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min time: {min_time:.2f} ms")
        print(f"  Max time: {max_time:.2f} ms")
        print(f"  Throughput: {fps:.1f} FPS")
        print("="*60)

        return avg_time, fps


def main():
    """示例：如何使用"""

    # 1. 初始化估计器
    estimator = JetsonPoseEstimator("best_model_jetson.engine")

    # 2. 性能测试
    estimator.benchmark(100)

    # 3. 示例推理
    print("\n" + "="*60)
    print("Example Inference:")
    print("="*60)

    # 创建模拟点云数据
    # 实际使用时，这里应该是从 RealSense/其他传感器获取的真实数据
    point_cloud = np.random.randn(3, 4096, 3).astype(np.float32)
    print(f"Input point cloud shape: {point_cloud.shape}")

    # 推理
    start = time.time()
    joints = estimator.estimate_pose(point_cloud)
    inference_time = (time.time() - start) * 1000

    print(f"Inference time: {inference_time:.2f} ms")
    print(f"\nEstimated joint positions:")

    joint_names = [
        "Head", "Neck", "R Shoulder", "L Shoulder",
        "R Elbow", "L Elbow", "R Hand", "L Hand", "Torso",
        "R Hip", "L Hip", "R Knee", "L Knee", "R Foot", "L Foot"
    ]

    for i, (name, pos) in enumerate(zip(joint_names, joints)):
        print(f"  {i:2d}. {name:12s}: [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]")

    print("\n" + "="*60)
    print("Integration Example:")
    print("="*60)
    print("""
# 在你的应用中集成：

from jetson_simple_inference import JetsonPoseEstimator

# 初始化（只需要一次）
estimator = JetsonPoseEstimator("best_model_jetson.engine")

# 实时推理循环
while True:
    # 获取点云数据
    point_cloud = get_point_cloud_from_sensor()  # 你的传感器接口

    # 估计姿态
    joints = estimator.estimate_pose(point_cloud)

    # 使用结果
    head_pos = joints[0]  # 头部位置
    print(f"Head at: {head_pos}")

    # 或者传给机器人控制、可视化等
    robot.follow(joints)
    visualizer.draw(joints)
""")


if __name__ == "__main__":
    main()
