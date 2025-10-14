#!/usr/bin/env python3
"""
ROS2 Pose Detector Node with TensorRT acceleration
Based on pose_detector.py but uses TensorRT engine for inference
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
import os
import time

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class PoseDetectorTRT(Node):
    """ROS2 node for real-time human pose detection using TensorRT"""

    def __init__(self):
        super().__init__('pose_detector_trt')

        # Declare parameters
        self.declare_parameter('engine_path', 'experiments/Custom/pretrained-full/log/best_model.trt')
        self.declare_parameter('confidence_threshold', 0.5)

        # Get parameters
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # Load TensorRT engine
        self.load_tensorrt_engine()

        # Pre-allocate rotation matrices
        self.rotation_x_90 = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

        self.rotation_x_neg90 = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)

        # Create subscribers and publishers
        self.pc_subscriber = self.create_subscription(
            PointCloud2,
            '/human_pointcloud',
            self.pointcloud_callback,
            10
        )

        self.pose_publisher = self.create_publisher(
            MarkerArray,
            '/pose_detection',
            10
        )

        # Joint names for pose array (upper body only - 9 joints)
        self.joint_names = [
            'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
            'R_Hand', 'L_Hand', 'Torso'
        ]

        self.get_logger().info('TensorRT Pose Detector node initialized')
        self.get_logger().info(f'Using TensorRT engine: {self.engine_path}')
        self.get_logger().info(f'Subscribing to: /human_pointcloud')
        self.get_logger().info(f'Publishing to: /pose_detection')

    def load_tensorrt_engine(self):
        """Load TensorRT engine"""
        try:
            self.get_logger().info('Loading TensorRT engine...')

            # Create TensorRT logger
            self.trt_logger = trt.Logger(trt.Logger.WARNING)

            # Load engine
            if not os.path.exists(self.engine_path):
                raise FileNotFoundError(f'TensorRT engine not found: {self.engine_path}')

            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.trt_logger)
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())

            self.trt_context = self.trt_engine.create_execution_context()

            # Get input/output shapes
            self.input_shape = self.trt_engine.get_binding_shape(0)  # (1, 3, 4096, 3)
            self.output_shape = self.trt_engine.get_binding_shape(1)  # (1, 45)

            self.get_logger().info(f'Input shape: {self.input_shape}')
            self.get_logger().info(f'Output shape: {self.output_shape}')

            # Allocate GPU memory for input/output
            self.input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
            self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize

            # Allocate device memory
            self.d_input = cuda.mem_alloc(self.input_size)
            self.d_output = cuda.mem_alloc(self.output_size)

            # Create CUDA stream
            self.stream = cuda.Stream()

            # Pre-allocate host output buffer
            self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

            self.get_logger().info(f'TensorRT engine loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load TensorRT engine: {str(e)}')
            raise

    def pointcloud_to_numpy(self, pc_msg):
        """Convert ROS2 PointCloud2 message to numpy array (optimized)"""
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])

        cloud_array = np.frombuffer(pc_msg.data, dtype=dtype)

        if pc_msg.height > 1:
            cloud_array = cloud_array.reshape((pc_msg.height, pc_msg.width))
            cloud_array = cloud_array.reshape(-1)

        points = np.column_stack((cloud_array['x'], cloud_array['y'], cloud_array['z']))

        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]

        return points.astype(np.float32)

    def preprocess_pointcloud(self, points):
        """Preprocess point cloud for TensorRT inference"""
        if len(points) == 0:
            return None

        # Rotate around x-axis by 90 degrees
        points_rotated = np.dot(points, self.rotation_x_90.T)

        # Convert from meters to millimeters
        points_mm = points_rotated * 1000.0

        # Center the point cloud
        centroid = np.mean(points_mm, axis=0)
        points_centered = points_mm - centroid

        # Resample to 4096 points (as expected by the ONNX model)
        target_num_points = 4096
        if len(points_centered) > target_num_points:
            indices = np.random.choice(len(points_centered), target_num_points, replace=False)
            points_resampled = points_centered[indices]
        elif len(points_centered) < target_num_points:
            indices = np.random.choice(len(points_centered), target_num_points, replace=True)
            points_resampled = points_centered[indices]
        else:
            points_resampled = points_centered

        return points_resampled, centroid

    def postprocess_joints(self, joints_pred, centroid):
        """Postprocess predicted joints back to original coordinate system"""
        joints_world = joints_pred + centroid.reshape(1, 3)
        joints_m = joints_world / 1000.0
        joints_original = np.dot(joints_m, self.rotation_x_neg90.T)
        return joints_original

    def predict_pose(self, points):
        """Run TensorRT inference"""
        try:
            preprocessed = self.preprocess_pointcloud(points)
            if preprocessed is None:
                return None, 0.0

            points_tensor, centroid = preprocessed

            # Prepare input: replicate to 3 frames (batch=1, frames=3, points=4096, coords=3)
            frames_per_clip = 3
            points_expanded = np.repeat(points_tensor[np.newaxis, :, :], frames_per_clip, axis=0)
            input_data = points_expanded[np.newaxis, :, :, :].astype(np.float32)  # (1, 3, 4096, 3)

            # Run inference
            inference_start = time.time()

            # Copy input to GPU
            cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)

            # Execute inference
            self.trt_context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle
            )

            # Copy output from GPU
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

            inference_time = (time.time() - inference_start) * 1000  # ms

            # Reshape output from (45,) to (15, 3)
            joints_pred = self.h_output.reshape(-1, 3)

            # Postprocess
            joints_original = self.postprocess_joints(joints_pred, centroid)

            return joints_original, inference_time

        except Exception as e:
            self.get_logger().error(f'Prediction failed: {str(e)}')
            return None, 0.0

    def create_skeleton_marker_array(self, joints, timestamp):
        """Create MarkerArray message for skeleton visualization"""
        marker_array = MarkerArray()

        skeleton_connections = [
            (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (1, 8),
        ]

        upper_body_joints = joints[:9]

        # Create joint markers (spheres)
        for i, joint_coord in enumerate(upper_body_joints):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = timestamp
            marker.ns = 'skeleton_joints'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(joint_coord[0])
            marker.pose.position.y = float(joint_coord[1])
            marker.pose.position.z = float(joint_coord[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

        # Create bone markers (lines)
        for i, (start_idx, end_idx) in enumerate(skeleton_connections):
            if start_idx < len(upper_body_joints) and end_idx < len(upper_body_joints):
                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = timestamp
                marker.ns = 'skeleton_bones'
                marker.id = i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD

                start_point = Point()
                start_point.x = float(upper_body_joints[start_idx][0])
                start_point.y = float(upper_body_joints[start_idx][1])
                start_point.z = float(upper_body_joints[start_idx][2])

                end_point = Point()
                end_point.x = float(upper_body_joints[end_idx][0])
                end_point.y = float(upper_body_joints[end_idx][1])
                end_point.z = float(upper_body_joints[end_idx][2])

                marker.points = [start_point, end_point]

                marker.scale.x = 0.01

                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.lifetime.sec = 1

                marker_array.markers.append(marker)

        return marker_array

    def pointcloud_callback(self, msg):
        """Callback function for point cloud messages"""
        try:
            callback_start_time = time.time()

            # Convert ROS message to numpy array
            points = self.pointcloud_to_numpy(msg)

            if len(points) < 100:
                self.get_logger().warn(f'Too few points in cloud: {len(points)}')
                return

            self.get_logger().debug(f'Processing point cloud with {len(points)} points')

            # Predict pose
            joints, inference_time = self.predict_pose(points)

            if joints is not None:
                # Create and publish skeleton marker array
                marker_msg = self.create_skeleton_marker_array(joints, msg.header.stamp)

                publish_time = time.time()
                self.pose_publisher.publish(marker_msg)

                # Calculate latencies
                input_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                current_ros_time = self.get_clock().now()
                current_time_sec = current_ros_time.nanoseconds * 1e-9

                end_to_end_latency = (current_time_sec - input_timestamp) * 1000
                processing_time = (publish_time - callback_start_time) * 1000

                self.get_logger().info(
                    f'Latency: End-to-end={end_to_end_latency:.1f}ms, '
                    f'Processing={processing_time:.1f}ms, '
                    f'Inference={inference_time:.1f}ms'
                )
                self.get_logger().debug(f'Published skeleton with {len(marker_msg.markers)} markers')
            else:
                self.get_logger().warn('Failed to predict pose')

        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {str(e)}')


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = PoseDetectorTRT()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
