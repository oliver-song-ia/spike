#!/usr/bin/env python3
"""
ROS2 Pose Detector Node
Subscribes to /human_pointcloud topic, performs real-time pose detection using trained SPiKE model,
and publishes results to /pose_detection topic.

Coordinate transformations:
1. Input: Rotate around x-axis by 90 degrees, convert m to mm
2. Inference: Use trained model to predict pose
3. Output: Convert back to original coordinate system (mm to m, rotate back)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
import torch
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import model_builder
from utils.config_utils import load_config


class PoseDetector(Node):
    """ROS2 node for real-time human pose detection from point clouds"""

    def __init__(self):
        super().__init__('pose_detector')

        # Declare parameters
        self.declare_parameter('config_path', 'experiments/Custom/pretrained')
        self.declare_parameter('model_path', 'experiments/Custom/pretrained/log/best_model.pth')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('confidence_threshold', 0.5)

        # Get parameters
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_str = self.get_parameter('device').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # Initialize device
        self.device = torch.device(self.device_str if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Load model
        self.load_model()

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

        self.get_logger().info('Pose Detector node initialized')
        self.get_logger().info(f'Subscribing to: /human_pointcloud')
        self.get_logger().info(f'Publishing to: /pose_detection')

    def load_model(self):
        """Load the trained SPiKE model"""
        try:
            # Load config
            config = load_config(self.config_path)

            # Create model - output should be 15 joints * 3 coordinates = 45
            num_coord_joints = 45  # 15 joints * 3 coordinates each
            self.model = model_builder.create_model(config, num_coord_joints)
            self.model.to(self.device)

            # Load trained weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model'], strict=True)
                self.model.eval()
                self.get_logger().info(f'Model loaded from: {self.model_path}')
            else:
                self.get_logger().error(f'Model file not found: {self.model_path}')
                raise FileNotFoundError(f'Model file not found: {self.model_path}')

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise

    def pointcloud_to_numpy(self, pc_msg):
        """Convert ROS2 PointCloud2 message to numpy array"""
        import struct

        # Extract point data
        points = []
        point_step = pc_msg.point_step
        row_step = pc_msg.row_step

        for v in range(pc_msg.height):
            for u in range(pc_msg.width):
                byte_offset = v * row_step + u * point_step

                # Extract x, y, z (assuming float32)
                x = struct.unpack_from('f', pc_msg.data, byte_offset)[0]
                y = struct.unpack_from('f', pc_msg.data, byte_offset + 4)[0]
                z = struct.unpack_from('f', pc_msg.data, byte_offset + 8)[0]

                # Filter out invalid points
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    points.append([x, y, z])

        return np.array(points, dtype=np.float32)

    def preprocess_pointcloud(self, points):
        """
        Preprocess point cloud for model inference
        1. Rotate around x-axis by 90 degrees
        2. Convert from meters to millimeters
        3. Center the point cloud
        4. Resample to fixed number of points
        """
        if len(points) == 0:
            return None

        # Step 1: Rotate around x-axis by 90 degrees (counter-clockwise)
        # Rotation matrix for 90 degrees around x-axis
        rotation_x_90 = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

        points_rotated = np.dot(points, rotation_x_90.T)

        # Step 2: Convert from meters to millimeters
        points_mm = points_rotated * 1000.0

        # Step 3: Center the point cloud (subtract centroid)
        centroid = np.mean(points_mm, axis=0)
        points_centered = points_mm - centroid

        # Step 4: Resample to fixed number of points (e.g., 2048)
        target_num_points = 2048
        if len(points_centered) > target_num_points:
            # Randomly sample points
            indices = np.random.choice(len(points_centered), target_num_points, replace=False)
            points_resampled = points_centered[indices]
        elif len(points_centered) < target_num_points:
            # Duplicate points to reach target
            indices = np.random.choice(len(points_centered), target_num_points, replace=True)
            points_resampled = points_centered[indices]
        else:
            points_resampled = points_centered

        return points_resampled, centroid

    def postprocess_joints(self, joints_pred, centroid):
        """
        Postprocess predicted joints back to original coordinate system
        1. Add back the centroid
        2. Convert from millimeters to meters
        3. Rotate back around x-axis by -90 degrees
        """
        # Step 1: Add back centroid
        joints_world = joints_pred + centroid.reshape(1, 3)

        # Step 2: Convert from millimeters to meters
        joints_m = joints_world / 1000.0

        # Step 3: Rotate back around x-axis by -90 degrees (clockwise)
        # Rotation matrix for -90 degrees around x-axis (inverse of preprocessing)
        rotation_x_neg90 = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)

        joints_original = np.dot(joints_m, rotation_x_neg90.T)

        return joints_original

    def predict_pose(self, points):
        """Run model inference on preprocessed point cloud"""
        start_time = time.time()

        try:
            # Preprocess
            preprocess_start = time.time()
            preprocessed = self.preprocess_pointcloud(points)
            if preprocessed is None:
                return None
            preprocess_time = time.time() - preprocess_start

            points_tensor, centroid = preprocessed

            # Convert to tensor and add batch and temporal dimensions
            tensor_start = time.time()
            # SPiKE expects (batch_size, frames_per_clip, num_points, 3)
            # We replicate the single frame 16 times to match training format
            frames_per_clip = 16
            points_expanded = np.repeat(points_tensor[np.newaxis, :, :], frames_per_clip, axis=0)  # (16, 2048, 3)
            input_tensor = torch.from_numpy(points_expanded).unsqueeze(0).to(self.device)  # (1, 16, 2048, 3)
            tensor_time = time.time() - tensor_start

            # Debug info
            self.get_logger().debug(f'Input points shape: {points.shape}')
            self.get_logger().debug(f'Preprocessed points shape: {points_tensor.shape}')
            self.get_logger().debug(f'Centroid shape: {centroid.shape}')
            self.get_logger().debug(f'Input tensor shape: {input_tensor.shape}')

            # Run inference
            inference_start = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)

                # Reshape output from (1, 45) to (15, 3) - 15 joints, 3 coordinates each
                joints_pred = output.reshape(-1, 3).cpu().numpy()  # Should be (15, 3)
            inference_time = time.time() - inference_start

            # Postprocess back to original coordinate system
            postprocess_start = time.time()
            joints_original = self.postprocess_joints(joints_pred, centroid)
            postprocess_time = time.time() - postprocess_start

            # Calculate total time
            total_time = time.time() - start_time

            # Print timing information
            self.get_logger().info(f'Inference timing: '
                                  f'Total={total_time*1000:.1f}ms '
                                  f'(Preprocess={preprocess_time*1000:.1f}ms, '
                                  f'Tensor={tensor_time*1000:.1f}ms, '
                                  f'Inference={inference_time*1000:.1f}ms, '
                                  f'Postprocess={postprocess_time*1000:.1f}ms)')

            return joints_original

        except Exception as e:
            total_time = time.time() - start_time
            self.get_logger().error(f'Prediction failed after {total_time*1000:.1f}ms: {str(e)}')
            return None

    def create_skeleton_marker_array(self, joints, timestamp):
        """Create MarkerArray message for skeleton visualization (upper body only)"""
        marker_array = MarkerArray()

        # Define skeleton connections (bone connections for upper body)
        skeleton_connections = [
            (0, 1),   # Head -> Neck
            (1, 2),   # Neck -> R_Shoulder
            (1, 3),   # Neck -> L_Shoulder
            (2, 4),   # R_Shoulder -> R_Elbow
            (3, 5),   # L_Shoulder -> L_Elbow
            (4, 6),   # R_Elbow -> R_Hand
            (5, 7),   # L_Elbow -> L_Hand
            (1, 8),   # Neck -> Torso
        ]

        # Get upper body joints only (first 9 joints)
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

            # Position
            marker.pose.position.x = float(joint_coord[0])
            marker.pose.position.y = float(joint_coord[1])
            marker.pose.position.z = float(joint_coord[2])
            marker.pose.orientation.w = 1.0

            # Scale (sphere size)
            marker.scale.x = 0.03  # 3cm diameter
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            # Color (blue for joints)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            # Lifetime
            marker.lifetime.sec = 1  # 1 second lifetime

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

                # Add start and end points
                start_point = Point()
                start_point.x = float(upper_body_joints[start_idx][0])
                start_point.y = float(upper_body_joints[start_idx][1])
                start_point.z = float(upper_body_joints[start_idx][2])

                end_point = Point()
                end_point.x = float(upper_body_joints[end_idx][0])
                end_point.y = float(upper_body_joints[end_idx][1])
                end_point.z = float(upper_body_joints[end_idx][2])

                marker.points = [start_point, end_point]

                # Line width
                marker.scale.x = 0.01  # 1cm line width

                # Color (green for bones)
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                # Lifetime
                marker.lifetime.sec = 1

                marker_array.markers.append(marker)

        return marker_array

    def pointcloud_callback(self, msg):
        """Callback function for point cloud messages"""
        try:
            # Convert ROS message to numpy array
            points = self.pointcloud_to_numpy(msg)

            if len(points) < 100:  # Minimum point threshold
                self.get_logger().warn(f'Too few points in cloud: {len(points)}')
                return

            self.get_logger().debug(f'Processing point cloud with {len(points)} points')

            # Predict pose
            joints = self.predict_pose(points)

            if joints is not None:
                # Create and publish skeleton marker array
                marker_msg = self.create_skeleton_marker_array(joints, msg.header.stamp)
                self.pose_publisher.publish(marker_msg)

                self.get_logger().debug(f'Published skeleton with {len(marker_msg.markers)} markers')
            else:
                self.get_logger().warn('Failed to predict pose')

        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {str(e)}')


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = PoseDetector()
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