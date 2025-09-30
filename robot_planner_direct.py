#!/usr/bin/env python3
"""
ROS2 Robot Trajectory Detector Node
Subscribes to /human_pointcloud topic, performs real-time robot trajectory prediction using trained SPiKE robot model,
and publishes results to /robot_trajectory topic.

Coordinate transformations:
1. Input: Rotate around x-axis by 90 degrees, convert m to mm
2. Inference: Use trained robot model to predict 4 arm points (Left_L1, Left_L2, Right_R1, Right_R2)
3. Output: Convert back to original coordinate system (mm to m, rotate back)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, PoseArray, Pose
import numpy as np
import torch
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import model_builder_robot
from utils.config_utils import load_config
from const import robot_trajectory


class RobotPlannerDirect(Node):
    """ROS2 node for real-time robot trajectory planning with target poses"""

    def __init__(self):
        super().__init__('robot_planner_direct')

        # Declare parameters
        self.declare_parameter('config_path', 'experiments/Robot/2')
        self.declare_parameter('model_path', 'experiments/Robot/2/log/best_model_robot.pth')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('frames_per_clip', 16)
        self.declare_parameter('num_points', 2048)

        # Get parameters
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_str = self.get_parameter('device').get_parameter_value().string_value
        self.frames_per_clip = self.get_parameter('frames_per_clip').get_parameter_value().integer_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value

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
            PoseArray,
            '/target_poses',
            10
        )

        # Keep visualization publisher for debugging
        self.robot_publisher = self.create_publisher(
            MarkerArray,
            '/robot_trajectory_debug',
            10
        )

        # Robot arm point names
        self.robot_point_names = robot_trajectory.robot_point_names  # ["Left_L1", "Left_L2", "Right_R1", "Right_R2"]

        # Buffer for point cloud frames (for temporal consistency)
        self.frame_buffer = []
        self.max_buffer_size = self.frames_per_clip

        # Smoothing buffer for poses
        self.pose_buffer = []
        self.pose_buffer_size = 5
        self.target_arm_length = 0.4  # 40cm target arm length

        # Logging control
        self.log_counter = 0
        self.log_interval = 20  # Log every 20 frames to reduce spam

        self.get_logger().info('Robot Planner Direct node initialized')
        self.get_logger().info(f'Subscribing to: /human_pointcloud')
        self.get_logger().info(f'Publishing to: /target_poses')
        self.get_logger().info(f'Debug visualization: /robot_trajectory_debug')
        self.get_logger().info(f'Expected model output: {robot_trajectory.NUM_ROBOT_COORDS} coordinates (4 points × 3 coords)')

    def load_model(self):
        """Load the trained SPiKE robot trajectory model"""
        try:
            # Load config
            config = load_config(self.config_path)

            # Create model - output should be 12 coordinates (4 robot points × 3 coordinates each)
            num_robot_coords = robot_trajectory.NUM_ROBOT_COORDS  # 12
            self.model = model_builder_robot.create_robot_model(config, num_robot_coords)
            self.model.to(self.device)

            # Load trained weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model'], strict=True)
                self.model.eval()
                self.get_logger().info(f'Robot model loaded from: {self.model_path}')
            else:
                self.get_logger().error(f'Model file not found: {self.model_path}')
                raise FileNotFoundError(f'Model file not found: {self.model_path}')

        except Exception as e:
            self.get_logger().error(f'Failed to load robot model: {str(e)}')
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
        Preprocess point cloud for robot trajectory model inference
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

        # Step 4: Resample to fixed number of points
        if len(points_centered) > self.num_points:
            # Randomly sample points
            indices = np.random.choice(len(points_centered), self.num_points, replace=False)
            points_resampled = points_centered[indices]
        elif len(points_centered) < self.num_points:
            # Duplicate points to reach target
            indices = np.random.choice(len(points_centered), self.num_points, replace=True)
            points_resampled = points_centered[indices]
        else:
            points_resampled = points_centered

        return points_resampled, centroid

    def adjust_arm_lengths(self, robot_coords):
        """Adjust L1 and R1 positions to make arm lengths exactly 40cm while keeping L2 and R2 fixed"""
        adjusted_coords = robot_coords.copy()

        # Left arm: Keep L2 (index 1) fixed, adjust L1 (index 0)
        l2_pos = robot_coords[1]  # Left_L2 position (fixed)
        l1_pos = robot_coords[0]  # Left_L1 position (to be adjusted)

        # Calculate direction from L2 to L1
        l_direction = l1_pos - l2_pos
        l_current_length = np.linalg.norm(l_direction)

        if l_current_length > 0:
            l_direction_normalized = l_direction / l_current_length
            # Set L1 position to be exactly 40cm from L2
            adjusted_coords[0] = l2_pos + l_direction_normalized * self.target_arm_length

        # Right arm: Keep R2 (index 3) fixed, adjust R1 (index 2)
        r2_pos = robot_coords[3]  # Right_R2 position (fixed)
        r1_pos = robot_coords[2]  # Right_R1 position (to be adjusted)

        # Calculate direction from R2 to R1
        r_direction = r1_pos - r2_pos
        r_current_length = np.linalg.norm(r_direction)

        if r_current_length > 0:
            r_direction_normalized = r_direction / r_current_length
            # Set R1 position to be exactly 40cm from R2
            adjusted_coords[2] = r2_pos + r_direction_normalized * self.target_arm_length

        return adjusted_coords

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def calculate_target_poses(self, robot_coords):
        """Calculate target poses with midpoint as origin and direction from 2→1"""
        poses = []

        # Left arm pose
        l1_pos = robot_coords[0]  # Left_L1
        l2_pos = robot_coords[1]  # Left_L2
        left_midpoint = (l1_pos + l2_pos) / 2
        left_direction = l1_pos - l2_pos  # Direction from L2 to L1
        left_direction_normalized = left_direction / np.linalg.norm(left_direction)

        # Right arm pose
        r1_pos = robot_coords[2]  # Right_R1
        r2_pos = robot_coords[3]  # Right_R2
        right_midpoint = (r1_pos + r2_pos) / 2
        right_direction = r1_pos - r2_pos  # Direction from R2 to R1
        right_direction_normalized = right_direction / np.linalg.norm(right_direction)

        # Create pose data: [position, direction]
        poses.append({
            'position': left_midpoint,
            'direction': left_direction_normalized,
            'name': 'left_arm'
        })
        poses.append({
            'position': right_midpoint,
            'direction': right_direction_normalized,
            'name': 'right_arm'
        })

        return poses

    def smooth_poses(self, current_poses):
        """Apply smoothing to poses using rolling average"""
        # Add current poses to buffer
        self.pose_buffer.append(current_poses)

        # Keep buffer size within limit
        if len(self.pose_buffer) > self.pose_buffer_size:
            self.pose_buffer.pop(0)

        # If we don't have enough samples, return current poses
        if len(self.pose_buffer) < 2:
            return current_poses

        # Calculate smoothed poses
        smoothed_poses = []
        for i in range(len(current_poses)):
            # Average positions
            positions = np.array([poses[i]['position'] for poses in self.pose_buffer])
            avg_position = np.mean(positions, axis=0)

            # Average directions (need to be careful with unit vectors)
            directions = np.array([poses[i]['direction'] for poses in self.pose_buffer])
            avg_direction = np.mean(directions, axis=0)
            avg_direction_normalized = avg_direction / np.linalg.norm(avg_direction)

            smoothed_poses.append({
                'position': avg_position,
                'direction': avg_direction_normalized,
                'name': current_poses[i]['name']
            })

        return smoothed_poses

    def direction_to_quaternion(self, direction_vector):
        """Convert a direction vector to a quaternion representing rotation from +X axis"""
        # Normalize the direction vector
        direction = direction_vector / (np.linalg.norm(direction_vector) + 1e-8)

        # Default forward direction is +X axis
        forward = np.array([1.0, 0.0, 0.0])

        # Calculate rotation axis (cross product)
        axis = np.cross(forward, direction)
        axis_length = np.linalg.norm(axis)

        if axis_length < 1e-6:  # Vectors are parallel
            if np.dot(forward, direction) > 0:
                # Same direction
                return [0.0, 0.0, 0.0, 1.0]
            else:
                # Opposite direction, rotate 180 degrees around Y axis
                return [0.0, 1.0, 0.0, 0.0]

        # Normalize rotation axis
        axis = axis / axis_length

        # Calculate rotation angle
        angle = np.arccos(np.clip(np.dot(forward, direction), -1.0, 1.0))

        # Convert to quaternion
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        return [
            float(axis[0] * sin_half),
            float(axis[1] * sin_half),
            float(axis[2] * sin_half),
            float(cos_half)
        ]

    def create_pose_array_msg(self, poses, timestamp):
        """Create PoseArray message from pose data"""
        pose_array = PoseArray()
        pose_array.header.frame_id = 'world'
        pose_array.header.stamp = timestamp

        for pose_data in poses:
            pose = Pose()

            # Set position
            pose.position.x = float(pose_data['position'][0])
            pose.position.y = float(pose_data['position'][1])
            pose.position.z = float(pose_data['position'][2])

            # Convert direction to quaternion
            quaternion = self.direction_to_quaternion(pose_data['direction'])
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            pose_array.poses.append(pose)

        return pose_array

    def postprocess_robot_coords(self, robot_coords_pred, centroid):
        """
        Postprocess predicted robot coordinates back to original coordinate system
        1. Add back the centroid
        2. Convert from millimeters to meters
        3. Rotate back around x-axis by -90 degrees
        """
        # Reshape from (12,) to (4, 3) - 4 robot points, 3 coordinates each
        robot_coords_reshaped = robot_coords_pred.reshape(4, 3)

        # Step 1: Add back centroid
        robot_coords_world = robot_coords_reshaped + centroid.reshape(1, 3)

        # Step 2: Convert from millimeters to meters
        robot_coords_m = robot_coords_world / 1000.0

        # Step 3: Rotate back around x-axis by -90 degrees (clockwise)
        # Rotation matrix for -90 degrees around x-axis (inverse of preprocessing)
        rotation_x_neg90 = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)

        robot_coords_original = np.dot(robot_coords_m, rotation_x_neg90.T)

        # Adjust arm lengths to 40cm
        robot_coords_adjusted = self.adjust_arm_lengths(robot_coords_original)

        return robot_coords_adjusted

    def update_frame_buffer(self, preprocessed_points):
        """Update frame buffer with new preprocessed point cloud"""
        # Add new frame to buffer
        self.frame_buffer.append(preprocessed_points)

        # Keep buffer size within limit
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)  # Remove oldest frame

        # If buffer is not full, duplicate the current frame to fill it
        if len(self.frame_buffer) < self.frames_per_clip:
            # Pad with repeated current frame
            frames_needed = self.frames_per_clip - len(self.frame_buffer)
            padded_frames = [preprocessed_points] * frames_needed + self.frame_buffer
            return np.array(padded_frames)
        else:
            # Use recent frames
            return np.array(self.frame_buffer[-self.frames_per_clip:])

    def predict_robot_trajectory(self, points):
        """Run robot trajectory model inference on preprocessed point cloud"""
        start_time = time.time()

        try:
            # Preprocess
            preprocess_start = time.time()
            preprocessed = self.preprocess_pointcloud(points)
            if preprocessed is None:
                return None
            preprocess_time = time.time() - preprocess_start

            preprocessed_points, centroid = preprocessed

            # Update frame buffer for temporal consistency
            buffer_start = time.time()
            clip_frames = self.update_frame_buffer(preprocessed_points)
            buffer_time = time.time() - buffer_start

            # Convert to tensor and add batch dimension
            tensor_start = time.time()
            # SPiKE robot model expects (batch_size, frames_per_clip, num_points, 3)
            input_tensor = torch.from_numpy(clip_frames).unsqueeze(0).to(self.device)  # (1, 16, 2048, 3)
            tensor_time = time.time() - tensor_start

            # Debug info
            self.get_logger().debug(f'Input points shape: {points.shape}')
            self.get_logger().debug(f'Preprocessed points shape: {preprocessed_points.shape}')
            self.get_logger().debug(f'Clip frames shape: {clip_frames.shape}')
            self.get_logger().debug(f'Input tensor shape: {input_tensor.shape}')

            # Run inference
            inference_start = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
                # Output should be (1, 12) - batch_size=1, 12 coordinates (4 points × 3 coords)
                robot_coords_pred = output.squeeze(0).cpu().numpy()  # Should be (12,)
            inference_time = time.time() - inference_start

            # Postprocess back to original coordinate system
            postprocess_start = time.time()
            robot_coords_original = self.postprocess_robot_coords(robot_coords_pred, centroid)
            postprocess_time = time.time() - postprocess_start

            # Calculate total time
            total_time = time.time() - start_time

            # Print timing information
            self.get_logger().info(f'Robot trajectory inference timing: '
                                  f'Total={total_time*1000:.1f}ms '
                                  f'(Preprocess={preprocess_time*1000:.1f}ms, '
                                  f'Buffer={buffer_time*1000:.1f}ms, '
                                  f'Tensor={tensor_time*1000:.1f}ms, '
                                  f'Inference={inference_time*1000:.1f}ms, '
                                  f'Postprocess={postprocess_time*1000:.1f}ms)')

            return robot_coords_original  # Shape: (4, 3)

        except Exception as e:
            total_time = time.time() - start_time
            self.get_logger().error(f'Robot trajectory prediction failed after {total_time*1000:.1f}ms: {str(e)}')
            return None

    def create_robot_arm_marker_array(self, robot_coords, timestamp):
        """Create MarkerArray message for robot arm visualization"""
        marker_array = MarkerArray()

        # Robot arm connections (representing arm segments)
        arm_connections = [
            (0, 1),  # Left_L1 -> Left_L2 (left arm)
            (2, 3),  # Right_R1 -> Right_R2 (right arm)
        ]

        # Create point markers for each robot arm point
        colors = [
            [1.0, 0.0, 0.0],  # Red for Left_L1
            [0.8, 0.0, 0.2],  # Dark red for Left_L2
            [0.0, 0.0, 1.0],  # Blue for Right_R1
            [0.0, 0.2, 0.8],  # Dark blue for Right_R2
        ]

        for i, (point_coord, color) in enumerate(zip(robot_coords, colors)):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = timestamp
            marker.ns = 'robot_points'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position
            marker.pose.position.x = float(point_coord[0])
            marker.pose.position.y = float(point_coord[1])
            marker.pose.position.z = float(point_coord[2])
            marker.pose.orientation.w = 1.0

            # Scale (sphere size)
            marker.scale.x = 0.05  # 5cm diameter
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            # Color
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            # Lifetime
            marker.lifetime.sec = 1  # 1 second lifetime

            marker_array.markers.append(marker)

        # Create arm segment markers (lines)
        arm_colors = [
            [1.0, 0.5, 0.5],  # Light red for left arm
            [0.5, 0.5, 1.0],  # Light blue for right arm
        ]

        for i, ((start_idx, end_idx), color) in enumerate(zip(arm_connections, arm_colors)):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = timestamp
            marker.ns = 'robot_arms'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Add start and end points
            start_point = Point()
            start_point.x = float(robot_coords[start_idx][0])
            start_point.y = float(robot_coords[start_idx][1])
            start_point.z = float(robot_coords[start_idx][2])

            end_point = Point()
            end_point.x = float(robot_coords[end_idx][0])
            end_point.y = float(robot_coords[end_idx][1])
            end_point.z = float(robot_coords[end_idx][2])

            marker.points = [start_point, end_point]

            # Line width
            marker.scale.x = 0.02  # 2cm line width

            # Color
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8

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

            self.get_logger().debug(f'Processing point cloud with {len(points)} points for robot trajectory prediction')

            # Predict robot trajectory
            robot_coords = self.predict_robot_trajectory(points)

            if robot_coords is not None:
                # Calculate target poses (midpoint + direction from 2→1)
                target_poses = self.calculate_target_poses(robot_coords)

                # Apply smoothing
                smoothed_poses = self.smooth_poses(target_poses)

                # Create and publish pose array message
                pose_array_msg = self.create_pose_array_msg(smoothed_poses, msg.header.stamp)
                self.pose_publisher.publish(pose_array_msg)

                # Optional: publish debug visualization
                marker_msg = self.create_robot_arm_marker_array(robot_coords, msg.header.stamp)
                self.robot_publisher.publish(marker_msg)

                # Log validation (reduce log spam by logging every N frames)
                self.log_counter += 1

                # Detailed logging every N frames
                if self.log_counter % self.log_interval == 0:
                    self.get_logger().info(f'Published {len(smoothed_poses)} target poses to /target_poses')

                    for i, pose in enumerate(pose_array_msg.poses):
                        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                        pos = [pose.position.x, pose.position.y, pose.position.z]
                        arm_name = "Left" if i == 0 else "Right"

                        # Log quaternion
                        self.get_logger().info(f"{arm_name} Arm: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                                              f"quat=[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")

                        # Calculate and log Euler angles for verification
                        roll, pitch, yaw = self.quaternion_to_euler(quat)
                        self.get_logger().info(f"{arm_name} Arm Euler: Roll={np.degrees(roll):.2f}°, "
                                              f"Pitch={np.degrees(pitch):.2f}°, Yaw={np.degrees(yaw):.2f}°")

                        # Log arm direction for debugging
                        pose_data = smoothed_poses[i]
                        direction = pose_data['direction']
                        self.get_logger().debug(f"{arm_name} Arm direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
            else:
                self.get_logger().warn('Failed to predict robot trajectory')

        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {str(e)}')


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = RobotPlannerDirect()
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