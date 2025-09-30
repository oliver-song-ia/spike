#!/usr/bin/env python3
"""
ROS2 Target Poses Comparison Node
Subscribes to both /target_poses and /target_poses_direct topics,
compares their data and analyzes differences to debug IK calculation issues.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import numpy as np
import time
from collections import deque


class TargetPosesComparator(Node):
    """ROS2 node for comparing target poses from different sources"""

    def __init__(self):
        super().__init__('target_poses_comparator')

        # Subscribers
        self.original_subscriber = self.create_subscription(
            PoseArray,
            '/target_poses',
            self.original_poses_callback,
            10
        )

        self.direct_subscriber = self.create_subscription(
            PoseArray,
            '/target_poses_direct',
            self.direct_poses_callback,
            10
        )

        # Data storage
        self.original_poses_buffer = deque(maxlen=10)
        self.direct_poses_buffer = deque(maxlen=10)

        # Timing
        self.last_original_time = None
        self.last_direct_time = None
        self.comparison_timer = self.create_timer(1.0, self.compare_poses)

        self.get_logger().info('Target Poses Comparator node initialized')
        self.get_logger().info('Subscribing to: /target_poses and /target_poses_direct')

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

        return np.array([roll, pitch, yaw])

    def pose_to_dict(self, pose):
        """Convert geometry_msgs/Pose to dictionary with position and orientation"""
        return {
            'position': np.array([pose.position.x, pose.position.y, pose.position.z]),
            'orientation_quat': np.array([pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z, pose.orientation.w]),
            'orientation_euler': self.quaternion_to_euler([pose.orientation.x, pose.orientation.y,
                                                         pose.orientation.z, pose.orientation.w])
        }

    def original_poses_callback(self, msg):
        """Callback for original target poses"""
        self.last_original_time = time.time()

        poses_data = []
        for i, pose in enumerate(msg.poses):
            pose_dict = self.pose_to_dict(pose)
            pose_dict['arm'] = 'left' if i == 0 else 'right'
            poses_data.append(pose_dict)

        self.original_poses_buffer.append({
            'timestamp': self.last_original_time,
            'poses': poses_data,
            'frame_id': msg.header.frame_id
        })

        self.get_logger().debug(f'Received original poses: {len(msg.poses)} poses')

    def direct_poses_callback(self, msg):
        """Callback for direct target poses"""
        self.last_direct_time = time.time()

        poses_data = []
        for i, pose in enumerate(msg.poses):
            pose_dict = self.pose_to_dict(pose)
            pose_dict['arm'] = 'left' if i == 0 else 'right'
            poses_data.append(pose_dict)

        self.direct_poses_buffer.append({
            'timestamp': self.last_direct_time,
            'poses': poses_data,
            'frame_id': msg.header.frame_id
        })

        self.get_logger().debug(f'Received direct poses: {len(msg.poses)} poses')

    def calculate_pose_difference(self, pose1, pose2):
        """Calculate difference between two poses"""
        pos_diff = np.linalg.norm(pose1['position'] - pose2['position'])

        # Angular difference (using Euler angles for easier interpretation)
        euler_diff = np.abs(pose1['orientation_euler'] - pose2['orientation_euler'])
        # Handle angle wrapping
        euler_diff = np.minimum(euler_diff, 2*np.pi - euler_diff)
        angular_diff = np.linalg.norm(euler_diff)

        return {
            'position_diff_magnitude': pos_diff,
            'position_diff_vector': pose1['position'] - pose2['position'],
            'angular_diff_magnitude': angular_diff,
            'angular_diff_euler': euler_diff,
            'orientation_euler_1': pose1['orientation_euler'],
            'orientation_euler_2': pose2['orientation_euler']
        }

    def compare_poses(self):
        """Compare the latest poses from both topics"""
        if not self.original_poses_buffer or not self.direct_poses_buffer:
            if not self.original_poses_buffer:
                self.get_logger().warn('No data from /target_poses yet')
            if not self.direct_poses_buffer:
                self.get_logger().warn('No data from /target_poses_direct yet')
            return

        # Get latest data
        original_data = self.original_poses_buffer[-1]
        direct_data = self.direct_poses_buffer[-1]

        # Check timing
        time_diff = abs(original_data['timestamp'] - direct_data['timestamp'])
        if time_diff > 0.5:  # More than 500ms difference
            self.get_logger().warn(f'Large time difference between topics: {time_diff:.3f}s')

        # Compare poses
        self.get_logger().info('=' * 80)
        self.get_logger().info('TARGET POSES COMPARISON')
        self.get_logger().info('=' * 80)

        # Check number of poses
        orig_count = len(original_data['poses'])
        direct_count = len(direct_data['poses'])

        if orig_count != direct_count:
            self.get_logger().error(f'Pose count mismatch: original={orig_count}, direct={direct_count}')
            return

        # Compare each pose pair
        for i in range(min(orig_count, direct_count)):
            orig_pose = original_data['poses'][i]
            direct_pose = direct_data['poses'][i]

            arm_name = orig_pose['arm']
            diff = self.calculate_pose_difference(orig_pose, direct_pose)

            self.get_logger().info(f'\n--- {arm_name.upper()} ARM COMPARISON ---')

            # Position comparison
            self.get_logger().info('POSITION:')
            self.get_logger().info(f'  Original:  [{orig_pose["position"][0]:.4f}, {orig_pose["position"][1]:.4f}, {orig_pose["position"][2]:.4f}]')
            self.get_logger().info(f'  Direct:    [{direct_pose["position"][0]:.4f}, {direct_pose["position"][1]:.4f}, {direct_pose["position"][2]:.4f}]')
            self.get_logger().info(f'  Difference: [{diff["position_diff_vector"][0]:.4f}, {diff["position_diff_vector"][1]:.4f}, {diff["position_diff_vector"][2]:.4f}]')
            self.get_logger().info(f'  Magnitude:  {diff["position_diff_magnitude"]:.4f} m')

            # Orientation comparison (Euler angles in degrees)
            orig_euler_deg = np.degrees(orig_pose['orientation_euler'])
            direct_euler_deg = np.degrees(direct_pose['orientation_euler'])
            euler_diff_deg = np.degrees(diff['angular_diff_euler'])

            self.get_logger().info('ORIENTATION (Roll, Pitch, Yaw in degrees):')
            self.get_logger().info(f'  Original:  [{orig_euler_deg[0]:.2f}, {orig_euler_deg[1]:.2f}, {orig_euler_deg[2]:.2f}]')
            self.get_logger().info(f'  Direct:    [{direct_euler_deg[0]:.2f}, {direct_euler_deg[1]:.2f}, {direct_euler_deg[2]:.2f}]')
            self.get_logger().info(f'  Difference: [{euler_diff_deg[0]:.2f}, {euler_diff_deg[1]:.2f}, {euler_diff_deg[2]:.2f}]')
            self.get_logger().info(f'  Magnitude:  {np.degrees(diff["angular_diff_magnitude"]):.2f} degrees')

            # Quaternion comparison
            self.get_logger().info('QUATERNION (x, y, z, w):')
            self.get_logger().info(f'  Original:  [{orig_pose["orientation_quat"][0]:.4f}, {orig_pose["orientation_quat"][1]:.4f}, {orig_pose["orientation_quat"][2]:.4f}, {orig_pose["orientation_quat"][3]:.4f}]')
            self.get_logger().info(f'  Direct:    [{direct_pose["orientation_quat"][0]:.4f}, {direct_pose["orientation_quat"][1]:.4f}, {direct_pose["orientation_quat"][2]:.4f}, {direct_pose["orientation_quat"][3]:.4f}]')

            # Assessment
            if diff['position_diff_magnitude'] > 0.05:  # 5cm
                self.get_logger().error(f'  ⚠️  LARGE POSITION DIFFERENCE: {diff["position_diff_magnitude"]:.4f}m > 0.05m')
            elif diff['position_diff_magnitude'] > 0.01:  # 1cm
                self.get_logger().warn(f'  ⚠️  Medium position difference: {diff["position_diff_magnitude"]:.4f}m > 0.01m')
            else:
                self.get_logger().info(f'  ✅ Position difference OK: {diff["position_diff_magnitude"]:.4f}m < 0.01m')

            if np.degrees(diff['angular_diff_magnitude']) > 15:  # 15 degrees
                self.get_logger().error(f'  ⚠️  LARGE ORIENTATION DIFFERENCE: {np.degrees(diff["angular_diff_magnitude"]):.2f}° > 15°')
            elif np.degrees(diff['angular_diff_magnitude']) > 5:  # 5 degrees
                self.get_logger().warn(f'  ⚠️  Medium orientation difference: {np.degrees(diff["angular_diff_magnitude"]):.2f}° > 5°')
            else:
                self.get_logger().info(f'  ✅ Orientation difference OK: {np.degrees(diff["angular_diff_magnitude"]):.2f}° < 5°')

        self.get_logger().info('=' * 80)


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = TargetPosesComparator()
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