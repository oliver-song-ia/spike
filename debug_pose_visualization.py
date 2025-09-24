#!/usr/bin/env python3
"""
Debug script for visualizing preprocessed point cloud and predicted skeleton
in the model's coordinate system (after preprocessing, before postprocessing)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
import os
import sys
import time
import threading
import queue
import struct

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available, using simulated data")
    ROS2_AVAILABLE = False

from model import model_builder
from utils.config_utils import load_config


class ROS2PointCloudSubscriber(Node):
    """ROS2 subscriber for point cloud data"""

    def __init__(self, data_queue):
        super().__init__('debug_pointcloud_subscriber')
        self.data_queue = data_queue

        self.subscription = self.create_subscription(
            PointCloud2,
            '/human_pointcloud',
            self.pointcloud_callback,
            10
        )

    def pointcloud_to_numpy(self, pc_msg):
        """Convert ROS2 PointCloud2 message to numpy array"""
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
                    points.append([x * 1000, y * 1000, z * 1000])  # Convert m to mm

        return np.array(points, dtype=np.float32)

    def pointcloud_callback(self, msg):
        """Callback for point cloud messages"""
        try:
            points = self.pointcloud_to_numpy(msg)
            if len(points) > 100:  # Minimum point threshold
                # Put data in queue (non-blocking)
                try:
                    self.data_queue.put(points, block=False)
                except queue.Full:
                    # Remove old data if queue is full
                    try:
                        self.data_queue.get(block=False)
                        self.data_queue.put(points, block=False)
                    except queue.Empty:
                        pass
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')


class PoseDebugVisualizer:
    """Debug visualizer for real-time pose detection"""

    def __init__(self, config_path='experiments/Custom/1', model_path='experiments/Custom/1/log/best_model.pth'):
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Joint names for upper body (9 joints)
        self.joint_names = [
            'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
            'R_Hand', 'L_Hand', 'Torso'
        ]

        # Skeleton connections for upper body
        self.skeleton_connections = [
            (0, 1),   # Head -> Neck
            (1, 2),   # Neck -> R_Shoulder
            (1, 3),   # Neck -> L_Shoulder
            (2, 4),   # R_Shoulder -> R_Elbow
            (3, 5),   # L_Shoulder -> L_Elbow
            (4, 6),   # R_Elbow -> R_Hand
            (5, 7),   # L_Elbow -> L_Hand
            (1, 8),   # Neck -> Torso
        ]

        # Data queue for real-time updates
        self.data_queue = queue.Queue(maxsize=5)

        # Current data
        self.current_points = None
        self.current_joints = None

        # Animation setup
        self.fig = None
        self.ax = None
        self.point_scatter = None
        self.joint_scatter = None
        self.bone_lines = []

        # ROS2 setup
        self.ros_node = None
        self.ros_thread = None

        # Animation settings
        self.animation_interval = 50  # Default 50ms = 20 FPS

        self.load_model()

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
                print(f'Model loaded from: {self.model_path}')
            else:
                raise FileNotFoundError(f'Model file not found: {self.model_path}')

        except Exception as e:
            print(f'Failed to load model: {str(e)}')
            raise

    def generate_test_pointcloud(self, num_points=2048):
        """Generate a test point cloud (random human-like shape)"""
        # Create a simple human-like point cloud for testing
        points = []

        # Torso (cylinder)
        for _ in range(num_points // 3):
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 200)  # 200mm radius
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(-400, 400)  # 800mm height
            points.append([x, y, z])

        # Arms (random points around shoulders)
        for _ in range(num_points // 3):
            # Left arm
            x = np.random.uniform(-150, -350)
            y = np.random.uniform(-50, 50)
            z = np.random.uniform(200, 400)
            points.append([x, y, z])

            # Right arm
            x = np.random.uniform(150, 350)
            y = np.random.uniform(-50, 50)
            z = np.random.uniform(200, 400)
            points.append([x, y, z])

        # Head
        for _ in range(num_points // 6):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, 100)  # 100mm radius
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = 500 + r * np.cos(phi)  # Head at top
            points.append([x, y, z])

        return np.array(points, dtype=np.float32)

    def preprocess_pointcloud(self, points):
        """
        Preprocess point cloud for model inference (same as in pose_detector.py)
        1. Rotate around x-axis by 90 degrees
        2. Convert from meters to millimeters (skip since already in mm)
        3. Center the point cloud
        4. Resample to fixed number of points
        """
        if len(points) == 0:
            return None

        print(f"Original points shape: {points.shape}")
        print(f"Original points range: x=[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}], "
              f"y=[{points[:, 1].min():.1f}, {points[:, 1].max():.1f}], "
              f"z=[{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]")

        # Step 1: Rotate around x-axis by 90 degrees (counter-clockwise)
        rotation_x_90 = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

        points_rotated = np.dot(points, rotation_x_90.T)
        print(f"After rotation shape: {points_rotated.shape}")
        print(f"After rotation range: x=[{points_rotated[:, 0].min():.1f}, {points_rotated[:, 0].max():.1f}], "
              f"y=[{points_rotated[:, 1].min():.1f}, {points_rotated[:, 1].max():.1f}], "
              f"z=[{points_rotated[:, 2].min():.1f}, {points_rotated[:, 2].max():.1f}]")

        # Step 2: Already in millimeters, skip conversion

        # Step 3: Center the point cloud (subtract centroid)
        centroid = np.mean(points_rotated, axis=0)
        points_centered = points_rotated - centroid
        print(f"Centroid: {centroid}")
        print(f"After centering range: x=[{points_centered[:, 0].min():.1f}, {points_centered[:, 0].max():.1f}], "
              f"y=[{points_centered[:, 1].min():.1f}, {points_centered[:, 1].max():.1f}], "
              f"z=[{points_centered[:, 2].min():.1f}, {points_centered[:, 2].max():.1f}]")

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

        print(f"Final preprocessed shape: {points_resampled.shape}")
        return points_resampled, centroid

    def predict_pose(self, points):
        """Run model inference on preprocessed point cloud"""
        try:
            # Preprocess
            preprocessed = self.preprocess_pointcloud(points)
            if preprocessed is None:
                return None, None

            points_tensor, centroid = preprocessed

            # Convert to tensor and add batch and temporal dimensions
            # SPiKE expects (batch_size, frames_per_clip, num_points, 3)
            frames_per_clip = 16
            points_expanded = np.repeat(points_tensor[np.newaxis, :, :], frames_per_clip, axis=0)
            input_tensor = torch.from_numpy(points_expanded).unsqueeze(0).to(self.device)

            print(f"Input tensor shape: {input_tensor.shape}")

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)

                # Reshape output from (1, 45) to (15, 3) - 15 joints, 3 coordinates each
                joints_pred = output.reshape(-1, 3).cpu().numpy()  # Should be (15, 3)

            print(f"Predicted joints shape: {joints_pred.shape}")
            print(f"Predicted joints range: x=[{joints_pred[:, 0].min():.1f}, {joints_pred[:, 0].max():.1f}], "
                  f"y=[{joints_pred[:, 1].min():.1f}, {joints_pred[:, 1].max():.1f}], "
                  f"z=[{joints_pred[:, 2].min():.1f}, {joints_pred[:, 2].max():.1f}]")

            # For debug visualization, keep joints in same coordinate system as points
            # Option 1: Return both in centered coordinate system (subtract centroid)
            # joints_centered = joints_pred  # Keep joints centered like points
            # return points_tensor, joints_centered

            # Option 2: Return both in world coordinate system (add centroid back to points)
            points_world = points_tensor + centroid.reshape(1, 3)
            joints_world = joints_pred + centroid.reshape(1, 3)

            return points_world, joints_world  # Both in world coordinate system

        except Exception as e:
            print(f'Prediction failed: {str(e)}')
            return None, None

    def visualize_3d(self, points, joints, title="3D Pose Visualization"):
        """Visualize point cloud and skeleton in 3D"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot point cloud
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c='lightblue', alpha=0.3, s=1, label='Point Cloud')

        # Plot skeleton joints and bones
        if joints is not None:
            upper_body_joints = joints[:9]  # Only upper body

            # Plot joints
            ax.scatter(upper_body_joints[:, 0], upper_body_joints[:, 1], upper_body_joints[:, 2],
                      c='red', s=100, label='Joints')

            # Add joint labels
            for i, (joint, name) in enumerate(zip(upper_body_joints, self.joint_names)):
                ax.text(joint[0], joint[1], joint[2], f'  {name}', fontsize=8)

            # Plot bones
            for start_idx, end_idx in self.skeleton_connections:
                if start_idx < len(upper_body_joints) and end_idx < len(upper_body_joints):
                    start_joint = upper_body_joints[start_idx]
                    end_joint = upper_body_joints[end_idx]

                    ax.plot([start_joint[0], end_joint[0]],
                           [start_joint[1], end_joint[1]],
                           [start_joint[2], end_joint[2]],
                           'g-', linewidth=3, label='Bones' if start_idx == 0 else "")

        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend()

        # Set equal aspect ratio
        max_range = 400  # mm
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        plt.tight_layout()
        plt.show()

    def start_ros2_thread(self):
        """Start ROS2 subscriber in separate thread"""
        if not ROS2_AVAILABLE:
            print("ROS2 not available, will use simulated data")
            return

        def ros_thread():
            rclpy.init()
            self.ros_node = ROS2PointCloudSubscriber(self.data_queue)
            try:
                rclpy.spin(self.ros_node)
            except KeyboardInterrupt:
                pass
            finally:
                self.ros_node.destroy_node()
                rclpy.shutdown()

        self.ros_thread = threading.Thread(target=ros_thread, daemon=True)
        self.ros_thread.start()
        print("ROS2 subscriber started, listening to /human_pointcloud")

    def update_data(self):
        """Update data from queue or generate simulated data"""
        if ROS2_AVAILABLE:
            # Try to get latest data from queue
            try:
                points = self.data_queue.get(block=False)
                preprocessed_points, predicted_joints = self.predict_pose(points)
                if preprocessed_points is not None and predicted_joints is not None:
                    self.current_points = preprocessed_points
                    self.current_joints = predicted_joints[:9]  # Upper body only
                    return True
            except queue.Empty:
                pass
        else:
            # Generate simulated data
            points = self.generate_test_pointcloud()
            preprocessed_points, predicted_joints = self.predict_pose(points)
            if preprocessed_points is not None and predicted_joints is not None:
                self.current_points = preprocessed_points
                self.current_joints = predicted_joints[:9]  # Upper body only
                return True

        return False

    def init_animation(self):
        """Initialize animation plot"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set up plot
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('Real-time Pose Detection Debug (Model Coordinate System)')

        # Set fixed limits
        max_range = 600
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

        # Initialize empty plots
        self.point_scatter = self.ax.scatter([], [], [], c='lightblue', alpha=0.3, s=1)
        self.joint_scatter = self.ax.scatter([], [], [], c='red', s=100)

        # Initialize bone lines
        self.bone_lines = []
        for _ in self.skeleton_connections:
            line, = self.ax.plot([], [], [], 'g-', linewidth=3)
            self.bone_lines.append(line)

        return self.point_scatter, self.joint_scatter, *self.bone_lines

    def animate(self, frame):
        """Animation update function"""
        # Update data
        data_updated = self.update_data()

        if data_updated and self.current_points is not None and self.current_joints is not None:
            # Update point cloud
            self.point_scatter._offsets3d = (
                self.current_points[:, 0],
                self.current_points[:, 1],
                self.current_points[:, 2]
            )

            # Update joints
            self.joint_scatter._offsets3d = (
                self.current_joints[:, 0],
                self.current_joints[:, 1],
                self.current_joints[:, 2]
            )

            # Update bones
            for i, (start_idx, end_idx) in enumerate(self.skeleton_connections):
                if start_idx < len(self.current_joints) and end_idx < len(self.current_joints):
                    start_joint = self.current_joints[start_idx]
                    end_joint = self.current_joints[end_idx]

                    self.bone_lines[i].set_data_3d(
                        [start_joint[0], end_joint[0]],
                        [start_joint[1], end_joint[1]],
                        [start_joint[2], end_joint[2]]
                    )

            # Print current frame info
            if frame % 30 == 0:  # Print every 30 frames
                print(f"Frame {frame}: Points={len(self.current_points)}, "
                      f"Joints={len(self.current_joints)}")

        return self.point_scatter, self.joint_scatter, *self.bone_lines

    def run_realtime_animation(self):
        """Run real-time animation"""
        print("=== SPiKE Real-time Pose Detection Debug ===")

        # Start ROS2 subscriber
        self.start_ros2_thread()

        # Wait a moment for ROS2 to start
        time.sleep(1)

        # Initialize animation
        self.init_animation()

        # Create animation
        ani = animation.FuncAnimation(
            self.fig, self.animate,
            init_func=self.init_animation,
            interval=self.animation_interval,
            blit=False,
            cache_frame_data=False
        )

        print("Starting real-time visualization...")
        print("Press Ctrl+C to stop")

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping visualization...")

    def run_debug(self):
        """Run debug visualization (backwards compatibility)"""
        self.run_realtime_animation()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="SPiKE Pose Detection Debug Visualizer")
    parser.add_argument('--config', type=str, default='experiments/Custom/1',
                        help='Path to config directory')
    parser.add_argument('--model', type=str, default='experiments/Custom/1/log/best_model.pth',
                        help='Path to model file')
    parser.add_argument('--mode', type=str, choices=['realtime', 'static'], default='realtime',
                        help='Visualization mode: realtime (animated) or static (single frame)')
    parser.add_argument('--fps', type=int, default=20,
                        help='Target FPS for realtime mode')

    args = parser.parse_args()

    try:
        visualizer = PoseDebugVisualizer(args.config, args.model)

        if args.mode == 'realtime':
            # Update animation interval based on FPS
            visualizer.animation_interval = 1000 // args.fps  # Convert to milliseconds
            visualizer.run_realtime_animation()
        else:
            # Static mode - get one frame from ROS2 and show
            print("=== SPiKE Static Pose Detection Debug ===")
            if ROS2_AVAILABLE:
                print("Waiting for point cloud data from /human_pointcloud...")

                # Start ROS2 subscriber
                visualizer.start_ros2_thread()
                time.sleep(1)  # Wait for ROS2 to start

                # Wait for one frame of data
                timeout = 10  # 10 seconds timeout
                start_time = time.time()
                points = None

                while time.time() - start_time < timeout:
                    try:
                        points = visualizer.data_queue.get(timeout=1)
                        print(f"Received point cloud with {len(points)} points")
                        break
                    except queue.Empty:
                        print("Waiting for data...")
                        continue

                if points is not None:
                    preprocessed_points, predicted_joints = visualizer.predict_pose(points)
                    if preprocessed_points is not None and predicted_joints is not None:
                        visualizer.visualize_3d(preprocessed_points, predicted_joints[:9],
                                              "Static Pose Detection (Model Coordinate System)")
                    else:
                        print("Failed to predict pose!")
                else:
                    print("Timeout: No point cloud data received!")
                    print("Make sure /human_pointcloud topic is publishing data")
            else:
                print("ROS2 not available! Cannot subscribe to point cloud data.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()