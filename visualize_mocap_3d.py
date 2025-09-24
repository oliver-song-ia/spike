"""
3D visualization of processed Mocap point clouds and pose annotations
Includes visualization of left and right arm coordinates
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse
import os
import glob
from const import skeleton_joints

def load_arm_coordinates(session_path):
    """
    Load arm coordinates data if available

    Args:
        session_path: Session directory path

    Returns:
        dict: Dictionary with 'left_arm' and 'right_arm' coordinates, or None if not found
    """
    arm_file = os.path.join(session_path, "arm_coordinates.h5")

    if not os.path.exists(arm_file):
        return None

    try:
        with h5py.File(arm_file, 'r') as f:
            print(f"Found arm coordinates file with keys: {list(f.keys())}")

            arm_data = {}
            for key in f.keys():
                data = f[key][:]
                arm_data[key] = data
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

            return arm_data
    except Exception as e:
        print(f"Error loading arm coordinates: {e}")
        return None

def load_mocap_session_data(session_path, data_format="spike"):
    """
    Load single Mocap session data including arm coordinates

    Args:
        session_path: Session directory path
        data_format: Data format ("spike" or "itop")

    Returns:
        tuple: (point_clouds_list, joints_coords_array, frame_ids_list, is_valid_flags, arm_data)
    """
    pointclouds_dir = os.path.join(session_path, "pointclouds")
    labels_file = os.path.join(session_path, "labels.h5")

    if not os.path.exists(pointclouds_dir) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing required files: {session_path}")

    # Load label data
    with h5py.File(labels_file, 'r') as f:
        joints_coords = f['real_world_coordinates'][:]
        frame_ids = f['id'][:]
        is_valid = f['is_valid'][:]

    print(f"Label data: {len(joints_coords)} frames")
    print(f"Data format: {data_format}")

    # Load arm coordinates if available
    arm_data = load_arm_coordinates(session_path)

    # Load point cloud data
    pc_files = sorted([f for f in os.listdir(pointclouds_dir) if f.endswith('.npz')])
    point_clouds = []

    for i, pc_file in enumerate(pc_files[:len(joints_coords)]):
        pc_path = os.path.join(pointclouds_dir, pc_file)
        pc_data = np.load(pc_path)
        point_clouds.append(pc_data['arr_0'])

    print(f"Point cloud data: {len(point_clouds)} frames")

    # Display data statistics
    if point_clouds:
        first_pc = point_clouds[0]
        first_joints = joints_coords[0]

        print(f"Data statistics:")
        print(f"  Point cloud: shape={first_pc.shape}, dtype={first_pc.dtype}")
        print(f"  Joints: shape={first_joints.shape}, dtype={first_joints.dtype}")

        if data_format == "itop":
            print(f"  Point cloud coordinate range:")
            print(f"    X: [{first_pc[:, 0].min():.1f}, {first_pc[:, 0].max():.1f}]")
            print(f"    Y: [{first_pc[:, 1].min():.1f}, {first_pc[:, 1].max():.1f}]")
            print(f"    Z: [{first_pc[:, 2].min():.1f}, {first_pc[:, 2].max():.1f}]")
        else:
            print(f"  Point cloud coordinate range:")
            print(f"    X: [{first_pc[:, 0].min():.3f}, {first_pc[:, 0].max():.3f}]")
            print(f"    Y: [{first_pc[:, 1].min():.3f}, {first_pc[:, 1].max():.3f}]")
            print(f"    Z: [{first_pc[:, 2].min():.3f}, {first_pc[:, 2].max():.3f}]")

        if arm_data:
            print(f"  Arm data available: {list(arm_data.keys())}")

    return point_clouds, joints_coords, frame_ids, is_valid, arm_data

def visualize_single_frame(point_cloud, joints, frame_id, arm_data=None, frame_idx=0, title="Mocap 3D Visualization"):
    """
    Visualize single frame point cloud, pose, and arm coordinates

    Args:
        point_cloud: Point cloud data (N, 3)
        joints: Joint coordinates (15, 3)
        frame_id: Frame ID
        arm_data: Dictionary with arm coordinate data
        frame_idx: Frame index for arm data
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Display point cloud (downsample for performance)
    if len(point_cloud) > 5000:
        indices = np.random.choice(len(point_cloud), 5000, replace=False)
        pc_sample = point_cloud[indices]
    else:
        pc_sample = point_cloud

    ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2],
               s=0.5, alpha=0.4, c='lightblue', depthshade=False, label='Point Cloud')

    # Display main body joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=80, c='red', alpha=0.8, label='Body Joints', edgecolors='black')

    # Add joint name labels (only for key joints to avoid clutter)
    key_joints = [0, 1, 8, 2, 3, 6, 7]  # Head, Neck, Torso, Shoulders, Hands
    for i in key_joints:
        if i < len(joints):
            joint_name = skeleton_joints.joint_indices.get(i, f"Joint{i}")
            ax.text(joints[i, 0], joints[i, 1], joints[i, 2],
                   f'{i}:{joint_name}', fontsize=8, alpha=0.9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    # Draw skeleton connections
    for j1, j2, color in skeleton_joints.joint_connections:
        if j1 < len(joints) and j2 < len(joints):
            ax.plot([joints[j1, 0], joints[j2, 0]],
                   [joints[j1, 1], joints[j2, 1]],
                   [joints[j1, 2], joints[j2, 2]],
                   color=color, linewidth=3, alpha=0.8)

    # Display arm coordinates if available
    all_points = [pc_sample, joints]

    if arm_data and frame_idx < len(list(arm_data.values())[0]):
        print(f"Visualizing arm data for frame {frame_idx}")

        for arm_name, arm_coords in arm_data.items():
            if frame_idx < len(arm_coords):
                current_arm = arm_coords[frame_idx]

                # Determine color based on arm type
                if 'left' in arm_name.lower():
                    color = 'green'
                    marker = 's'  # square
                    label = f'Left Arm ({arm_name})'
                elif 'right' in arm_name.lower():
                    color = 'blue'
                    marker = '^'  # triangle
                    label = f'Right Arm ({arm_name})'
                else:
                    color = 'purple'
                    marker = 'o'
                    label = f'Arm ({arm_name})'

                # Handle different data shapes
                if current_arm.ndim == 2 and current_arm.shape[1] == 3:
                    # Multiple points per arm
                    ax.scatter(current_arm[:, 0], current_arm[:, 1], current_arm[:, 2],
                             s=60, c=color, alpha=0.7, marker=marker,
                             label=label, edgecolors='black')

                    # Draw connections between arm points
                    for i in range(len(current_arm) - 1):
                        ax.plot([current_arm[i, 0], current_arm[i+1, 0]],
                               [current_arm[i, 1], current_arm[i+1, 1]],
                               [current_arm[i, 2], current_arm[i+1, 2]],
                               color=color, linewidth=2, alpha=0.6)

                    all_points.append(current_arm)

                elif current_arm.ndim == 1 and len(current_arm) >= 3:
                    # Single point (x, y, z)
                    ax.scatter([current_arm[0]], [current_arm[1]], [current_arm[2]],
                             s=100, c=color, alpha=0.8, marker=marker,
                             label=label, edgecolors='black')

                    all_points.append(current_arm[:3].reshape(1, 3))

    # Set coordinate axes
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{title} - Frame {frame_id}')
    ax.legend()

    # Set equal aspect ratio
    if all_points:
        combined_points = np.vstack(all_points)
        set_equal_aspect_3d(ax, combined_points)

    plt.tight_layout()
    plt.show()

def create_animation(point_clouds, joints_coords, frame_ids, arm_data=None, num_frames=50, interval=200):
    """
    Create animation visualization with arm coordinates

    Args:
        point_clouds: Point cloud list
        joints_coords: Joint coordinates array (N, 15, 3)
        frame_ids: Frame ID list
        arm_data: Dictionary with arm coordinate data
        num_frames: Number of animation frames
        interval: Frame interval (ms)
    """
    print(f"Creating animation with {min(num_frames, len(point_clouds))} frames...")

    # 计算所有数据的边界以固定坐标轴
    all_joints = joints_coords[:num_frames].reshape(-1, 3)
    all_pc_points = []
    for i in range(min(num_frames, len(point_clouds))):
        pc = point_clouds[i]
        if len(pc) > 1000:
            indices = np.random.choice(len(pc), 1000, replace=False)
            all_pc_points.append(pc[indices])
        else:
            all_pc_points.append(pc)

    if all_pc_points:
        all_pc_combined = np.vstack(all_pc_points)
        all_data = np.vstack([all_joints, all_pc_combined])
    else:
        all_data = all_joints

    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 设置固定的坐标轴范围
    x_range = [all_data[:, 0].min() - 0.1, all_data[:, 0].max() + 0.1]
    y_range = [all_data[:, 1].min() - 0.1, all_data[:, 1].max() + 0.1]
    z_range = [all_data[:, 2].min() - 0.1, all_data[:, 2].max() + 0.1]

    def update_frame(frame_idx):
        ax.clear()

        # 重新设置坐标轴
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        if frame_idx >= len(point_clouds) or frame_idx >= len(joints_coords):
            return

        # 当前帧数据
        current_pc = point_clouds[frame_idx]
        current_joints = joints_coords[frame_idx]
        current_id = frame_ids[frame_idx]

        # 降采样点云
        if len(current_pc) > 3000:
            indices = np.random.choice(len(current_pc), 3000, replace=False)
            pc_sample = current_pc[indices]
        else:
            pc_sample = current_pc

        # 绘制点云
        ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2],
                   s=0.3, alpha=0.3, c='lightblue', depthshade=False)

        # 绘制关节点
        ax.scatter(current_joints[:, 0], current_joints[:, 1], current_joints[:, 2],
                   s=60, c='red', alpha=0.9, edgecolors='black')

        # 绘制骨架连接线
        for j1, j2, color in skeleton_joints.joint_connections:
            if j1 < len(current_joints) and j2 < len(current_joints):
                ax.plot([current_joints[j1, 0], current_joints[j2, 0]],
                       [current_joints[j1, 1], current_joints[j2, 1]],
                       [current_joints[j1, 2], current_joints[j2, 2]],
                       color=color, linewidth=2, alpha=0.8)

        # 绘制arm数据 (如果可用)
        if arm_data and frame_idx < len(list(arm_data.values())[0]):
            for arm_name, arm_coords in arm_data.items():
                if frame_idx < len(arm_coords):
                    current_arm = arm_coords[frame_idx]

                    if 'left' in arm_name.lower():
                        color = 'green'
                        marker = 's'
                    elif 'right' in arm_name.lower():
                        color = 'blue'
                        marker = '^'
                    else:
                        color = 'purple'
                        marker = 'o'

                    if current_arm.ndim == 2 and current_arm.shape[1] == 3:
                        ax.scatter(current_arm[:, 0], current_arm[:, 1], current_arm[:, 2],
                                 s=40, c=color, alpha=0.7, marker=marker, edgecolors='black')
                    elif current_arm.ndim == 1 and len(current_arm) >= 3:
                        ax.scatter([current_arm[0]], [current_arm[1]], [current_arm[2]],
                                 s=60, c=color, alpha=0.8, marker=marker, edgecolors='black')

        # 设置标题
        ax.set_title(f'Mocap Animation - Frame {frame_idx+1}/{min(num_frames, len(point_clouds))} (ID: {current_id})',
                    fontsize=12)

    # 创建动画
    actual_frames = min(num_frames, len(point_clouds), len(joints_coords))
    anim = FuncAnimation(fig, update_frame, frames=actual_frames,
                        interval=interval, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    return anim

def set_equal_aspect_3d(ax, data):
    """设置3D图的等比例显示"""
    # 计算数据范围
    x_range = [data[:, 0].min(), data[:, 0].max()]
    y_range = [data[:, 1].min(), data[:, 1].max()]
    z_range = [data[:, 2].min(), data[:, 2].max()]

    # 计算最大范围
    max_range = max(x_range[1]-x_range[0],
                   y_range[1]-y_range[0],
                   z_range[1]-z_range[0])

    # 调整显示比例
    mid_x = sum(x_range) / 2
    mid_y = sum(y_range) / 2
    mid_z = sum(z_range) / 2

    ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])

def list_available_sessions(data_dir):
    """列出可用的会话目录"""
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return []

    sessions = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    sessions.sort()

    print(f"\n📁 可用的会话 ({len(sessions)}个):")
    for i, session in enumerate(sessions):
        print(f"  {i}: {session}")

    return sessions

def main():
    parser = argparse.ArgumentParser(description="3D可视化处理后的Mocap数据")

    parser.add_argument("--data-dir",
                       default="/home/oliver/Documents/data/Mocap/itop_format",
                       help="spike_format数据目录路径")
    parser.add_argument("--session", type=str,
                       help="指定会话名称或索引")
    parser.add_argument("--frame", type=int, default=0,
                       help="单帧模式：指定帧索引")
    parser.add_argument("--mode", choices=["single", "animation"], default="single",
                       help="可视化模式: single(单帧) 或 animation(动画)")
    parser.add_argument("--frames", type=int, default=50,
                       help="动画模式：显示的帧数")
    parser.add_argument("--interval", type=int, default=200,
                       help="动画模式：帧间隔(ms)")

    args = parser.parse_args()

    # 列出可用会话
    sessions = list_available_sessions(args.data_dir)
    if not sessions:
        return

    # 选择会话
    if args.session is None:
        print("\n请指定要可视化的会话:")
        print("使用 --session <会话名称> 或 --session <索引>")
        return

    # 解析会话参数
    try:
        session_idx = int(args.session)
        if 0 <= session_idx < len(sessions):
            session_name = sessions[session_idx]
        else:
            print(f"❌ 会话索引超出范围: {session_idx}")
            return
    except ValueError:
        session_name = args.session
        if session_name not in sessions:
            print(f"❌ 会话不存在: {session_name}")
            return

    session_path = os.path.join(args.data_dir, session_name)
    print(f"\nLoading session: {session_name}")

    try:
        # 加载数据
        point_clouds, joints_coords, frame_ids, is_valid, arm_data = load_mocap_session_data(session_path)

        if args.mode == "single":
            # 单帧可视化
            if args.frame >= len(point_clouds) or args.frame >= len(joints_coords):
                print(f"Frame index out of range: {args.frame}")
                return

            print(f"Visualizing frame {args.frame}...")
            visualize_single_frame(point_clouds[args.frame],
                                 joints_coords[args.frame],
                                 frame_ids[args.frame],
                                 arm_data,
                                 args.frame,
                                 f"Session: {session_name}")

        elif args.mode == "animation":
            # 动画可视化
            print(f"Creating animation visualization...")
            anim = create_animation(point_clouds, joints_coords, frame_ids, arm_data,
                                  args.frames, args.interval)

    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()