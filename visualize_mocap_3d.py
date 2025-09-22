"""
3D可视化处理后的Mocap点云和姿态标注
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

def load_mocap_session_data(session_path, data_format="spike"):
    """
    加载单个Mocap会话的数据

    Args:
        session_path: 会话目录路径
        data_format: 数据格式 ("spike" 或 "itop")

    Returns:
        tuple: (点云列表, 关节坐标数组, 帧ID列表, 有效性标记)
    """
    pointclouds_dir = os.path.join(session_path, "pointclouds")
    labels_file = os.path.join(session_path, "labels.h5")

    if not os.path.exists(pointclouds_dir) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"缺少必要文件: {session_path}")

    # 读取标签数据
    with h5py.File(labels_file, 'r') as f:
        joints_coords = f['real_world_coordinates'][:]
        frame_ids = f['id'][:]
        is_valid = f['is_valid'][:]

    print(f"标签数据: {len(joints_coords)} 帧")
    print(f"数据格式: {data_format}")

    # 读取点云数据
    pc_files = sorted([f for f in os.listdir(pointclouds_dir) if f.endswith('.npz')])
    point_clouds = []

    for i, pc_file in enumerate(pc_files[:len(joints_coords)]):
        pc_path = os.path.join(pointclouds_dir, pc_file)
        pc_data = np.load(pc_path)
        point_clouds.append(pc_data['arr_0'])

    print(f"点云数据: {len(point_clouds)} 帧")

    # 显示数据统计
    if point_clouds:
        first_pc = point_clouds[0]
        first_joints = joints_coords[0]

        print(f"数据统计:")
        print(f"  点云: 形状={first_pc.shape}, 类型={first_pc.dtype}")
        print(f"  关节: 形状={first_joints.shape}, 类型={first_joints.dtype}")

        if data_format == "itop":
            print(f"  点云坐标范围:")
            print(f"    X: [{first_pc[:, 0].min():.1f}, {first_pc[:, 0].max():.1f}]")
            print(f"    Y: [{first_pc[:, 1].min():.1f}, {first_pc[:, 1].max():.1f}]")
            print(f"    Z: [{first_pc[:, 2].min():.1f}, {first_pc[:, 2].max():.1f}]")
        else:
            print(f"  点云坐标范围:")
            print(f"    X: [{first_pc[:, 0].min():.3f}, {first_pc[:, 0].max():.3f}]")
            print(f"    Y: [{first_pc[:, 1].min():.3f}, {first_pc[:, 1].max():.3f}]")
            print(f"    Z: [{first_pc[:, 2].min():.3f}, {first_pc[:, 2].max():.3f}]")

    return point_clouds, joints_coords, frame_ids, is_valid

def visualize_single_frame(point_cloud, joints, frame_id, title="Mocap 3D Visualization"):
    """
    可视化单帧点云和姿态

    Args:
        point_cloud: 点云数据 (N, 3)
        joints: 关节坐标 (15, 3)
        frame_id: 帧ID
        title: 图表标题
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 显示点云（降采样以提高性能）
    if len(point_cloud) > 5000:
        indices = np.random.choice(len(point_cloud), 5000, replace=False)
        pc_sample = point_cloud[indices]
    else:
        pc_sample = point_cloud

    ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2],
               s=0.5, alpha=0.4, c='lightblue', depthshade=False, label='Point Cloud')

    # 显示关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=80, c='red', alpha=0.8, label='Joints', edgecolors='black')

    # 添加关节名称标签
    for i, (x, y, z) in enumerate(joints):
        joint_name = skeleton_joints.joint_indices.get(i, f"Joint{i}")
        ax.text(x, y, z, f'{i}:{joint_name}', fontsize=8, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    # 绘制骨架连接线
    for j1, j2, color in skeleton_joints.joint_connections:
        if j1 < len(joints) and j2 < len(joints):
            ax.plot([joints[j1, 0], joints[j2, 0]],
                   [joints[j1, 1], joints[j2, 1]],
                   [joints[j1, 2], joints[j2, 2]],
                   color=color, linewidth=3, alpha=0.8)

    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{title} - Frame {frame_id}')
    ax.legend()

    # 设置等比例坐标轴
    all_points = np.vstack([pc_sample, joints])
    set_equal_aspect_3d(ax, all_points)

    plt.tight_layout()
    plt.show()

def create_animation(point_clouds, joints_coords, frame_ids, num_frames=50, interval=200):
    """
    创建动画可视化

    Args:
        point_clouds: 点云列表
        joints_coords: 关节坐标数组 (N, 15, 3)
        frame_ids: 帧ID列表
        num_frames: 动画帧数
        interval: 帧间隔(ms)
    """
    print(f"创建 {min(num_frames, len(point_clouds))} 帧动画...")

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
                       default="/home/oliver/Documents/data/Mocap/spike_format",
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
    print(f"\n🎯 加载会话: {session_name}")

    try:
        # 加载数据
        point_clouds, joints_coords, frame_ids, is_valid = load_mocap_session_data(session_path)

        if args.mode == "single":
            # 单帧可视化
            if args.frame >= len(point_clouds) or args.frame >= len(joints_coords):
                print(f"❌ 帧索引超出范围: {args.frame}")
                return

            print(f"可视化第 {args.frame} 帧...")
            visualize_single_frame(point_clouds[args.frame],
                                 joints_coords[args.frame],
                                 frame_ids[args.frame],
                                 f"Session: {session_name}")

        elif args.mode == "animation":
            # 动画可视化
            print(f"创建动画可视化...")
            anim = create_animation(point_clouds, joints_coords, frame_ids,
                                  args.frames, args.interval)

    except Exception as e:
        print(f"❌ 可视化失败: {e}")

if __name__ == "__main__":
    main()