import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_skeleton_data(csv_path):
    """加载 CSV 格式的骨骼数据，去除最后两列标注"""
    try:
        # 读取 CSV 文件并去除最后两列
        df = pd.read_csv(csv_path)
        df = df.iloc[:, :-2]  # 移除最后两列标注
        num_frames = df.shape[0]
        
        # 验证列数是否符合25个节点×3坐标（x,y,z）
        if df.shape[1] != 25 * 3:
            print(f"警告：列数为{df.shape[1]}，预期75列（25节点×3坐标）")
        
        # 重塑为(帧数, 节点数, 3)
        skeleton_data = df.values.reshape(num_frames, 25, 3)
        print(f"成功加载骨骼数据，形状: {skeleton_data.shape}")
        return skeleton_data
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None


def get_skeleton_connections():
    """
    根据25个节点的实际顺序定义骨骼连接关系
    节点顺序：nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
              left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
              left_wrist(9), right_wrist(10), left_pinky(11), right_pinky(12),
              left_index(13), right_index(14), left_hip(15), right_hip(16),
              left_knee(17), right_knee(18), left_ankle(19), right_ankle(20),
              left_heel(21), right_heel(22), left_foot(23), right_foot(24)
    """
    return [
        # 头部连接
        (0, 1),    # 鼻子-左眼
        (0, 2),    # 鼻子-右眼
        (1, 3),    # 左眼-左耳
        (2, 4),    # 右眼-右耳
        
        # 左臂连接
        (5, 7),    # 左肩-左肘
        (7, 9),    # 左肘-左腕
        (9, 11),   # 左腕-左小指
        (9, 13),   # 左腕-左食指
        
        # 右臂连接
        (6, 8),    # 右肩-右肘
        (8, 10),   # 右肘-右腕
        (10, 12),  # 右腕-右小指
        (10, 14),  # 右腕-右食指
        
        # 躯干连接（
        (5, 6),
        (5, 15),   # 左肩-左髋
        (6, 16),   # 右肩-右髋
        (15, 16),  # 左髋-右髋（骨盆）
        
        # 左腿连接
        (15, 17),  # 左髋-左膝
        (17, 19),  # 左膝-左踝
        (19, 21),  # 左踝-左脚跟
        (21, 23),  # 左脚跟-左脚
        
        # 右腿连接
        (16, 18),  # 右髋-右膝
        (18, 20),  # 右膝-右踝
        (20, 22),  # 右踝-右脚跟
        (22, 24)   # 右脚跟-右脚
    ]


def visualize_skeleton(skeleton_data, frame_index=0):
    """可视化指定帧的骨骼数据"""
    if skeleton_data is None:
        return

    # 确保帧索引有效
    if frame_index >= skeleton_data.shape[0]:
        frame_index = 0
        print(f"帧索引超出范围，使用第 0 帧")

    # 获取指定帧数据
    frame_data = skeleton_data[frame_index]
    num_joints = frame_data.shape[0]
    connections = get_skeleton_connections()  # 获取25节点专用连接关系

    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制骨骼连接
    for i, j in connections:
        x = [frame_data[i, 0], frame_data[j, 0]]
        y = [frame_data[i, 1], frame_data[j, 1]]
        z = [frame_data[i, 2], frame_data[j, 2]]
        ax.plot(x, y, z, 'b-', linewidth=2)  # 统一用蓝色连线（均为有效节点）

    # 绘制关节点并标记（显示节点名称缩写）
    node_names = [
        '鼻', '左眼', '右眼', '左耳', '右耳',
        '左肩', '右肩', '左肘', '右肘',
        '左腕', '右腕', '左小指', '右小指',
        '左食指', '右食指', '左髋', '右髋',
        '左膝', '右膝', '左踝', '右踝',
        '左脚跟', '右脚跟', '左脚', '右脚'
    ]

    # 设置坐标轴和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'骨骼关节可视化 (帧索引: {frame_index}, 总节点数: {num_joints})')
    ax.legend()
    plt.show()

    # 打印节点坐标信息
    print("\n节点坐标信息:")
    for i in range(num_joints):
        x, y, z = frame_data[i]
        print(f"{node_names[i]} ({i}): ({x:.2f}, {y:.2f}, {z:.2f})")


def main():
    # CSV文件路径
    csv_file_path = "dataset/MMAD_advanced/MMASD+/3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/Twist_Pose/processed_tw_41004_D1_002_i_4_0.csv"

    # 加载并可视化数据
    skeleton_data = load_skeleton_data(csv_file_path)
    if skeleton_data is not None:
        visualize_skeleton(skeleton_data, frame_index=110)


if __name__ == "__main__":
    main()