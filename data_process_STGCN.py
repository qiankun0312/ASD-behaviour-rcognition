import csv
import numpy as np
import os
import pickle
import random
from tqdm import tqdm
from numpy.lib.format import open_memmap

# 25个关节名称
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "Left_hip", "Right_hip",
    "Left_knee", "Right_knee", "Left_ankle", "Right_ankle",
    "Left_heel", "Right_heel", "Left_foot", "Right_foot"
]

def parse_mmad3d_csv(csv_path):
    """解析单个MMAD3D的CSV文件，返回3D骨骼数据(T, V, C)"""
    skeleton_frames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # 按表头读取字段
        for row in reader:
            frame_data = []
            for joint in JOINT_NAMES:
                # 提取x、y、z坐标（字段名如"nose_x"）
                x = float(row.get(f"{joint}_x", 0.0))
                y = float(row.get(f"{joint}_y", 0.0))
                z = float(row.get(f"{joint}_z", 0.0))
                frame_data.append([x, y, z])  # 单个关节的3D坐标
            skeleton_frames.append(frame_data)
    return np.array(skeleton_frames, dtype=np.float32)  # 形状：(T, 25, 3)

def preprocess_skeleton(skeleton, target_frames=300):
    """
    预处理骨骼数据：
    1. 统一帧数（不足补零，过长均匀采样）
    2. 归一化坐标至[-1, 1]
    """
    T, V, C = skeleton.shape  # T:原始帧数，V:25，C:3
    
    # 1. 统一帧数
    if T < target_frames:
        # 帧数不足：末尾补零（填充无效帧）
        pad = np.zeros((target_frames - T, V, C), dtype=np.float32)
        skeleton = np.concatenate([skeleton, pad], axis=0)
    else:
        # 帧数过多：均匀采样至target_frames帧
        indices = np.linspace(0, T-1, target_frames, dtype=int)
        skeleton = skeleton[indices]
    
    # 2. 归一化（基于坐标最大值缩放）
    max_coord = np.max(np.abs(skeleton))
    if max_coord > 0:
        skeleton = skeleton / max_coord  # 缩放到[-1, 1]
    return skeleton  # 形状：(300, 25, 3)

def adjust_dimension(skeleton):
    """调整维度为(C, T, V, M=1)"""
    # 原始形状：(T=300, V=25, C=3) → 转置为(C, T, V)
    skeleton = skeleton.transpose(2, 0, 1)  # 形状：(3, 300, 25)
    # 新增人数维度M=1（单个人）
    skeleton = np.expand_dims(skeleton, axis=-1)  # 形状：(3, 300, 25, 1)
    return skeleton

def generate_stgcn_dataset(mmad3d_root, out_root, target_frames=300, train_split=0.8):
    """
    批量处理MMAD3D数据，生成ST-GCN格式数据集
    mmad3d_root: MMAD3D根目录（含动作子文件夹）
    out_root: 输出数据集根目录
    """
    # 1. 筛选有效文件并划分训练/验证集
    valid_samples = []  # 存储(文件路径, 动作标签)
    LABEL_MAP = {  # 动作子文件夹→标签
        "Arm_Swing": 0, "Body_pose": 1, "chest_expansion": 2,
        "Drumming": 3, "Frog_Pose": 4, "Marcas_Forward": 5,
        "Marcas_Shaking": 6, "Sing_Clap": 7, "Squat_Pose": 8,
        "Tree_Pose": 9, "Twist_Pose": 10
    }
    
    K = 11  # 区分ASD和非ASD的标签偏移量

    # 遍历动作子文件夹
    for action in os.listdir(mmad3d_root):
        action_dir = os.path.join(mmad3d_root, action)
        if not os.path.isdir(action_dir):
            continue  # 跳过非文件夹
        action_label = LABEL_MAP.get(action, None)
        if action_label is None:
            print(f"跳过未知动作文件夹：{action}")
            continue

        # 处理所有CSV文件
        for csv_file in os.listdir(action_dir):
            if not csv_file.endswith(".csv"):
                continue  # 仅处理CSV文件
            file_name = os.path.splitext(csv_file)[0]  # 获取不含扩展名的文件名
            if len(file_name) == 0:
                continue  # 跳过空文件名

            # 1. 判断人群（ASD/正常）→ 直接映射为二分类标签 【核心修改】
            is_asd = (file_name[-1] == '0')
            final_label = 1 if is_asd else 0  # ASD=1（异常），正常=0（正常）
            
            valid_samples.append((os.path.join(action_dir, csv_file), final_label))
            

            # # 判断是否为ASD儿童（文件名末尾为0）
            # is_asd = (file_name[-1] == '0')
            
            # # 根据规则计算最终标签
            # if is_asd:
            #     final_label = action_label  # ASD儿童：标签=动作类别索引
            # else:
            #     final_label = action_label + K  # 非ASD儿童：标签=动作类别索引 + K
            
            # valid_samples.append((os.path.join(action_dir, csv_file), final_label))
        


        # # 筛选文件名末尾为0的CSV
        # for csv_file in os.listdir(action_dir):
        #     if not csv_file.endswith(".csv"):
        #         continue
        #     file_name = os.path.splitext(csv_file)[0]
        #     if len(file_name) == 0 or file_name[-1] != '0':
        #         continue  # 仅保留末尾为0的文件
            
        #     valid_samples.append((os.path.join(action_dir, csv_file), action_label))
    

    
    # # 划分训练/验证集
    # random.shuffle(valid_samples)
    # split_idx = int(len(valid_samples) * train_split)
    # train_samples = valid_samples[:split_idx]
    # val_samples = valid_samples[split_idx:]
    

    # 2. 数据划分：确保二分类样本平衡（关键新增步骤）
    random.shuffle(valid_samples)
    # 分离0类（正常）和1类（ASD）样本
    normal_samples = [s for s in valid_samples if s[1] == 0]
    asd_samples = [s for s in valid_samples if s[1] == 1]
    # 按较少样本数平衡（避免某类占比过高）
    min_sample_num = min(len(normal_samples), len(asd_samples))
    balanced_samples = normal_samples[:min_sample_num] + asd_samples[:min_sample_num]
    random.shuffle(balanced_samples)  # 重新打乱
    # 按train_split划分训练/验证集
    split_idx = int(len(balanced_samples) * train_split)
    train_samples = balanced_samples[:split_idx]
    val_samples = balanced_samples[split_idx:]

    # 2. 生成训练集数据和标签
    def generate_split(samples, split_name):
        split_dir = os.path.join(out_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        N = len(samples)
        V = 25
        C = 3
        T = target_frames
        M = 1
        
        # 创建特征数据文件（内存映射，节省内存）
        data_path = os.path.join(split_dir, f"{split_name}_data.npy")
        fp = open_memmap(data_path, dtype='float32', mode='w+', shape=(N, C, T, V, M))
        
        # 处理每个样本
        sample_names = []
        sample_labels = []
        for i, (csv_path, label) in enumerate(tqdm(samples, desc=f"处理{split_name}集")):
            # 解析+预处理+调整维度
            raw_skeleton = parse_mmad3d_csv(csv_path)
            if raw_skeleton.size == 0:
                continue  # 跳过空数据
            processed_skeleton = preprocess_skeleton(raw_skeleton, target_frames)
            stgcn_skeleton = adjust_dimension(processed_skeleton)
            # 写入特征文件
            fp[i] = stgcn_skeleton
            # 记录样本名和标签
            sample_name = f"{os.path.basename(os.path.dirname(csv_path))}_{os.path.splitext(os.path.basename(csv_path))[0]}"
            sample_names.append(sample_name)
            sample_labels.append(label)
        
        # 保存标签文件
        label_path = os.path.join(split_dir, f"{split_name}_label.pkl")
        with open(label_path, 'wb') as f:
            pickle.dump((sample_names, sample_labels), f)
        print(f"{split_name}集生成完成：{N}个样本，保存至{split_dir}")
    
    # 生成训练集和验证集
    generate_split(train_samples, "train")
    generate_split(val_samples, "val")


if __name__ == "__main__":
    generate_stgcn_dataset(
        mmad3d_root="dataset/MMAD_advanced/MMASD+/3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/",  # 你的MMAD3D根目录（含动作子文件夹）
        out_root="dataset/mmad_plus_stgcn_dataset/",  # 输出数据集路径
        target_frames=179  # 与ST-GCN配置一致
    )