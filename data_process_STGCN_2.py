import numpy as np
import pickle
import os

def convert_3d_skeleton_to_mmaction2(
    dataset_root: str,
    output_pkl_path: str
):
    """
    将自定义3D骨骼数据（npy+pickle标签）转换为mmaction2标准格式
    :param dataset_root: 数据集根目录（mmad3d_stgcn_dataset）
    :param output_pkl_path: 输出标准pickle文件路径
    """
    # -------------------------- 1. 读取训练集和验证集数据 --------------------------
    # 训练集数据
    train_data_path = os.path.join(dataset_root, "train", "train_data.npy")
    train_label_path = os.path.join(dataset_root, "train", "train_label.pkl")
    # 验证集数据
    val_data_path = os.path.join(dataset_root, "val", "val_data.npy")
    val_label_path = os.path.join(dataset_root, "val", "val_label.pkl")

    # 读取npy特征数据（N, C, T, V, M）
    train_data = np.load(train_data_path, allow_pickle=True)
    val_data = np.load(val_data_path, allow_pickle=True)

    # 读取label.pkl（格式：(样本名列表, 标签列表)）
    with open(train_label_path, "rb") as f:
        train_names, train_labels = pickle.load(f)
    with open(val_label_path, "rb") as f:
        val_names, val_labels = pickle.load(f)

    # 校验数据一致性（样本数匹配）
    assert len(train_names) == len(train_labels) == train_data.shape[0], "训练集样本名、标签、数据数量不匹配"
    assert len(val_names) == len(val_labels) == val_data.shape[0], "验证集样本名、标签、数据数量不匹配"
    print(f"训练集：{len(train_names)}个样本，验证集：{len(val_names)}个样本")

    # -------------------------- 2. 构造 annotations 列表 --------------------------
    annotations = []

    # 处理训练集样本
    for idx in range(len(train_names)):
        sample_name = train_names[idx]
        label = int(train_labels[idx])
        # 取出单个样本数据：(C, T, V, M) → (3, 300, 25, 1)
        single_data = train_data[idx]
        # print(single_data.shape)
        # 转换维度：(C, T, V, M) → (M, T, V, C)（mmaction2要求）
        keypoint = single_data.transpose(3, 1, 2, 0)  # 转置顺序：M(3)、T(2)、V(1)、C(0)
        # 构造单个样本的annotation字典（仅保留3D必需字段）
        anno = {
            "frame_dir": sample_name,          # 样本名作为frame_dir
            "total_frames": 179,               # 固定300帧（你的数据帧长）
            "label": label,                    # 动作标签
            "keypoint": keypoint.astype(np.float32)  # 转换为float32，适配模型输入
        }
        annotations.append(anno)

    # 处理验证集样本
    for idx in range(len(val_names)):
        sample_name = val_names[idx]
        label = int(val_labels[idx])
        single_data = val_data[idx]
        keypoint = single_data.transpose(3, 1, 2, 0)  # 同训练集维度转换
        anno = {
            "frame_dir": sample_name,
            "total_frames": 179,
            "label": label,
            "keypoint": keypoint.astype(np.float32)
        }
        annotations.append(anno)

    # -------------------------- 3. 构造 split 字典 --------------------------
    split = {
        "train": train_names,  # 训练集样本名列表
        "val": val_names       # 验证集样本名列表
    }

    # -------------------------- 4. 保存为标准pickle文件 --------------------------
    output_data = {
        "split": split,
        "annotations": annotations
    }

    with open(output_pkl_path, "wb") as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"转换完成！标准pickle文件已保存至：{output_pkl_path}")

    # -------------------------- 5. 验证转换结果（可选） --------------------------
    with open(output_pkl_path, "rb") as f:
        test_data = pickle.load(f)
    print(f"\n验证结果：")
    print(f"- split划分：{list(test_data['split'].keys())}")
    print(f"- 总样本数：{len(test_data['annotations'])}")
    print(f"- 第一个样本frame_dir：{test_data['annotations'][0]['frame_dir']}")
    print(f"- 第一个样本label：{test_data['annotations'][0]['label']}")
    print(f"- 第一个样本keypoint形状（M×T×V×C）：{test_data['annotations'][0]['keypoint'].shape}")


if __name__ == "__main__":
    # 你的数据集根目录（mmad3d_stgcn_dataset的路径）
    DATASET_ROOT = "dataset/mmad_plus_stgcn_dataset"  # 替换为实际路径（如Windows："D:/mmad3d_stgcn_dataset"）
    # 输出的标准pickle文件路径
    OUTPUT_PKL = "dataset/mmad_plus_stgcn_dataset/mmad3d_mmaction2.pkl"  # 替换为输出路径

    # 执行转换
    convert_3d_skeleton_to_mmaction2(
        dataset_root=DATASET_ROOT,
        output_pkl_path=OUTPUT_PKL
    )