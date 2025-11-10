import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 1. 定义标签映射
LABEL_MAP = {
    "Arm_Swing": 0, "Body_pose": 1, "chest_expansion": 2,
    "Drumming": 3, "Frog_Pose": 4, "Marcas_Forward": 5,
    "Marcas_Shaking": 6, "Sing_Clap": 7, "Squat_Pose": 8,
    "Tree_Pose": 9, "Twist_Pose": 10
}
K = 11  # 非ASD标签偏移量

# 生成22类标签的可读名称（ASD_动作名 / Normal_动作名）
CLASS_NAMES = []
for action in LABEL_MAP:
    CLASS_NAMES.append(f"ASD_{action}")  # ASD儿童的11类：0-10
for action in LABEL_MAP:
    CLASS_NAMES.append(f"Normal_{action}")  # 非ASD儿童的11类：11-21


# 2. 加载结果并提取真实标签和预测标签
def load_labels(result_path):
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    
    gt_labels = []  # 真实标签列表
    pred_labels = []  # 预测标签列表
    for sample in results:
        gt_labels.append(sample['gt_label'].item())
        pred_labels.append(sample['pred_label'].item())
    return np.array(gt_labels), np.array(pred_labels)


# 3. 生成并可视化混淆矩阵
def plot_confusion_matrix(gt_labels, pred_labels, class_names, save_path="confusion_matrix.png"):
    # 计算混淆矩阵（行=真实标签，列=预测标签）
    cm = confusion_matrix(gt_labels, pred_labels)
    
    # 归一化：按真实标签的总样本数归一化（便于观察每类的误判比例）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比
    
    # 设置画布大小（22类标签需要较大画布）
    plt.figure(figsize=(16, 12))
    # 绘制热力图
    sns.heatmap(
        cm_normalized,
        annot=False,  # 22类太多，不显示具体数值（避免混乱）
        fmt='.1f',
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': '百分比（%）'}
    )
    # 添加标签和标题
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('ASD与正常医师动作分类混淆矩阵（归一化）', fontsize=15)
    # 旋转标签（避免重叠）
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"混淆矩阵已保存至：{save_path}")
    plt.show()
    
    return cm, cm_normalized  # 返回原始矩阵和归一化矩阵（用于后续分析）


# 4. 聚焦ASD相关误判的统计函数（计算特定误判比例）
def analyze_asd_misjudgments(cm, class_names):
    # 提取ASD和正常医师的标签索引范围
    asd_indices = list(range(11))  # ASD儿童：0-10
    normal_indices = list(range(11, 22))  # 正常医师：11-21
    
    # 统计1：非ASD动作被误判为ASD同类动作（如Normal_Drumming→ASD_Drumming）
    print("\n=== 非ASD动作被误判为ASD同类动作的比例 ===")
    for i in range(11):
        normal_idx = 11 + i  # 非ASD动作的第i个动作标签（11-21）
        asd_idx = i  # 对应的ASD动作标签（0-10）
        action_name = list(LABEL_MAP.keys())[i]
        total = cm[normal_idx].sum()  # 该正常动作的总样本数
        if total == 0:
            continue
        misjudge_count = cm[normal_idx, asd_idx]  # 被误判为ASD同类动作的样本数
        misjudge_ratio = misjudge_count / total * 100
        print(f"Normal_{action_name} → ASD_{action_name}：{misjudge_ratio:.1f}%（{misjudge_count}/{total}）")
    
    # 统计2：ASD儿童动作被误判为其他ASD动作（内部混淆）
    print("\n=== ASD儿童动作被误判为其他ASD动作的主要类型 ===")
    for i in range(11):
        asd_idx = i
        action_name = list(LABEL_MAP.keys())[i]
        total = cm[asd_idx].sum()
        if total == 0:
            continue
        # 排除正确预测（对角线），找到最大误判类别
        cm_row = cm[asd_idx].copy()
        cm_row[asd_idx] = 0  # 忽略正确预测
        max_mis_idx = cm_row.argmax()
        max_mis_count = cm_row[max_mis_idx]
        max_mis_ratio = max_mis_count / total * 100
        max_mis_action = class_names[max_mis_idx]
        print(f"ASD_{action_name} 最易误判为 {max_mis_action}：{max_mis_ratio:.1f}%（{max_mis_count}/{total}）")


# 主函数
if __name__ == "__main__":
    # 替换为你的result.pkl路径
    result_path = "test/result_22_pp.pkl"
    save_path = 'image/confusion_matrix_22_pp.png'
    
    # 执行流程
    gt_labels, pred_labels = load_labels(result_path)
    cm, cm_normalized = plot_confusion_matrix(gt_labels, pred_labels, CLASS_NAMES,save_path)
    analyze_asd_misjudgments(cm, CLASS_NAMES)