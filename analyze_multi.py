import pickle
import pandas as pd
from collections import defaultdict

# 1. 定义标签映射
LABEL_MAP = {
    "Arm_Swing": 0, "Body_pose": 1, "chest_expansion": 2,
    "Drumming": 3, "Frog_Pose": 4, "Marcas_Forward": 5,
    "Marcas_Shaking": 6, "Sing_Clap": 7, "Squat_Pose": 8,
    "Tree_Pose": 9, "Twist_Pose": 10
}
# 反转映射：动作索引→动作名称（用于表格展示）
INDEX_TO_ACTION = {v: k for k, v in LABEL_MAP.items()}
K = 11  # 非ASD标签偏移量


# 2. 加载result.pkl结果文件
def load_results(result_path):
    with open(result_path, 'rb') as f:
        results = pickle.load(f)  # 列表，每个元素是一个样本的结果字典
    return results


# 3. 统计每个（动作类别+人群）的准确率
def calculate_accuracy(results):
    # 初始化统计字典：key=(动作索引, 人群类型), value={'total': 总样本数, 'correct': 正确样本数}
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for sample in results:
        # 解析真实标签
        gt_label = sample['gt_label'].item()  # 转换为Python标量
        # 确定人群类型（ASD儿童/非ASD儿童）和真实动作索引
        if gt_label < K:
            crowd = "ASD儿童"
            action_idx = gt_label  # ASD动作索引：0-10
        else:
            crowd = "非ASD儿童"
            action_idx = gt_label - K  # 正常医师动作索引：0-10（减去偏移量）
        
        # 解析预测标签
        pred_label = sample['pred_label'].item()
        
        # 判断预测是否正确
        is_correct = (pred_label == gt_label)
        
        # 更新统计：总样本数+1，正确则正确数+1
        stats[(action_idx, crowd)]['total'] += 1
        if is_correct:
            stats[(action_idx, crowd)]['correct'] += 1
    
    return stats


# 4. 生成对比表格
def generate_table(stats):
    # 整理数据为列表（按动作索引排序）
    table_data = []
    for action_idx in sorted(INDEX_TO_ACTION.keys()):
        action_name = INDEX_TO_ACTION[action_idx]
        
        # 获取ASD儿童的统计
        asd_stats = stats.get((action_idx, "ASD儿童"), {'total': 0, 'correct': 0})
        asd_total = asd_stats['total']
        asd_acc = asd_stats['correct'] / asd_total if asd_total > 0 else 0.0
        
        # 获取非ASD儿童的统计
        normal_stats = stats.get((action_idx, "非ASD儿童"), {'total': 0, 'correct': 0})
        normal_total = normal_stats['total']
        normal_acc = normal_stats['correct'] / normal_total if normal_total > 0 else 0.0
        
        # 计算准确率差异（非ASD儿童 - ASD儿童）
        acc_diff = normal_acc - asd_acc
        
        table_data.append({
            "动作类别": action_name,
            "ASD儿童样本数": asd_total,
            "ASD儿童准确率": f"{asd_acc:.2%}",
            "非ASD儿童样本数": normal_total,
            "非ASD儿童准确率": f"{normal_acc:.2%}",
            "准确率差异（正常-ASD）": f"{acc_diff:.2%}"
        })
    
    # 转换为DataFrame并美化
    df = pd.DataFrame(table_data)
    return df


# 5. 主函数：运行整个流程
if __name__ == "__main__":
    # 替换为你的result.pkl路径
    result_path = "test/result_22_pp.pkl"  # 例如："work_dirs/stgcn_multi_class/result.pkl"
    
    # 执行统计并生成表格
    results = load_results(result_path)
    stats = calculate_accuracy(results)
    df = generate_table(stats)
    
    # 打印表格
    print("动作类别+人群准确率对比表：")
    print(df.to_string(index=False))
    
    # 保存为Excel（可选）
    df.to_excel("asd_vs_normal_accuracy_pp.xlsx", index=False)
    print("\n表格已保存为 asd_vs_normal_accuracy.xlsx")