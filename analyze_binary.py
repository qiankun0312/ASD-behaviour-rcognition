import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_results(pkl_path):
    """加载pkl文件中的结果"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    print(f"成功加载 {len(results)} 个样本")
    return results

def extract_labels_and_scores(results):
    """从结果中提取真实标签、预测标签和预测分数"""
    gt_labels = []
    pred_labels = []
    pred_scores = []  # 正类的概率分数
    
    for result in results:
        # 提取真实标签
        if isinstance(result['gt_label'], torch.Tensor):
            gt_label = result['gt_label'].item()
        else:
            gt_label = result['gt_label']
        gt_labels.append(gt_label)
        
        # 提取预测标签
        if isinstance(result['pred_label'], torch.Tensor):
            pred_label = result['pred_label'].item()
        else:
            pred_label = result['pred_label']
        pred_labels.append(pred_label)
        
        # 提取预测分数（取第二个类别的分数作为正类概率）
        if isinstance(result['pred_score'], torch.Tensor):
            pred_score = result['pred_score'].numpy()
        else:
            pred_score = np.array(result['pred_score'])
        
        # 假设类别1是正类（ASD行为），取第二个分数
        if len(pred_score) == 2:
            pred_scores.append(pred_score[1])  # 正类概率
        else:
            # 如果只有一个分数，假设是二分类的logits
            pred_scores.append(pred_score[0] if len(pred_score) == 1 else pred_score[1])
    
    return np.array(gt_labels), np.array(pred_labels), np.array(pred_scores)

def calculate_metrics(gt_labels, pred_labels, pred_scores, positive_class=1, class_names=None):
    """计算所有评估指标"""
    
    if class_names is None:
        class_names = ['Class_0', 'Class_1']
    
    # 基础分类指标
    accuracy = accuracy_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels, average='binary', pos_label=positive_class)
    recall = recall_score(gt_labels, pred_labels, average='binary', pos_label=positive_class)
    f1 = f1_score(gt_labels, pred_labels, average='binary', pos_label=positive_class)
    
    # 多类别平均指标（对于二分类，macro和weighted通常相同）
    precision_macro = precision_score(gt_labels, pred_labels, average='macro')
    recall_macro = recall_score(gt_labels, pred_labels, average='macro')
    f1_macro = f1_score(gt_labels, pred_labels, average='macro')
    
    precision_weighted = precision_score(gt_labels, pred_labels, average='weighted')
    recall_weighted = recall_score(gt_labels, pred_labels, average='weighted')
    f1_weighted = f1_score(gt_labels, pred_labels, average='weighted')
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(gt_labels, pred_scores)
    except Exception as e:
        print(f"计算ROC-AUC时出错: {e}")
        roc_auc = None
    
    # 混淆矩阵
    cm = confusion_matrix(gt_labels, pred_labels)
    
    # 各类别详细指标
    class_report = classification_report(gt_labels, pred_labels, target_names=class_names, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'gt_labels': gt_labels,
        'pred_labels': pred_labels
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")

def plot_roc_curve(gt_labels, pred_scores, save_path):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    roc_auc = roc_auc_score(gt_labels, pred_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")

def print_detailed_metrics(metrics, class_names=['Typical', 'ASD']):
    """打印详细的评估指标"""
    
    print("\n" + "="*70)
    print("STGCN 模型性能评估结果")
    print("="*70)
    
    print(f"\n📊 基础指标:")
    print("-" * 40)
    print(f"准确率 (Accuracy):          {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision):         {metrics['precision']:.4f}")
    print(f"召回率 (Recall):            {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):          {metrics['f1_score']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:                    {metrics['roc_auc']:.4f}")
    
    print(f"\n📈 宏平均指标:")
    print("-" * 40)
    print(f"宏平均精确率:               {metrics['precision_macro']:.4f}")
    print(f"宏平均召回率:               {metrics['recall_macro']:.4f}")
    print(f"宏平均F1分数:               {metrics['f1_macro']:.4f}")
    
    print(f"\n⚖️  加权平均指标:")
    print("-" * 40)
    print(f"加权平均精确率:             {metrics['precision_weighted']:.4f}")
    print(f"加权平均召回率:             {metrics['recall_weighted']:.4f}")
    print(f"加权平均F1分数:             {metrics['f1_weighted']:.4f}")
    
    print(f"\n🎯 各类别详细指标:")
    print("-" * 40)
    for class_name in class_names:
        if class_name in metrics['classification_report']:
            report = metrics['classification_report'][class_name]
            print(f"{class_name}:")
            print(f"  精确率: {report['precision']:.4f}")
            print(f"  召回率: {report['recall']:.4f}")
            print(f"  F1分数: {report['f1-score']:.4f}")
            print(f"  支持数: {report['support']}")
    
    print(f"\n📋 总体统计:")
    print("-" * 40)
    print(f"总样本数: {len(metrics['gt_labels'])}")
    print(f"准确率:   {metrics['accuracy']:.4f}")
    
    print(f"\n🔢 混淆矩阵:")
    print("-" * 40)
    print(metrics['confusion_matrix'])

def save_metrics_to_file(metrics, save_path, class_names=['Typical_Behavior', 'ASD_Behavior']):
    """将指标保存到文件"""
    with open(save_path, 'w',encoding='utf-8') as f:
        f.write("STGCN模型评估结果\n")
        f.write("="*50 + "\n\n")
        
        f.write("基础指标:\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"F1分数 (F1-Score): {metrics['f1_score']:.4f}\n")
        if metrics['roc_auc'] is not None:
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        f.write("\n混淆矩阵:\n")
        f.write(str(metrics['confusion_matrix']))
        f.write("\n\n详细分类报告:\n")
        
        # 使用 classification_report 生成字符串报告
        report_str = classification_report(
            metrics['gt_labels'], 
            metrics['pred_labels'], 
            target_names=class_names
        )
        f.write(report_str)
    
    print(f"详细结果已保存至: {save_path}")

def main(pkl_path, output_dir='evaluation_results', positive_class=1):
    """主函数"""
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 类别名称（根据您的数据集调整）
    class_names = ['Typical_Behavior', 'ASD_Behavior']
    
    # 加载结果
    print("正在加载结果文件...")
    results = load_results(pkl_path)
    
    # 提取标签和分数
    print("正在提取标签和预测分数...")
    gt_labels, pred_labels, pred_scores = extract_labels_and_scores(results)
    
    # 计算指标
    print("正在计算评估指标...")
    metrics = calculate_metrics(gt_labels, pred_labels, pred_scores, positive_class, class_names)
    
    # 打印结果
    print_detailed_metrics(metrics, class_names)
    
    # 绘制图表
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, 
                         f'{output_dir}/confusion_matrix.png')
    
    if metrics['roc_auc'] is not None:
        plot_roc_curve(gt_labels, pred_scores, f'{output_dir}/roc_curve.png')
    
    # 保存结果到文件
    save_metrics_to_file(metrics, f'{output_dir}/evaluation_results.txt', class_names)
    
    # 保存结构化数据
    results_data = {
        'gt_labels': gt_labels,
        'pred_labels': pred_labels,
        'pred_scores': pred_scores,
        'metrics': metrics
    }
    with open(f'{output_dir}/evaluation_data.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\n✅ 所有评估完成！结果保存在: {output_dir}/")
    
    return metrics

if __name__ == "__main__":
    # 使用示例
    pkl_file_path = "test/result_2_pp.pkl"  # 替换为您的pkl文件路径
    output_directory = "evaluation_results"
    
    # 运行评估
    metrics = main(pkl_file_path, output_directory, positive_class=1)