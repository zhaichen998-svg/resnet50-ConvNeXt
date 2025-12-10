"""
Visualization utilities for training results and model comparison
可视化工具 - 绘制训练曲线、混淆矩阵和模型对比图
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def plot_results(results_dir, save_path=None):
    """
    绘制训练结果曲线
    
    Args:
        results_dir: 训练结果目录（包含 results.csv）
        save_path: 保存路径（可选）
    """
    results_file = Path(results_dir) / 'results.csv'
    
    if not results_file.exists():
        print(f"❌ 找不到结果文件: {results_file}")
        return
    
    # 读取结果数据
    import pandas as pd
    df = pd.read_csv(results_file)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold')
    
    # 1. Loss 曲线
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2)
    if 'train/dfl_loss' in df.columns:
        ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. mAP 曲线
    ax = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Validation mAP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision & Recall
    ax = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax.plot(df['epoch'], df['lr/pg0'], label='LR', linewidth=2, color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵数组
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 8))
    
    # 归一化
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Probability'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存: {save_path}")
    else:
        plt.show()


def compare_models(model_results, save_path=None):
    """
    对比多个模型的性能
    
    Args:
        model_results: 字典，格式 {模型名称: {指标名: 值}}
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    models = list(model_results.keys())
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = [model_results[model].get(metric, 0) for model in models]
        bars = ax.bar(models, values, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比图已保存: {save_path}")
    else:
        plt.show()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Visualization utilities')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['results', 'compare'],
                        help='可视化模式')
    parser.add_argument('--results_dir', type=str,
                        help='训练结果目录')
    parser.add_argument('--compare_json', type=str,
                        help='模型对比 JSON 文件')
    parser.add_argument('--save', type=str,
                        help='保存路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'results':
        if not args.results_dir:
            print("❌ 错误: results 模式需要指定 --results_dir")
            return
        
        print(f"📊 绘制训练结果: {args.results_dir}")
        plot_results(args.results_dir, args.save)
    
    elif args.mode == 'compare':
        if not args.compare_json:
            print("❌ 错误: compare 模式需要指定 --compare_json")
            return
        
        # 读取对比数据
        with open(args.compare_json, 'r') as f:
            model_results = json.load(f)
        
        print(f"📊 绘制模型对比")
        compare_models(model_results, args.save)


if __name__ == '__main__':
    main()
