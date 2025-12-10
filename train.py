"""
YOLOv8 Training Script with Custom Backbones
支持 ResNet50 和 ConvNeXt 的 YOLOv8 训练脚本
"""

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 with custom backbones')
    
    # 基础参数
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.yaml',
                        help='模型配置文件（yolov8n/s/m/l/x）')
    parser.add_argument('--backbone', type=str, default='default',
                        choices=['default', 'resnet50', 'convnext'],
                        help='选择骨干网络：default（YOLOv8原生）/ resnet50 / convnext')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU 设备 (e.g., 0 or 0,1,2,3 or cpu)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率（相对于 lr0）')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD 动量/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='权重衰减')
    
    # 数据增强
    parser.add_argument('--hsv_h', type=float, default=0.015,
                        help='HSV 色调增强')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                        help='HSV 饱和度增强')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                        help='HSV 亮度增强')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='旋转角度（度）')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='平移比例')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='缩放比例')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic 增强概率')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='MixUp 增强概率')
    
    # 其他设置
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用预训练权重')
    parser.add_argument('--resume', type=str, default='',
                        help='从检查点恢复训练')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--exist_ok', action='store_true',
                        help='是否覆盖已存在的实验')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--save_period', type=int, default=-1,
                        help='每 N 个 epoch 保存一次模型（-1 表示只保存最后）')
    
    return parser.parse_args()


def modify_model_with_custom_backbone(model, backbone_type):
    """
    使用自定义骨干网络替换 YOLOv8 的默认骨干
    
    注意：这需要修改 YOLOv8 的模型结构，可能需要自定义 YAML 配置
    目前仅作为示例，实际使用需要根据 Ultralytics 的 API 调整
    """
    if backbone_type == 'resnet50':
        from backbones import ResNet50Backbone
        print("🔧 使用 ResNet50 作为骨干网络")
        # TODO: 这里需要实现将 ResNet50 集成到 YOLOv8 的逻辑
        # 可能需要修改 ultralytics 源码或使用自定义模型配置
        
    elif backbone_type == 'convnext':
        from backbones import ConvNeXtBackbone
        print("🔧 使用 ConvNeXt 作为骨干网络")
        # TODO: 这里需要实现将 ConvNeXt 集成到 YOLOv8 的逻辑
        
    else:
        print("✅ 使用 YOLOv8 默认骨干网络")
    
    return model


def main():
    """主训练函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 YOLOv8 训练脚本 - 棉铃检测")
    print("=" * 60)
    print(f"📊 数据集: {args.data}")
    print(f"🏗️  模型: {args.model}")
    print(f"🔧 骨干网络: {args.backbone}")
    print(f"📦 批次大小: {args.batch}")
    print(f"🔁 训练轮数: {args.epochs}")
    print(f"💻 设备: {args.device}")
    print("=" * 60)
    
    # 加载模型
    if args.resume:
        print(f"♻️  从检查点恢复: {args.resume}")
        model = YOLO(args.resume)
    else:
        # 加载预训练权重或从头训练
        if args.pretrained:
            print("✅ 使用预训练权重")
            model = YOLO(f'{args.model.replace(".yaml", ".pt")}')
        else:
            print("🆕 从头开始训练")
            model = YOLO(args.model)
    
    # 使用自定义骨干网络（如果指定）
    if args.backbone != 'default':
        model = modify_model_with_custom_backbone(model, args.backbone)
    
    # 训练配置
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'workers': args.workers,
        'save_period': args.save_period,
        'verbose': True,
        'plots': True,
    }
    
    # 开始训练
    print("\n🏋️  开始训练...\n")
    results = model.train(**train_kwargs)
    
    print("\n✅ 训练完成！")
    print(f"📁 结果保存在: {model.trainer.save_dir}")
    
    return results


if __name__ == '__main__':
    main()
