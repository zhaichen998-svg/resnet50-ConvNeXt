"""
YOLOv8 Inference Script
单张图片或批量推理脚本，支持可视化和结果保存
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 Inference Script')
    
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径 (e.g., runs/train/exp/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源：图片路径、文件夹路径、视频路径或摄像头 (0)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='目标置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU 阈值')
    parser.add_argument('--device', type=str, default='',
                        help='GPU 设备 (e.g., 0 or cpu)')
    parser.add_argument('--save', action='store_true',
                        help='是否保存推理结果')
    parser.add_argument('--save_txt', action='store_true',
                        help='是否保存文本标注结果')
    parser.add_argument('--save_conf', action='store_true',
                        help='在保存的标注中包含置信度')
    parser.add_argument('--save_crop', action='store_true',
                        help='保存裁剪的检测目标')
    parser.add_argument('--nosave', action='store_true',
                        help='不保存图片/视频')
    parser.add_argument('--view_img', action='store_true',
                        help='显示推理结果')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--exist_ok', action='store_true',
                        help='是否覆盖已存在的实验')
    parser.add_argument('--line_thickness', type=int, default=3,
                        help='边界框线条粗细')
    parser.add_argument('--hide_labels', action='store_true',
                        help='隐藏标签')
    parser.add_argument('--hide_conf', action='store_true',
                        help='隐藏置信度')
    parser.add_argument('--half', action='store_true',
                        help='使用 FP16 半精度推理')
    parser.add_argument('--vid_stride', type=int, default=1,
                        help='视频帧率步长')
    
    return parser.parse_args()


def draw_boxes(image, boxes, class_names, line_thickness=3, hide_labels=False, hide_conf=False):
    """
    在图像上绘制检测框
    
    Args:
        image: 输入图像
        boxes: 检测框列表 [(x1, y1, x2, y2, conf, cls), ...]
        class_names: 类别名称字典
        line_thickness: 线条粗细
        hide_labels: 是否隐藏标签
        hide_conf: 是否隐藏置信度
    
    Returns:
        绘制了检测框的图像
    """
    img = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # 随机颜色（基于类别）
        np.random.seed(cls)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        # 绘制标签
        if not hide_labels:
            label = f"{class_names.get(cls, f'class{cls}')}"
            if not hide_conf:
                label += f" {conf:.2f}"
            
            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制背景
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return img


def run_inference(args):
    """运行推理"""
    
    print("=" * 60)
    print("🚀 YOLOv8 推理")
    print("=" * 60)
    print(f"🏗️  模型: {args.model}")
    print(f"📁 输入源: {args.source}")
    print(f"📏 图像尺寸: {args.imgsz}")
    print(f"🎯 置信度阈值: {args.conf}")
    print(f"💻 设备: {args.device if args.device else 'auto'}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(args.model)
    
    # 推理配置
    predict_kwargs = {
        'source': args.source,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'device': args.device,
        'save': not args.nosave,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'show': args.view_img,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'line_width': args.line_thickness,
        'hide_labels': args.hide_labels,
        'hide_conf': args.hide_conf,
        'half': args.half,
        'vid_stride': args.vid_stride,
        'verbose': True,
    }
    
    # 开始推理
    print("\n🔍 开始推理...\n")
    results = model.predict(**predict_kwargs)
    
    # 统计检测结果
    total_detections = 0
    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)
    
    print("\n✅ 推理完成！")
    print(f"📊 总检测目标数: {total_detections}")
    
    if not args.nosave:
        save_dir = Path(args.project) / args.name
        print(f"📁 结果保存在: {save_dir}")
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入源是否存在
    source_path = Path(args.source)
    if not source_path.exists() and args.source != '0':
        print(f"❌ 错误: 输入源不存在: {args.source}")
        return
    
    # 运行推理
    results = run_inference(args)
    
    # 打印详细结果
    print("\n" + "=" * 60)
    print("📊 检测详情")
    print("=" * 60)
    
    for i, result in enumerate(results):
        print(f"\n图片 {i+1}:")
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names.get(cls, f'class{cls}')
                print(f"  - {class_name}: {conf:.3f}")
        else:
            print("  未检测到目标")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
