"""
Convert YOLO format annotations to COCO format
将 YOLO 格式标注转换为 COCO 格式
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Convert YOLO format to COCO format')
    
    parser.add_argument('--images_dir', type=str, required=True,
                        help='图片目录路径')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='YOLO 标注目录路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 COCO JSON 文件路径')
    parser.add_argument('--class_names', type=str, nargs='+', 
                        default=['cotton_boll'],
                        help='类别名称列表')
    
    return parser.parse_args()


def convert_yolo_to_coco(images_dir, labels_dir, class_names):
    """
    将 YOLO 格式转换为 COCO 格式
    
    Args:
        images_dir: 图片目录路径
        labels_dir: YOLO 标注目录路径
        class_names: 类别名称列表
    
    Returns:
        COCO 格式的字典
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # 初始化 COCO 格式数据结构
    coco_format = {
        'info': {
            'description': 'Cotton Boll Detection Dataset',
            'version': '1.0',
            'year': 2024,
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 添加类别信息
    for i, class_name in enumerate(class_names):
        coco_format['categories'].append({
            'id': i,
            'name': class_name,
            'supercategory': 'object'
        })
    
    # 获取所有图片文件
    image_files = list(images_dir.glob('*.jpg')) + \
                  list(images_dir.glob('*.jpeg')) + \
                  list(images_dir.glob('*.png'))
    
    annotation_id = 1
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 遍历所有图片
    for image_id, image_path in enumerate(tqdm(image_files, desc="转换中"), 1):
        # 读取图片尺寸
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"无法读取图片 {image_path}: {e}")
            continue
        
        # 添加图片信息
        coco_format['images'].append({
            'id': image_id,
            'file_name': image_path.name,
            'width': width,
            'height': height,
        })
        
        # 读取对应的 YOLO 标注文件
        label_path = labels_dir / (image_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # 解析 YOLO 标注
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])
            
            # 转换为 COCO 格式 (x, y, width, height)
            # YOLO: (x_center, y_center, width, height) 归一化
            # COCO: (x_min, y_min, width, height) 像素坐标
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width_px = bbox_width * width
            bbox_height_px = bbox_height * height
            
            # 添加标注信息
            coco_format['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': class_id,
                'bbox': [x_min, y_min, bbox_width_px, bbox_height_px],
                'area': bbox_width_px * bbox_height_px,
                'iscrowd': 0,
            })
            
            annotation_id += 1
    
    return coco_format


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🔄 YOLO 转 COCO 格式")
    print("=" * 60)
    print(f"📁 图片目录: {args.images_dir}")
    print(f"📝 标注目录: {args.labels_dir}")
    print(f"💾 输出文件: {args.output}")
    print(f"🏷️  类别: {args.class_names}")
    print("=" * 60)
    
    # 检查目录是否存在
    if not os.path.exists(args.images_dir):
        print(f"❌ 错误: 图片目录不存在: {args.images_dir}")
        return
    
    if not os.path.exists(args.labels_dir):
        print(f"❌ 错误: 标注目录不存在: {args.labels_dir}")
        return
    
    # 转换格式
    coco_data = convert_yolo_to_coco(
        args.images_dir,
        args.labels_dir,
        args.class_names
    )
    
    # 保存 COCO JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 转换完成！")
    print(f"📊 统计信息:")
    print(f"  • 图片数量: {len(coco_data['images'])}")
    print(f"  • 标注数量: {len(coco_data['annotations'])}")
    print(f"  • 类别数量: {len(coco_data['categories'])}")
    print(f"📁 输出文件: {output_path}")


if __name__ == '__main__':
    main()
