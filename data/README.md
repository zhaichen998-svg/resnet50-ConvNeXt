# 数据集说明

本目录用于存放棉铃检测数据集。

## 📁 目录结构

```
data/
├── data.yaml              # YOLOv8 数据集配置文件
├── images/
│   ├── train/            # 训练集图片
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/              # 验证集图片
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/            # 训练集标注（YOLO 格式）
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/              # 验证集标注
        ├── image1.txt
        └── ...
```

## 📝 YOLO 标注格式

每个图片对应一个同名的 `.txt` 标注文件，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 类别 ID（棉铃为 0）
- `x_center, y_center`: 目标中心点归一化坐标（0-1）
- `width, height`: 目标宽高归一化尺寸（0-1）

示例：
```
0 0.5 0.5 0.2 0.3
0 0.3 0.4 0.15 0.25
```

## 🔄 数据格式转换

如果你的标注是其他格式（如 COCO、VOC），可以使用工具转换：

```bash
# COCO 转 YOLO
python utils/convert_to_coco.py --input annotations.json --output data/

# 使用 Roboflow 或 Labelme 等工具导出 YOLO 格式
```

## 📊 数据集统计

训练前建议检查数据集质量：

```bash
# 使用 YOLOv8 内置工具
yolo val data=data/data.yaml model=yolov8n.pt

# 查看数据分布
python -c "from ultralytics import YOLO; YOLO().val(data='data/data.yaml', plots=True)"
```

## 💡 数据增强建议

训练时推荐使用以下增强策略：
- 随机水平翻转
- 随机缩放（0.5-1.5x）
- HSV 颜色抖动
- Mosaic 拼接（YOLOv8 默认开启）
- MixUp（可选）

## 🎯 注意事项

1. **图片格式**：支持 `.jpg`, `.jpeg`, `.png`
2. **标注质量**：确保边界框准确，避免错标/漏标
3. **类别平衡**：如果类别不平衡，考虑数据采样策略
4. **数据划分**：建议训练集:验证集 = 8:2 或 7:3

## 📚 参考资源

- [YOLOv8 数据格式文档](https://docs.ultralytics.com/)
- [Roboflow 数据标注平台](https://roboflow.com/)
- [Labelme 标注工具](https://github.com/wkentaro/labelme)