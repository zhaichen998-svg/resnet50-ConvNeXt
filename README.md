# 🌱 Cotton Boll Detection with Custom Backbones

基于 YOLOv8 的棉铃目标检测项目，支持 **ConvNeXt** 和 **ResNet50** 作为骨干网络。

## ✨ 特性

- 🎯 支持多种骨干网络（ConvNeXt、ResNet50）
- 📊 完整的训练/验证/测试流程
- 🔄 自动数据格式转换（YOLO ↔ COCO）
- 📈 性能对比可视化
- 🚀 支持 Ultralytics YOLOv8 和 MMDetection 两种框架

## 📁 项目结构

```
resnet50-ConvNeXt/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── backbones/
│   ├── __init__.py
│   ├── convnext_backbone.py
│   └── resnet50_backbone.py
├── models/
│   └── README.md
├── configs/
│   ├── yolo_convnext.py
│   └── yolo_resnet50.py
├── utils/
│   ├── __init__.py
│   ├── convert_to_coco.py
│   └── visualize.py
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/zhaichen998-svg/resnet50-ConvNeXt.git
cd resnet50-ConvNeXt

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

将你的 YOLO 格式数据集放入 `data/` 目录：

```
data/
├── images/
│   ├── train/  # 训练图片
│   └── val/    # 验证图片
└── labels/
    ├── train/  # 训练标签 (.txt)
    └── val/    # 验证标签 (.txt)
```

**标签格式示例**（每行一个目标）：
```
0 0.4255 0.1615 0.149 0.057
0 0.9165 0.0285 0.043 0.011
0 0.1995 0.36 0.143 0.06
```

格式说明：`class x_center y_center width height`（坐标已归一化）

### 3. 训练模型

```bash
# 训练 ResNet50 骨干
python train.py --backbone resnet50 --epochs 100 --batch 16

# 训练 ConvNeXt 骨干
python train.py --backbone convnext --epochs 100 --batch 16
```

### 4. 评估模型

```bash
python evaluate.py --model runs/detect/yolov8_resnet50/weights/best.pt
```

### 5. 推理预测

```bash
python inference.py --model runs/detect/yolov8_resnet50/weights/best.pt --source test.jpg
```

## 📊 性能对比

| 骨干网络 | mAP@0.5 | mAP@0.5:0.95 | 参数量 | 速度 (ms) |
|---------|---------|--------------|--------|-----------|
| ResNet50 | - | - | 25.6M | - |
| ConvNeXt-Tiny | - | - | 28.6M | - |

*运行 `python evaluate.py --compare` 自动生成对比图表*

## 🔧 数据集配置

修改 `data/data.yaml`：

```yaml
path: ./data
train: images/train
val: images/val

nc: 1  # 类别数量
names: ['cotton_boll']  # 类别名称
```

## 📖 使用 MMDetection（可选）

如果需要更多高级功能，可以使用 MMDetection 框架：

```bash
# 1. 安装 MMDetection
pip install openmim
mim install mmengine mmcv mmdet

# 2. 转换数据格式
python utils/convert_to_coco.py

# 3. 训练
python tools/train.py configs/yolo_resnet50.py
python tools/train.py configs/yolo_convnext.py
```

## 📝 引用

如果此项目对你有帮助，请引用：

```bibtex
@software{cotton_boll_detection_2025,
  author = {zhaichen998-svg},
  title = {Cotton Boll Detection with Custom Backbones},
  year = {2025},
  url = {https://github.com/zhaichen998-svg/resnet50-ConvNeXt}
}
```

## 📄 许可证

MIT License

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)

## 📧 联系

如有问题或建议，请提交 Issue 或 Pull Request。

---

⭐ 如果这个项目对你有帮助，请给一个 Star！
