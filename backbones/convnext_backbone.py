"""
ConvNeXt Backbone for YOLOv8
适用于棉铃检测任务的 ConvNeXt 主干网络
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt-Tiny 骨干网络，提取多尺度特征用于目标检测
    
    输出特征层：
    - P3: 1/8 分辨率, 192 通道
    - P4: 1/16 分辨率, 384 通道
    - P5: 1/32 分辨率, 768 通道
    """
    
    def __init__(self, pretrained=True, out_indices=(1, 2, 3)):
        """
        Args:
            pretrained (bool): 是否使用 ImageNet 预训练权重
            out_indices (tuple): 输出特征层索引
        """
        super().__init__()
        
        # 加载预训练模型
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)
        
        # ConvNeXt 结构：features 包含多个 stage
        self.features = convnext.features
        
        # ConvNeXt-Tiny 的 stage 划分
        # Stage 0: Downsample (stem)
        # Stage 1-2: Block group 1 (stride 4)
        # Stage 3-4: Block group 2 (stride 8)
        # Stage 5-6: Block group 3 (stride 16)
        # Stage 7: Block group 4 (stride 32)
        
        self.out_indices = out_indices
        self.stage_indices = [0, 2, 4, 6, 7]  # 各阶段的结束索引
        
        # 通道数信息
        self.out_channels = [96, 192, 384, 768]
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 tensor, shape (B, 3, H, W)
            
        Returns:
            list: 多尺度特征列表
        """
        outputs = []
        stage_idx = 0
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # 检查是否到达输出阶段
            if stage_idx < len(self.stage_indices) - 1:
                if i == self.stage_indices[stage_idx + 1]:
                    stage_idx += 1
                    if stage_idx in self.out_indices:
                        outputs.append(x)
        
        return outputs
    
    def freeze_stages(self, num_stages):
        """冻结前 N 个阶段的参数"""
        frozen_layers = 0
        stage_idx = 0
        
        for i, layer in enumerate(self.features):
            if stage_idx < num_stages:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
                    
            if stage_idx < len(self.stage_indices) - 1:
                if i == self.stage_indices[stage_idx + 1]:
                    stage_idx += 1


class ConvNeXtLargeBackbone(ConvNeXtBackbone):
    """ConvNeXt-Large 版本（更大容量）"""
    
    def __init__(self, pretrained=True, out_indices=(1, 2, 3)):
        super().__init__(pretrained=False, out_indices=out_indices)
        
        from torchvision.models import convnext_large, ConvNeXt_Large_Weights
        
        weights = ConvNeXt_Large_Weights.DEFAULT if pretrained else None
        convnext = convnext_large(weights=weights)
        self.features = convnext.features
        
        # Large 版本的通道数
        self.out_channels = [192, 384, 768, 1536]


def test_convnext_backbone():
    """测试 ConvNeXt 骨干网络"""
    model = ConvNeXtBackbone(pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    
    outputs = model(x)
    print("ConvNeXt Backbone 输出:")
    for i, feat in enumerate(outputs):
        print(f"  P{i+3}: {feat.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.2f}M")


if __name__ == '__main__':
    test_convnext_backbone()
