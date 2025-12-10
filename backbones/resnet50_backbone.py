"""
ResNet50 Backbone for YOLOv8
适用于棉铃检测任务的 ResNet50 主干网络
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """
    ResNet50 骨干网络，提取多尺度特征用于目标检测
    
    输出特征层：
    - P3: 1/8 分辨率, 512 通道
    - P4: 1/16 分辨率, 1024 通道
    - P5: 1/32 分辨率, 2048 通道
    """
    
    def __init__(self, pretrained=True, out_indices=(1, 2, 3)):
        """
        Args:
            pretrained (bool): 是否使用 ImageNet 预训练权重
            out_indices (tuple): 输出特征层索引
        """
        super().__init__()
        
        # 加载预训练模型
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        
        # 提取各个阶段
        self.stem = nn.Sequential(
            resnet.conv1,      # 7x7 conv, stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool     # 3x3 pool, stride 2
        )
        
        # ResNet stages
        self.stage1 = resnet.layer1  # 256 通道, stride 1 (总stride 4)
        self.stage2 = resnet.layer2  # 512 通道, stride 2 (总stride 8)
        self.stage3 = resnet.layer3  # 1024 通道, stride 2 (总stride 16)
        self.stage4 = resnet.layer4  # 2048 通道, stride 2 (总stride 32)
        
        self.out_indices = out_indices
        
        # 通道数信息（用于后续 Neck 连接）
        self.out_channels = [256, 512, 1024, 2048]
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 tensor, shape (B, 3, H, W)
            
        Returns:
            list: 多尺度特征列表
        """
        outputs = []
        
        x = self.stem(x)
        
        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)
        
        return outputs
    
    def freeze_stages(self, num_stages):
        """冻结前 N 个阶段的参数"""
        if num_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
                
        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        for i in range(num_stages):
            stages[i].eval()
            for param in stages[i].parameters():
                param.requires_grad = False


def test_resnet50_backbone():
    """测试 ResNet50 骨干网络"""
    model = ResNet50Backbone(pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    
    outputs = model(x)
    print("ResNet50 Backbone 输出:")
    for i, feat in enumerate(outputs):
        print(f"  P{i+3}: {feat.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.2f}M")


if __name__ == '__main__':
    test_resnet50_backbone()
