"""
Custom Backbones for YOLOv8
"""

from .resnet50_backbone import ResNet50Backbone
from .convnext_backbone import ConvNeXtBackbone, ConvNeXtLargeBackbone

__all__ = ['ResNet50Backbone', 'ConvNeXtBackbone', 'ConvNeXtLargeBackbone']
