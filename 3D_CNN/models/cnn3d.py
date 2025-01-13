import torch.nn as nn
import logging
import torch
from config.config import Config

logger = logging.getLogger(__name__)

class CNN3D(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            # 输入形状: (batch, channels=1, time=10, height=256, width=256)
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=0.25),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=0.25),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=0.25),
        )
        
        # 计算特征图的尺寸
        with torch.no_grad():
            # 使用配置中的尺寸创建示例输入
            x = torch.zeros(1, 1, 10, Config.INPUT_HEIGHT, Config.INPUT_WIDTH)
            x = self.features(x)
            self.feature_size = x.numel() // x.size(0)
            logger.info(f"Feature size after convolutions: {self.feature_size}")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        # 打印输入形状以便调试
        logger.debug(f"Input shape: {x.shape}")
        
        x = self.features(x)
        logger.debug(f"After features shape: {x.shape}")
        
        x = self.classifier(x)
        logger.debug(f"Output shape: {x.shape}")
        
        return x 