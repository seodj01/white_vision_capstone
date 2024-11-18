import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 경량화된 ASPP 모듈
class LightweightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_pool = nn.BatchNorm2d(out_channels)

        self.conv_out = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))

        x4 = self.pool(x)
        x4 = F.relu(self.bn_pool(self.conv_pool(x4)))
        x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_out(x)
        return x

# 경량화된 DeepLabV3 모델
class LightweightDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(LightweightDeepLabV3, self).__init__()

        # MobileNetV2 백본 사용
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            backbone.features,
            nn.Conv2d(1280, 320, kernel_size=1)  # 마지막 레이어를 압축
        )

        # 경량화된 ASPP 모듈
        self.aspp = LightweightASPP(in_channels=320, out_channels=128)

        # 피처 조정을 위한 추가 Conv 레이어
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # 최종 출력 클래스 수에 맞추기
        self.conv2 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()[2:]

        x = self.backbone(x)

        x = self.aspp(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=False)
        return x

# GPU 또는 CPU에 모델 로드
model = LightweightDeepLabV3(num_classes=8)  # 클래스 수에 맞게 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
