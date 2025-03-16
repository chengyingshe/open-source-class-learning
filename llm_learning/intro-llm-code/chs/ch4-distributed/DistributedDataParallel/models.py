import torch
import torch.nn as nn
import torchvision.models as models

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=21):
        super(DeepLab, self).__init__()
        # 选择骨干网络
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError("Unsupported backbone: {}".format(backbone))

        # 替换最后的全连接层
        if backbone == 'resnet50':
            self.backbone.fc = nn.Identity()  # 移除全连接层
        elif backbone == 'mobilenet_v2':
            self.backbone.classifier = nn.Identity()  # 移除分类器

        # ASPP 模块
        self.aspp = ASPP(in_channels=2048 if backbone == 'resnet50' else 1280, out_channels=256)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        # 骨干网络提取特征
        if isinstance(self.backbone, models.ResNet):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        elif isinstance(self.backbone, models.MobileNetV2):
            x = self.backbone.features(x)

        # ASPP 模块
        x = self.aspp(x)

        # 分类头
        x = self.classifier(x)

        # 上采样到输入分辨率
        x = nn.functional.upsample(x, size=(h, w), mode="bilinear")
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 空洞卷积模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.final_conv(x)
        return x