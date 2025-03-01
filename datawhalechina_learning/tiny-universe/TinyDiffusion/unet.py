import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math

class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=None):
        super().__init__()
        if not mid_dim:
            mid_dim = out_dim
        self.double_conv = nn.Sequential(
            # Conv -> BN -> Act
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_dim, out_dim)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_dim, out_dim, bilinear=False):
        super().__init__()
        # if bilinear, use the normal conv to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # unlearnable (not change the number of channel)
            self.conv = DoubleConv(in_dim, out_dim, in_dim // 2)
        else:
            self.up = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)  # change the number of channel to half
            self.conv = DoubleConv(in_dim, out_dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # [B, C, H, W]
        diff_x = x2.shape[3] - x1.shape[3]
        diff_y = x2.shape[2] - x1.shape[2]
        # 这部分与原论文存在差异，原论文中将左边的输入进行裁剪，最后导致output segmentation map比输入图片的尺寸小，为了保持最后输出与输入图片的尺寸一致，这里采用padding的方式
        x1 = F.pad(x1, [diff_x // 2,  # left
                        diff_x - diff_x // 2,  # right
                        diff_y // 2,  # top
                        diff_y - diff_y // 2  # bottom
                        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 使用1x1卷积，只改变输出的通道数
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=False):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x
    
class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        
        
        return x

if __name__ == "__main__":
    unet = UNet(3, 1)
    summary(unet, (1, 3, 572, 572))