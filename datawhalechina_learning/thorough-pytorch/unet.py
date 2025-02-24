import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch import optim
from tqdm import tqdm


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


class CarvanaDataset(Dataset):
    def __init__(self, base_dir, idx_list, mode="train", transform=None):
        self.base_dir = base_dir
        self.idx_list = idx_list
        self.images = os.listdir(os.path.join(base_dir, "train"))
        self.masks = os.listdir(os.path.join(base_dir, "train_masks"))
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        image_file = self.images[self.idx_list[index]]
        mask_file = image_file[:-4] + "_mask.gif"
        image = Image.open(os.path.join(self.base_dir, "train", image_file))
        if self.mode == "train":
            mask = Image.open(os.path.join(self.base_dir, "train_masks", mask_file))
            if self.transform is not None:
                image = self.transform(image)
                mask = self.transform(mask)
                mask[mask != 0] = 1.0
            return image, mask.float()
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image


def train_unet(model):
    base_dir = "datasets/carvana-image-masking-challenge"
    transform = transforms.Compose([
                    transforms.Resize((256, 256)), 
                    transforms.ToTensor()])
    train_idxs, val_idxs = train_test_split(range(len(os.listdir(os.path.join(base_dir, "train_masks")))), test_size=0.3)
    train_data = CarvanaDataset(base_dir, train_idxs, transform=transform)
    val_data = CarvanaDataset(base_dir, val_idxs, transform=transform)
    train_loader = DataLoader(train_data, batch_size=8, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=2, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    
    def dice_coeff(pred, target):
        eps = 0.0001
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()
        return (2. * intersection + eps) / (m1.sum() + m2.sum() + eps)

    
    def train(epoch):
        unet.train()
        train_loss = 0
        pbar = tqdm(train_loader)
        for data, mask in pbar:
            data, mask = data.cuda(), mask.cuda()
            optimizer.zero_grad()
            output = unet(data)
            loss = criterion(output,mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_description(f'Train: [{epoch}]')
            pbar.set_postfix({'loss': train_loss})
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    def val(epoch):  
        print("current learning rate: ", optimizer.state_dict()["param_groups"][0]["lr"])
        unet.eval()
        val_loss = 0
        dice_score = 0
        with torch.no_grad():
            pbar = tqdm(val_loader)
            for data, mask in pbar:
                data, mask = data.cuda(), mask.cuda()
                output = unet(data)
                loss = criterion(output, mask)
                val_loss += loss.item()
                pbar.set_postfix({'loss': val_loss})
                dice_score += dice_coeff(torch.sigmoid(output).cpu(), mask.cpu())*data.size(0)
        val_loss = val_loss/len(val_loader)
        dice_score = dice_score/len(val_loader.dataset)
        print('Epoch: {} \tValidation Loss: {:.6f}, Dice score: {:.6f}'.format(epoch, val_loss, dice_score))
    
    for epoch in range(10):
        train(epoch)
        val(epoch)
        torch.save(model.state_dict(), f"models/unet_ep{epoch}.pt")

    
    
if __name__ == "__main__":
    unet = UNet(3, 1)
    summary(unet, (1, 3, 572, 572))
    train_unet(unet)
