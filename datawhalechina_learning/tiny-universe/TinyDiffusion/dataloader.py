from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_cifar10_dataloader(root='data', download=False, batch_size=256, img_size=32, num_workers=4):
    train_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 将数据缩放到[0, 1]范围
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 将数据缩放到[-1, 1]范围
    ])
    test_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=train_data_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=test_data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def cvt_to_pil_image(img_tensor):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # 将数据从[-1, 1]缩放到[0, 1]范围
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # 将通道顺序从CHW改为HWC
        transforms.Lambda(lambda t: t * 255.),  # 将数据缩放到[0, 255]范围
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # 将数据转换为uint8类型
        transforms.ToPILImage(),  # 将数据转换为PIL图像格式
    ])

    # 如果图像是批次数据,则取第一个图像
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0, :, :, :]
    return reverse_transforms(img_tensor)

if __name__ == "__main__":
    train_loader, test_loader = load_cifar10_dataloader(download=True)
    image, _ = next(iter(train_loader))
    image = cvt_to_pil_image(image)
    plt.imshow(image)
    plt.show()
    # plt.imsave('test.jpg', image)
