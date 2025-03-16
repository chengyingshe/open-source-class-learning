import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class Cityscaples(Dataset):
    def __init__(self, root='./data/cityscapes', split='train', image_transform =None, label_transform=None):
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.label_transform = label_transform

        # 加载图像和标签路径
        self.images = []
        self.labels = []
        image_dir = os.path.join(root, 'leftImg8bit', split)
        label_dir = os.path.join(root, 'gtFine', split)

        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            for file_name in os.listdir(city_image_dir):
                if file_name.endswith('.png'):
                    self.images.append(os.path.join(city_image_dir, file_name))
                    label_name = file_name.replace('leftImg8bit', 'gtFine_labelIds')
                    self.labels.append(os.path.join(city_label_dir, label_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        # 将标签转换为 Tensor
        label = torch.from_numpy(np.array(label)).long().squeeze(0)
        return image, label