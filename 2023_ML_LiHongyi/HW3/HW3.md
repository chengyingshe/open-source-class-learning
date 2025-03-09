# HW3

## Grading Policy

|Baseline|Accuracy|Hints|
|----|----|----|
| Simple | 0.637 | Run Sample Code |
| Medium | 0.700 | Do some Data Augmentation & Train longer |
| Strong | 0.814 | Use predefined CNN from torchvision or TensorFlow |
| Boss | 0.874 | Cross Validation + Ensemble or any other methods you know |

#### Simple

- Add label smoothing

```python
# Add label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Epoch=10, Accuracy=0.633

#### Medium

- Add data augmentation

```python
new_train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop((128, 128), scale=(0.5, 1)),
    transforms.ToTensor(),
])
```

Epoch=50, Accuracy=0.719

#### Strong

- Use predefined model: `ResNet101`
- Add learning rate scheduler

```python
import timm
from torch.optim.lr_scheduler import StepLR

resnet = timm.create_model('resnet101', pretrained=False, num_classes=11).to(device)

def get_dataloader(root, train_transform, test_transform, batch_size, num_workers=0):
    """重新划分train, val数据集，增加train的比例"""
    train_set = FoodDataset(os.path.join(root, "train"), tfm=train_transform)
    valid_set = FoodDataset(os.path.join(root, "valid"), tfm=test_transform)
    # Repartition the dataset
    train_set.files.extend(valid_set.files[:2000])
    valid_set.files = valid_set.files[2000:]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, valid_loader

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # with label smoothing
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001, weight_decay=1e-5)
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # add learning rate scheduler
train_loader, valid_loader = get_dataloader('oscl_datasets', new_train_tfm, test_tfm, batch_size)
train(resnet, 50, train_loader, valid_loader, device, criterion, optimizer, lr_scheduler=lr_scheduler, exp_name='strong_resnet', patience=-1)
```

Epoch=50, Accuracy=0.718
