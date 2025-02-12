## Model

```python
# setup model
class MyModel(nn.Module):
    """
    简单的MLP模型
    """
    def __init__(self, input_dim, hidden_layers_dim=[64, 32, 8]):
        super().__init__()
        self.layers = []
        for i in range(len(hidden_layers_dim)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_layers_dim[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers_dim[i-1], hidden_layers_dim[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers_dim[-1], 1))
        self.layers = nn.Sequential(*self.layers)
      
    def forward(self, x):
        x = self.layers(x)  # [B, 1]
        x = x.squeeze(1)
        return x

MyModel(88)
```

## Optimizer

```python
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.7)
# 经测试发现，原始的SGD优化器优于Adam优化器
```

## Config

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 200,     # Number of epochs.          
    'batch_size': 256, 
    'learning_rate': 1e-5,            
    'early_stop': 600,    # If model has not improved for this many consecutive epochs, stop training.   
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
```

## Training loss

```python
Epoch [200/200]: 100%|██████████| 10/10 [00:00<00:00, 173.69it/s, loss=1.98]
Epoch [200/200]: Train loss: 1.6976, Valid loss: 1.6309
```
