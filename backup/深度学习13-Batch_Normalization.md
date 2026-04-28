# Batch Normalization 批量归一化
考虑一个问题，当一个网络很深的情况下
梯度反传过程中，上面的梯度会更大，下面的梯度会更小
这样会导致上面的网络会很快收敛，而下面的网络会更新很慢

这样会有一个什么问题呢？
当下面的参数发生变化后，对输入抽象出来的特征会发生变化，则上面的网络又需要重新训练
导致收敛变慢

因此我们可以用batch normalization 来将输出的均值和方差进行固定，
这样的好处是在训练底部时可以避免变化顶部层

## 1 批量归一化

对于一个批量数据，计算均值和方差

![Image](https://github.com/user-attachments/assets/ae2e4981-d004-4362-ba02-249d5e4b4f6a)

利用均值和方差来进行归一化

![Image](https://github.com/user-attachments/assets/d56f9e07-9069-4606-9060-715ed691bee4)

特点：
可以学习的参数是\beta 和 \gamma
作用的位置：
 全连接层和卷积层的输出上，激活函数前
 全连接层和卷积层的输入上
对于全连接层，作用在特征维
对于卷积层，作用在通道层

总结：
批量归一化固定小批量的均值和方差，然后学习出适合的偏移的缩放
可以加速收敛速度，但一般不改变模型精度

## 批量归一化实现

从0实现批量归一化

```python
import torch
from torch import nn
from d2l import torch as d2l

# 批量归一化从0开始实现,（输入，（可以学的两个参数），全局的均值，全局的方差，避免除0，用来更新全局均值和方差的）
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
  # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
  if not torch.is_grad_enabled():
    # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
    X_hat = (X- moving_mean) / torch.sqrt(moving_var + eps)
  else:
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
      # 使用全连接层的情况，计算特征维上的均值和方差
      mean = X.mean(dim=0)
      var = ((X- mean) ** 2).mean(dim=0)
    else:
      # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
      # 这里我们需要保持X的形状以便后面可以做广播运算
      mean = X.mean(dim=(0, 2, 3), keepdim=True)
      var = ((X- mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    # 训练模式下，用当前的均值和方差做标准化
    X_hat = (X- mean) / torch.sqrt(var + eps)
    # 更新移动平均的均值和方差
    moving_mean = momentum * moving_mean + (1.0- momentum) * mean
    moving_var = momentum * moving_var + (1.0- momentum) * var
  Y = gamma * X_hat + beta # 缩放和移位
  return Y, moving_mean.data, moving_var.data
```

构建归一化层

```python
class BatchNorm(nn.Module):
  # num_features：完全连接层的输出数量或卷积层的输出通道数。
  # num_dims：2表示完全连接层，4表示卷积层
  def __init__(self, num_features, num_dims):
    super().__init__()
    if num_dims == 2:
      shape = (1, num_features)
    else:
      shape = (1, num_features, 1, 1)
    # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
    self.gamma = nn.Parameter(torch.ones(shape))
    self.beta = nn.Parameter(torch.zeros(shape))
    # 非模型参数的变量初始化为0和1
    self.moving_mean = torch.zeros(shape)
    self.moving_var = torch.ones(shape)

  def forward(self, X):
    # 如果X不在内存上，将moving_mean和moving_var
    # 复制到X所在显存上
    if self.moving_mean.device != X.device:
      self.moving_mean = self.moving_mean.to(X.device)
      self.moving_var = self.moving_var.to(X.device)
    # 保存更新过的moving_mean和moving_var
    Y, self.moving_mean, self.moving_var = batch_norm(
      X, self.gamma, self.beta, self.moving_mean,
      self.moving_var, eps=1e-5, momentum=0.9)
    return Y
```

在LeNet中引入批量归一化

```python
# 实现LeNet
net = nn.Sequential(
  nn.Conv2d(1, 6, kernel_size=5), 
  BatchNorm(6, num_dims=4), # 6表示通道数
  nn.Sigmoid(),
  nn.AvgPool2d(kernel_size=2, stride=2),
  nn.Conv2d(6, 16, kernel_size=5), 
  BatchNorm(16, num_dims=4), 
  nn.Sigmoid(),
  nn.AvgPool2d(kernel_size=2, stride=2), 
  nn.Flatten(),
  nn.Linear(16*4*4, 120), 
  BatchNorm(120, num_dims=2), 
  nn.Sigmoid(),
  nn.Linear(120, 84), 
  BatchNorm(84, num_dims=2), 
  nn.Sigmoid(),
  nn.Linear(84, 10))
```

打印输出维度

```python
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

# 打印每一层的输出维度
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

结果：
Conv2d output shape: 	 torch.Size([1, 6, 24, 24])
BatchNorm output shape: 	 torch.Size([1, 6, 24, 24])
Sigmoid output shape: 	 torch.Size([1, 6, 24, 24])
AvgPool2d output shape: 	 torch.Size([1, 6, 12, 12])
Conv2d output shape: 	 torch.Size([1, 16, 8, 8])
BatchNorm output shape: 	 torch.Size([1, 16, 8, 8])
Sigmoid output shape: 	 torch.Size([1, 16, 8, 8])
AvgPool2d output shape: 	 torch.Size([1, 16, 4, 4])
Flatten output shape: 	 torch.Size([1, 256])
Linear output shape: 	 torch.Size([1, 120])
BatchNorm output shape: 	 torch.Size([1, 120])
Sigmoid output shape: 	 torch.Size([1, 120])
Linear output shape: 	 torch.Size([1, 84])
BatchNorm output shape: 	 torch.Size([1, 84])
Sigmoid output shape: 	 torch.Size([1, 84])
Linear output shape: 	 torch.Size([1, 10])

开始训练

```python
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

结果：
loss 0.268, train acc 0.900, test acc 0.791
17149.0 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/d7d4dcce-668a-4e96-96a4-732e1752d11f)

查看\gamma和\beta的参数

```python
# 查看gamma和beta的参数
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

结果：
(tensor([1.8129, 2.6654, 3.1536, 4.2873, 3.2546, 1.9251], device='cuda:0',
        grad_fn=<ViewBackward0>),
 tensor([ 2.2025, -2.3604,  3.3588,  1.6798, -1.3494, -2.2466], device='cuda:0',
        grad_fn=<ViewBackward0>))

使用pytorch简单实现批量归一化

```python
# 简单实现
net = nn.Sequential(
  nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
  nn.AvgPool2d(kernel_size=2, stride=2),
  nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
  nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
  nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
  nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
  nn.Linear(84, 10))
```

开始训练

```python
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

结果：
loss 0.266, train acc 0.902, test acc 0.773
29665.3 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/72d27c26-41f2-43ad-89bd-6fa14e265c3b)

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/16.1batch_normalization.ipynb)