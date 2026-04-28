# ResNet残差网络

考虑这样一个问题，网络是不是做的越深越好
网络越深越能拟合更加复杂的问题？
但是随着网络的加深，模型训练过程中可能会发生偏差，可能性能不会增加，而开销会增大

我们要解决的问题是：当增加新的层的时候至少不会让模型变差。

## 1 ResNet模型

### 1.1 残差块
通过串联一个层来改变函数类，我们希望能扩大函数类
残差块加入快速通道（右边）来得到f(x)=x+g(x)

![Image](https://github.com/user-attachments/assets/8414e04e-57c9-4d19-829f-6d032d85ab66)

**可以这样理解：**
如果左边f(x)为新加的层，如果没有它的话，意味着这个层什么都没学到
但是下面的输入X还是会过去，到下一个网络
这样做的好处是，就算加了新的层，至少不会让模型的性能变差

也可以理解为，这样设计利用残差的传递来让模型先拟合小网络。

有两中残差块，第一个是输出维度不变，另一种来降低输出维度，高宽简单，但是增加通道数，残差部分也会加入1x1卷积来增加通道，这样才能相加

![Image](https://github.com/user-attachments/assets/1a4ebc6a-785f-477a-909d-4f41e5b33814)

当然这个残差部分可以加在任何地方，比如在ReLU之后，或者卷积之后都可以

### 1.2 ResNet模型
通过不同的残差块来构建ResNet模型
下面是ResNet-18的模型

![Image](https://github.com/user-attachments/assets/99ebf04d-5cef-4f08-a2fc-0c7047795922)

总结：
残差块使得很深的网络更加容易训练，甚至可以训练一千层

残差网络对随后的深层神经网络设计产生了深远影响，无论是卷积类网络还是全连接网络

## 2 ResNet模型的实现

实现残差块

```python
# 实现残差块
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module): 
  def __init__(self, input_channels, num_channels,
                use_1x1conv=False, strides=1):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, num_channels,
                  kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(num_channels, num_channels,
                    kernel_size=3, padding=1)
    if use_1x1conv:  # 判断是否使用1x1卷积
      self.conv3 = nn.Conv2d(input_channels, num_channels,
                kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)

  def forward(self, X):
    Y = F.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    return F.relu(Y)
```

查看两种残差块的区别，第一种是输出维度不变，另一种是高宽减半，但是通道数增加一倍

```python
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape

#结果： torch.Size([4, 3, 6, 6])

# 增加通道数的同时，减半输出的高和宽
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape

# 结果：torch.Size([4, 6, 3, 3])

```

构建ResNet模型

```python
# ResNet的前两层和之前的GoogLeNet是一样的
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
        first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(Residual(input_channels, num_channels,
              use_1x1conv=True, strides=2))
    else:
      blk.append(Residual(num_channels, num_channels))
  return blk

# 构建后面的网络
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Linear(512, 10))
```

打印每一层的维度

```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
  X = layer(X)
  print(layer.__class__.__name__,'output shape:\t', X.shape)
```
结果：
Sequential output shape:	 torch.Size([1, 64, 56, 56])
Sequential output shape:	 torch.Size([1, 64, 56, 56])
Sequential output shape:	 torch.Size([1, 128, 28, 28])
Sequential output shape:	 torch.Size([1, 256, 14, 14])
Sequential output shape:	 torch.Size([1, 512, 7, 7])
AdaptiveAvgPool2d output shape:	 torch.Size([1, 512, 1, 1])
Flatten output shape:	 torch.Size([1, 512])
Linear output shape:	 torch.Size([1, 10])

开始训练

```python
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

结果：
loss 0.015, train acc 0.996, test acc 0.903
1749.4 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/ec702370-965b-4dc6-8a8d-816b050d1510)

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/17.1ResNet.ipynb)

