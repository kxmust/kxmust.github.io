# GoogLeNet模型

它是一种含并行连接的网络
之前的模型比如LeNet，AlexNet，V2G用了不同规模的卷积神经网络来提取图片特征，哪到底选择哪种？

GoogLeNet的做法就是全都要

## 1 GoogLeNet模型的结构

### 1.1 Inception块
Inception块从四个不同的路径来抽取不同层面的信息，然后在输出通道维度进行合并

![Image](https://github.com/user-attachments/assets/110d6acd-620b-4d01-a1ae-22881c317bd7)

Inception块由四条并行路径组成

前三条路径使用窗口大小为1×1、3×3和5×5的卷积层，
从不同空间大小中提取信息。中间的两条路径在输入上执行1×1卷积，以减少通道数，从而降低模型的复杂性

第四条路径使用3×3最大汇聚层，然后使用1×1卷积层来改变通道数

这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出

在Inception块中，通常调整的超参数是每层输出通道数

Inception块的特点：
拥有更少的参数量和计算复杂度

### 1.2 用Inception块构建GoogLeNet

模型大致结构如下：

![Image](https://github.com/user-attachments/assets/2238407e-8d45-4969-a91b-3e8a345cdde1)

### 1.3 Inception块的各种后续变种

- Inception-BN(v2)，使用了batch normalization
- Inception-V3，修改了Inception块
替换5x5为多个3x3卷积层；
替换5x5为1x7和7x1卷积层；
替换3x3为1x3和3x1卷积层；
更深；
- Inception-V4，使用残差连接

例如：V3版本-替换5x5为1x7和7x1卷积层；

![Image](https://github.com/user-attachments/assets/8fda5500-469f-497e-8323-2c4e4f1fbacc)

### 1.4 总结
Inception块用四条不同超参数的卷积层和池化层的路来抽取不同的信息

它的一个主要优点是模型参数小，计算复杂度低

GoogLeNet使用了9个Inception块，是一个达到上百层的网络

进过后续一系列改进，拥有很高的精度，它的问题在于模型非常复杂


## 2 GoogLeNet代码实现

构建Inception块
```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 创建Inception块
class Inception(nn.Module):
  # c1--c4是每条路径的输出通道数
  def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
    super(Inception, self).__init__(**kwargs)
    # 线路1，单1x1卷积层
    self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
    # 线路2，1x1卷积层后接3x3卷积层
    self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
    self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
    # 线路3，1x1卷积层后接5x5卷积层
    self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
    self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
    # 线路4，3x3最大汇聚层后接1x1卷积层
    self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

  def forward(self, x):
    p1 = F.relu(self.p1_1(x))
    p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
    p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
    p4 = F.relu(self.p4_2(self.p4_1(x)))
    # 在通道维度上连结输出
    return torch.cat((p1, p2, p3, p4), dim=1)
```

利用Inception块构建GoogLeNet模型

```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
  nn.ReLU(),
  nn.Conv2d(64, 192, kernel_size=3, padding=1),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
  Inception(256, 128, (128, 192), (32, 96), 64),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
  Inception(512, 160, (112, 224), (24, 64), 64),
  Inception(512, 128, (128, 256), (24, 64), 64),
  Inception(512, 112, (144, 288), (32, 64), 64),
  Inception(528, 256, (160, 320), (32, 128), 128),
  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
  Inception(832, 384, (192, 384), (48, 128), 128),
  nn.AdaptiveAvgPool2d((1,1)),
  nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

打印每一层的输出

```python
# 打印每一层的输出维度
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
  X = layer(X)
  print(layer.__class__.__name__,'output shape:\t', X.shape)
```

结果：
Sequential output shape:	 torch.Size([1, 64, 24, 24])
Sequential output shape:	 torch.Size([1, 192, 12, 12])
Sequential output shape:	 torch.Size([1, 480, 6, 6])
Sequential output shape:	 torch.Size([1, 832, 3, 3])
Sequential output shape:	 torch.Size([1, 1024])
Linear output shape:	 torch.Size([1, 10])

开始训练
```python
# 开始训练
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)  #将图片改为96x96来降低训练难度
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

结果：
loss 0.252, train acc 0.903, test acc 0.888
1405.7 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/ffbd5fec-5711-4997-aeea-bac7f2b12846)

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/15.1GoogLeNet.ipynb)

