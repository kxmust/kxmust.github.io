# VGG模型
AlexNet比LeNet更深更大，得到更好的精度
能不能设计的更深更大？

有哪些选项：
更多全连接层（太贵）
更多的卷积层
将卷积层组合成块

## 1 VGG的结构
VGG块将多个卷积层和一个池化层组合成块：

![Image](https://github.com/user-attachments/assets/d1f806b7-8baa-4021-9eab-d59295263eeb)

它这里面用了3x3的卷积核，原因是在相同的计算开销下，3x3比5
x5的卷积核拥有更好的效果。

用多个VGG块就可以构成一个VGG网络：

![Image](https://github.com/user-attachments/assets/d9874633-54da-43ce-976f-5e25019bc5a6)

VGG其实可以看作是一个更大更深的AlexNet

总结：
VGG使用可重复使用的卷积块来构建深度卷积神经网络
不同的卷积块个数和超参数可以得到不同复杂度的变种

## 2 VGG模型的实现

首先是构建VGG块：
```python
import torch
from torch import nn
from d2l import torch as d2l

# VGG块的实现
def vgg_block(num_convs, in_channels, out_channels): # 几层CNN，输入的通道,输出的通道
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))    # 最后加一个最大池化层
    return nn.Sequential(*layers)
```

然后设计一个VGG11模型

```python
# 一个五块的VGG
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 实现VGG11, 8+3
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
    
net = vgg(conv_arch)
```

手动打印每一层的输出维度

```python
# 打印输出维度
X = torch.randn(size=(1, 1, 224, 224))

for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

结果：
Sequential output shape:	 torch.Size([1, 64, 112, 112])
Sequential output shape:	 torch.Size([1, 128, 56, 56])
Sequential output shape:	 torch.Size([1, 256, 28, 28])
Sequential output shape:	 torch.Size([1, 512, 14, 14])
Sequential output shape:	 torch.Size([1, 512, 7, 7])
Flatten output shape:	 torch.Size([1, 25088])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 10])

为了方便实验，降低了VGG的参数，然后开始训练

```python
ratio = 4

# 因为网络太大,训练难度大不利于演示，因此这个减小通道数
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# 训练模型
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

```

结果：
loss 0.170, train acc 0.938, test acc 0.925
614.9 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/ef2d754a-c07a-4519-b677-5e25d8ffe155)

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/13.1VGG.ipynb)