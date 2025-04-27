# AlexNet深度卷积神经网络
AlexNet是引起了深度学习热潮的第一个网络

> 在深度学习之前，最常用的机器学习方法是核方法
首先提取特征
利用核函数来计算相关性，判断高维空间内两个点是否有相关性
经过核函数处理之后就会变成凸优化问题
有非常好的定理和数学基础

现在SVM支持向量机也被广泛使用，因为它不怎么需要调参，对参数不怎么敏感

在早些年，计算机视觉的工作主要在特征提取方面
如果将原始图像直接输入SVM效果还非常差
因此，需要科学家或者工程师都提出了大量的方法来抽取图片中的特征信息

AlexNet 赢得了2012年的ImageNet竞赛

## 1 AlexNet模型

主要改进：
Dropout  (模型变大了，用dropout来正则)
ReLU
MaxPooling

AlexNet就是一个更深更大的LeNet网络，两个网络结果对比如下：

![Image](https://github.com/user-attachments/assets/feb4ae15-401d-4cec-9484-b2e6dd84d3f7)


相比于LeNet，AlexNet使用了更大的卷积核，更大的步长，因为输入的图片更大，并且使用了更大的池化窗口，使用了MaxPooling而不是AvgPooling
并且增加了更多的卷积层
最后也用了三层全连接层

更多细节：
激活函数从Sigmoid变为了ReLU，减缓梯度消失
隐藏全连接层后加入了Dropout层
做了数据增强(将图片做了很多变化，随机截取，调节亮度，随机调节色温来增加数据的变种)

AlexNet的参数量大概是46M，LeNet大概有0.6M
AlexNet做一次先前计算大概比LeNet贵了250倍

总结：
AlexNet是一个更深的LeNet，10X的参数个数，250X的计算复杂度
新引入了丢弃法(Dropout)，ReLU，最大池化层，和数据增强
AlexNet赢下了2012年的ImageNet竞赛，标志着新一轮的神经网络热潮的开始

## 2 AlexNet的代码实现

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),  #因为训练用的是MNIST因此输入通道是1
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
```

手动打印每一层的输出维度

```python
# 打印出每一层的输出维度
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

输出：
Conv2d output shape:	 torch.Size([1, 96, 54, 54])
ReLU output shape:	 torch.Size([1, 96, 54, 54])
MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
Conv2d output shape:	 torch.Size([1, 256, 26, 26])
ReLU output shape:	 torch.Size([1, 256, 26, 26])
MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
Conv2d output shape:	 torch.Size([1, 384, 12, 12])
ReLU output shape:	 torch.Size([1, 384, 12, 12])
Conv2d output shape:	 torch.Size([1, 384, 12, 12])
ReLU output shape:	 torch.Size([1, 384, 12, 12])
Conv2d output shape:	 torch.Size([1, 256, 12, 12])
ReLU output shape:	 torch.Size([1, 256, 12, 12])
MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
Flatten output shape:	 torch.Size([1, 6400])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 10])

读取和训练数据

```python
# 读取数据集
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224) #将图片拉到224X224匹配模型需求

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
结果：
loss 0.328, train acc 0.879, test acc 0.882
918.8 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/43023432-747c-4420-8a8d-689094a5de9c)

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/12.1AlexNet.ipynb)

