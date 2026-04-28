# LeNet模型
LeNet是最早比较经典的CNN模型，用来识别邮件上的邮编，或者支票中的金额。

与之一起发布的还有MNIST手写数字数据集，包含了50000个0到9的手写字训练集，10000个测试数据集，图像大小为28x28，包含了10个类别。

## 1 LeNet网络结构

结构图如下

![Image](https://github.com/user-attachments/assets/76b46ca6-179f-48ef-b8c7-f1beb815f018)

包含了两个卷积层，两个池化层和三个全连接层

LeNet是早期成功的升级网络
先用卷积层来学习图片的空间信息
然后使用全连接层来转换到类别空间

## 2 用pytorch实现LeNet模型

使用pytorch构建LeNet模型
```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  #Padding=2将图片填充为32*32
    nn.AvgPool2d(kernel_size=2, stride=2),  #均值池化层
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 输入通道是6,输出通道是16
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),   # 用Flatten层将输入变成一维的向量，才能被全连接层处理
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```
可以手动打印一下每一层的输出维度

```python
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

# 打印每一层的输出维度
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

输出：
Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
Flatten output shape: 	 torch.Size([1, 400])
Linear output shape: 	 torch.Size([1, 120])
Sigmoid output shape: 	 torch.Size([1, 120])
Linear output shape: 	 torch.Size([1, 84])
Sigmoid output shape: 	 torch.Size([1, 84])
Linear output shape: 	 torch.Size([1, 10])

导入MNIST数据集

```python
# 训练数据和测试数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

模型导入GPU中训练，需要构建一个用于GPU上的评估函数

```python
# 构建评估函数
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):    # 因为教程中会实现手写的版本和torch.nn的版本，这里进行判断 
        net.eval() # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device  # 看你的网络存在哪里,直接按照你网络存储的位置
    
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)    # 构建一个累加器来存储和计算测试数据和评估精度
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]   # 是list则每个数据都要移动到device中去
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel()) # y.numel表示y的元素个数,最终计算准确率
    return metric[0] / metric[1]
```

构建训练函数，让数据在GPU中训练

```python
# 构建训练函数,让数据在GPU中训练
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)   # 使用xavier初始化权重参数     
    net.apply(init_weights)  # 权重初始化
    
    print('training on', device)  # 打印在哪里训练
    net.to(device)  # 把模型参数搬到GPU上

    # 优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  
    loss = nn.CrossEntropyLoss()

    # 动画一下训练效果
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                    legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()   # 初始化梯度
            X, y = X.to(device), y.to(device)   # 将输入输出移动到GPU中
            y_hat = net(X)
            l = loss(y_hat, y)    # 计算损失
            l.backward()
            optimizer.step()      # 更新参数
            
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches- 1:
                animator.add(epoch + (i + 1) / num_batches,
                            (train_l, train_acc, None))
                
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 评估模型
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')
```

开始训练，并打印训练结果

```python
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
结果：
loss 0.467, train acc 0.824, test acc 0.786
27559.6 examples/sec on cuda:0

![Image](https://github.com/user-attachments/assets/870e3e10-a7f2-472f-bd7b-9d7a981d13c5)

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/11.1CNN_LeNet.ipynb)

