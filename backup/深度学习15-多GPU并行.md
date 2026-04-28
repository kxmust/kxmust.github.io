# 单机多卡并行
一个机器可以安装多个GPU
在训练和预测时，可以将小批量计算切分到多个GPU上来达到加速的目的

常用的切分方案有：
- 数据并行-将小批量数据分成n块，每个GPU拿到完整的参数计算一块数据的梯度（通常性能更好）
- 模型并行-（将模型分成n块，每个GPU拿到一块模型计算它的前向和反向结果，通常适用于模型大到单GPU放不下）
- 通道并行（数据+模型并行）

## 1 数据并行

以两个GPU为例：
- 在任何一次迭代中，给定的随机小批量数据将被分成两份，然后均匀分配给每个GPU
- 每个GPU会利用分配的数据计算梯度
- 每个GPU会将计算的梯度发出去，并且进行相加，以获得当前小批量的随机梯度
- 将计算的聚合梯度发送给每个GPU，每个GPU利用得到的随机梯度来更新模型参数

大致的流程图为：

![Image](https://github.com/user-attachments/assets/b10e95f5-9f1d-4768-a684-f6103fa8f600)

总结：
-  当一个模型能用到单卡计算时，通常使用数据并行拓展到多卡上
- 模型并行则用在超大模型上

## 2 数据并行的从0实现

以LeNet为例来来实现

- 先构建LeNet模型

```python
%matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]


# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0],-1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat
    
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')
```

- 向多个设备分发参数

```python
# 给一个参数，然后将它发送到哪个GPU上
def get_params(params, device):  
    new_params = [p.to(device) for p in params] 
    for p in new_params:    # 需要计算梯度
        p.requires_grad_()
    return new_params

# 将所有参数复制到一个GPU
new_params = get_params(params, d2l.try_gpu(0))
print('b1 权重:', new_params[1])
print('b1 梯度:', new_params[1].grad)
```

- 使用allreduce函数将所有向量相加，并将结果广播给所有GPU

```python
# 将所有向量相加,并广播给所有GPU
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)  # 将数据复制到GPU0上进行相加
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)  # GPU0计算得到的结果发送给所有GPU

data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之后：\n', data[0], '\n', data[1])
```

- 将一个小批量数据均匀分布在多个GPU上

```python
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)  # 将数据均匀切开，分配给每个GPU上
print('input :', data)
print('load into', devices)
print('output:', split)

```

- 为了方便使用，定义一个split_batch函数，将数据和标签都进行拆分

```python
def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

- 定义一个多GPU训练小批量数据的函数

```python
# 定义一个多GPU训练小批量数据的函数
def train_batch(X, y, device_params, devices, lr): # 输入,所有GPU上的参数，GPU,学习率
    X_shards, y_shards = split_batch(X, y, devices)  # 拆分数据
    # 在每个GPU上分别计算损失, ls中包含了所有GPU中的损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
                    for X_shard, y_shard, device_W in zip(
                        X_shards, y_shards, device_params)]
    for l in ls: # 反向传播在每个GPU上分别执行
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量
```

- 开始训练

```python
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
        # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()  # 同步一次,保证每个GPU都完成了
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
            f'在{str(devices)}')

# 单GPU运行
train(num_gpus=1, batch_size=256, lr=0.2)

# 多GPU运行
train(num_gpus=2, batch_size=256, lr=0.2)
```

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/18.1parallel_computing.ipynb)

## 3 数据并行的简单实现

- 首先构建模型

```python
import torch
from torch import nn
from d2l import torch as d2l

# 定义模型
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                    first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels,
                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
    nn.Linear(512, num_classes)))
    return net
```

- 网络初始化

```python
# 网络初始化
net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()
# 我们将在训练代码实现中初始化网络
```

- 训练函数

```python
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices) #给定一个网络,给定GPU，然后将Net复制到每个GPU上
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
        f'在{str(devices)}')

# 单卡训练
train(net, num_gpus=1, batch_size=256, lr=0.1)
# 多卡训练
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/18.2parallel_computing_simple.ipynb)
