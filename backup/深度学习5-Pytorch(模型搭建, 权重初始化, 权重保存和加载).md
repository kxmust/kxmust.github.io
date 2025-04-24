# Pytorch基础
## 1 Pytorch基础知识

用nn.Sequential模组搭建一个简单的神经网络
```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)

net(X)

```
nn.Sequential是一个模组，我们也可以自己定义一个模组来构建神经网络

```python

class MLP(nn.Module):   # 表示是nn模组的子类
    def __init__(self):   # 定义有哪些参数
        super().__init__()  # 调用一个父类，来先设置需要的内部的参数
        self.hidden = nn.Linear(20, 256)   # 定义一个线下层存在一个成员变量中
        self.out = nn.Linear(256,10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
net(X)

```

如何实现前面的nn.Sequential模组
```python

class MySequential(nn.Module):
    def __init__(self, *args):  #*args表示接入一个有序列表, **表示字典
        super().__init__()
        for block in args:
            self._modules[block] = block  # 是一个有序的字典,_modules是一个特殊的容器，表示放进去的是每一层网络

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)    #利用每一层网络处理输入X
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)

```

当Sequential这个类无法满足我们的计算需求时,我们可以自定义类来实现特殊的计算

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)
```

可以嵌套使用各种模组中的子类

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
        
    def forward(self, X):
        return self.linear(self.net(X))
        
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)

```
[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/8.1pytorch_basic.ipynb)

## 2 参数的访问和权重初始化

### 2.1 参数的访问
```python
# pytorch如何访问参数
import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

X = torch.rand(size=(3,4))
net(X)

# 把每一层中的权重打印出来
print(net[2].state_dict())   #显示的是nn.Linear(8,1)

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

构建一个嵌套的块
```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())   # 嵌套四个block1
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

print(rgnet)  # 打印网络结构
```
查看网络结构

![Image](https://github.com/user-attachments/assets/5761c153-0587-4188-8411-13cb9594b688)

访问第一个主要的块中、第二个子块的第一层的偏置项
```python
rgnet[0][1][0].bias.data
```
### 2.2 权重的初始化

```python
# 如何初始化网络参数
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
        
net.apply(init_normal)  # apply表示对所有net中所有层进行遍历
net[0].weight.data[0], net[0].bias.data[0]
```
```python
# 比如将权重参数初始化为1
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
        
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

利用xavier进行初始化,让训练更稳定

```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 将参数初始化为42
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# 对每个层用不同的初始化
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

还可以自定义初始化
```python
# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", 
              *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10, 10)  # 均匀初始化
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```
当然也可以直接初始化

```python
# 可以直接初始化
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

有时我们希望在多个层间共享参数：我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数
```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
# 无论怎么更新net网络,两个shared的网络参数应该是相同的
net = nn.Sequential(nn.Linear(4, 8), 
                    nn.ReLU(),
                    shared, 
                    nn.ReLU(),
                    shared, 
                    nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100

# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/8.2pytorch_parameter.ipynb)

## 3 自定义层
我们可以自己定义需要的层，以满足计算需求
```python
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return X- X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
```

自定义一个线性层
```python
# 自定义线性层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
linear.weight
```
使用自己的线性层
```python
linear(torch.rand(2, 5))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```
[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/8.3pytorch_layer.ipynb)

## 4 模型参数的保存和加载
可以用torch.load来保存向量、列表或者字典，然后用torch.load来加载

```python
import torch
from torch import nn
from torch.nn import functional as F

# 存一个向量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
x2

# 存储一个list
y = torch.zeros(4)
torch.save([x, y],'x-files')

x2, y2 = torch.load('x-files')
(x2, y2)

# 存一个字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')

mydict2 = torch.load('mydict')
mydict2
```

加载和保存模型的参数
```python
# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
        
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 存储模型的参数
torch.save(net.state_dict(), 'mlp.params')

# 加载保存的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))  # 加载保存的参数
clone.eval()

```
[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/8.4pytorch_save_model.ipynb)





