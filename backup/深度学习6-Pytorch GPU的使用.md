# GPU的使用

使用!nvidia-smi来查看GPU状态
```python
!nvidia-smi
```

```python
import torch
from torch import nn
torch.device('cpu'), torch.cuda.device('cuda') # torch.cuda.device('cuda:1') 访问第一个GPU

```

定义两个函数来测试是否存在GPU如果没有则用CPU

```python
def try_gpu(i=0): #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
    
def try_all_gpus(): #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
    
try_gpu(), try_gpu(10), try_all_gpus()
```

查询张量坐在的设备

```python
x = torch.tensor([1, 2, 3])
x.device
```
创建张量时，放在GPU上

```python
# 创建张量时，放在GPU上
X = torch.ones(2, 3, device=try_gpu())
X

# 也可以在第二个GPU上创建张量
# Y = torch.rand(2, 3, device=try_gpu(1))
```

如果你有多个GPU，可以将一个GPU上的值copy到另一个GPU上
Z = X.cuda(1)  可以将X的值从GPU0 copy到GPU1
数值在一个GPU上才能做计算

在GPU上做神经网络

```python
# 在GPU上做神经网络
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())  # net.to将神经网络挪动到0号GPU上 ,等于将网络的参数在0号GPU上copy一份

net(X)  # x也在0号GPU上
net[0].weight.data.device  # 查看权重参数所在的位置
```




