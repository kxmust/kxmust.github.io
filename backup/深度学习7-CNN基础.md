# CNN卷积神经网络
## 1 CNN基础
从多层感知机（MLP）到卷积神经网络（CNN）的演进中，​​平移不变性​​和​​局部性​​是两大核心设计原则，它们解决了传统MLP处理图像时的低效问题，并成为卷积操作的理论基础。

### 1.1 平移不变性（Translation Invariance）
通俗解释​​：无论目标出现在图像的哪个位置，模型都能识别出它的特征，而不会因为位置变化导致结果不同。
​​类比​​：就像用同一把“特征探测器”扫描整张图片，无论猫在左上角还是右下角，探测器都能捕捉到猫耳朵或尾巴的特征，并正确识别为“猫”。

​​为什么需要？​​
传统MLP为每个像素位置单独分配权重，导致模型需要为同一特征在不同位置重复学习参数（例如猫耳朵出现在左边和右边时，权重完全独立）。这不仅参数爆炸，还容易因训练数据中位置不全面导致泛化能力差。

​​卷积如何实现？​​
通过​​共享权重​​：卷积核在图像上滑动时，无论扫描到哪个区域，都使用同一组参数计算特征响应。例如，识别边缘的卷积核在图像任何位置都检测垂直或水平边缘，无需为不同位置重新学习。

### 1.2 局部性（Locality）
通俗解释​​：模型在识别特征时，只关注目标周围的局部区域，而非整张图像。
​​类比​​：像侦探用放大镜仔细查看某个小区域，判断是否有指纹或划痕，而不需要一次性观察整个犯罪现场。

​​为什么需要？​​
图像中相邻像素的关联性远高于遥远像素。例如，判断某个像素是否属于“猫耳朵”，只需看它周围的毛发纹理，而无需参考图像底部的“草地”像素。MLP的全局连接会导致参数冗余且无法聚焦局部特征。

​​卷积如何实现？​​
通过​​局部感受野​​：卷积核仅覆盖一个小窗口（如3×3或5×5像素），每次只处理窗口内的局部信息。例如，一个检测“边缘”的卷积核仅分析当前像素及其周围8个邻居的亮度变化。

### 1.3 卷积的计算过程

![Image](https://github.com/user-attachments/assets/ff70a46e-a503-4d60-bf81-fae135372aff)

从0实现
```python
import torch
from torch import nn
from d2l import torch as d2l

# 实现互相关运算
def corr2d(X, K): #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0]- h + 1, X.shape[1]- w + 1))  # 输出的维度
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```
利用构建的corr2d函数构建卷积层

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```
通过使用卷积核可以检测输入X的边
也可以给定输入和输出来学习一个K矩阵

```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2 # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat- Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:]-= lr * conv2d.weight.grad   #更新权重
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
```
[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/10.1CNN_layer.ipynb)

## 2 CNN的参数设置

### 2.1 卷积核大小、填充和步幅
设置卷积核大小，以及填充padding，还有步幅
填充表示往输入周围填充0来保证输出的大小
步幅表示卷积核移动的步长

```python
# 卷积核的大小, 填充0来控制输出维度的减少量, 步幅用来控制每次滑动窗口时滑动的行/列的步长可以成倍的减少输出形状
import torch
from torch import nn

# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])
    
# 请注意，padding=1这里每边都填充了1行或1列，因此总共添加了2行或2列
# 第一个1表示输出通道,第二个1表示输入通道
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)

X = torch.rand(size=(8,8))
comp_conv2d(conv2d, X).shape
```

通过设置卷积核大小，填充和步幅可以控制输出的维度
填充对输出维度的影响，计算过程如下：

![Image](https://github.com/user-attachments/assets/9ec1876f-05f5-46b6-80c1-b75aa20efdba)

步幅对输出维度的影响，计算过程如下：

![Image](https://github.com/user-attachments/assets/f3100731-9ec9-43fc-9935-5ea2f1541a47)

不同的卷积核大小，填充和步长的设置

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

# 设置步幅为2
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/10.2CNN_parameters.ipynb)

### 2.2 通道数

多个通道数的计算过程

![Image](https://github.com/user-attachments/assets/42f4edc3-7888-450e-9d84-8c12d3d2b887)

可以通过增加多个三维卷积核来设定多个输出通道

![Image](https://github.com/user-attachments/assets/f07a2556-800d-45d4-88cf-6d5702367b94)

1x1的卷积核不会识别空间模式，只是融合通道，可以用作后续的特征融合

![Image](https://github.com/user-attachments/assets/3a374491-65eb-4238-81e2-5036733ed3d9)

多个输入通道的从0实现

```python
# 多输入通道
import torch
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
corr2d_multi_in(X, K)
```

多输出通道的从0实现

```python
# 多输出通道
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0)
K.shape

corr2d_multi_in_out(X, K)
```

1X1的卷积核实现

```python
# 1*1 的卷积核
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1- Y2).sum()) < 1e-6
Y2
```
[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/10.3CNN_channel.ipynb)

## 3 池化层

池化层的作用有两个：
第一是让卷积核对位置信息没那么敏感（需要一定的平移不变性）
第二是可以降低输入数据的大小

池化层返回窗口中最大或者平均值
并且同样有窗口大小，填充和步幅的超参数

二维最大池化的过程

![Image](https://github.com/user-attachments/assets/6d55fbfc-202d-479c-b977-ec7e9d7b65e6)

从0开始实现池化层
```python
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0]- p_h + 1, X.shape[1]- p_w + 1))   # 输出的大小
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))

pool2d(X, (2, 2), 'avg')
```

用torch实现池化层
```python
X = torch.arange(16, dtype = torch.float32).reshape(1, 1, 4, 4)
X

# 深度学习框架中池化层的大小和步长是相同的
pool2d = nn.MaxPool2d(3)
pool2d(X)

#设置填充为1，步长为2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)


pool2d = nn.MaxPool2d((2,3), padding=(1,1), stride=(2, 3))
pool2d(X)
```

多通道池化

```python
X = torch.cat((X, X+1), 1)
X

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

```

[代码链接](https://github.com/kxmust/Deep_learning_note/blob/main/10.4CNN_pool.ipynb)