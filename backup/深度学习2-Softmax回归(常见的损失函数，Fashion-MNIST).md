# Softmax回归

softmax回归其实是一个分类问题。

**回归**估计一个预测值，比如房价预测。

> 单连续数值输出；
> 自然区间R；
> 跟真实值的区别作为损失

**分类**预测一个离散的类别，比如手写数字识别(MNIST)，自然物体分类(ImageNet)。

>通常是多个输出；
>输出i是预测为第i类的置信度；

## 1 Softmax回归的过程

1.对分类标签进行one-hot编码

​	   y=[y1, y2, y3,....,y_n]^T

​	如果y_i是真是的预测类别则y_i=1，其他数值为0

2.使用softmax来输出匹配概率(非负，和为1)

<img src="https://latex.codecogs.com/svg.image?\hat{y}_i=\frac{\text{exp}(o_i)}{\sum_k\text{exp}(o_k)}" title="\hat{y}_i=\frac{\text{exp}(o_i)}{\sum_k\text{exp}(o_k)}" />

​使用概率y和\hat{y}的区别来作为损失

3.使用交叉熵损失函数

交叉熵来衡量两个概率的区别

<img src="https://latex.codecogs.com/svg.image?H(p,q)=\sum_i-p_i\log(q_i)" title="H(p,q)=\sum_i-p_i\log(q_i)" />

将其作为损失:

<img src="https://latex.codecogs.com/svg.image?L(y,\hat{y})=-\sum_i&space;y_i\log\hat{y}_i=-\log\hat{y}_y" title="L(y,\hat{y})=-\sum_i y_i\log\hat{y}_i=-\log\hat{y}_y" />

## 2 常见的损失函数

1.**L2 Loss（均方误差）**

<img src="https://latex.codecogs.com/svg.image?l(y,y^,)=\frac{1}{2}(y-y^,)^2" title="l(y,y^,)=\frac{1}{2}(y-y^,)^2" />

​当预测值和真实值离得比较远时，更新幅度会非常大，当接近真实值的时候，更新幅度会变小，这谁又L2 loss的梯度来决定的。当然有时候我们并不希望更新幅度多大，因此会使用L1损失。

2.**L1 Loss**

<img src="https://latex.codecogs.com/svg.image?l(y,y^,)=|y-y^,|" title="l(y,y^,)=|y-y^,|" />

​当预测值不等于真实值时，倒数都是一个常数，当预测值大于0时，导数为1，小于0时导数为-1，这可以带来很多稳定性上的好处。但是该损失函数在0点出不可导，当预测值接近真实值时，训练会很不稳定。

3.**Huber's Robust Loss**

​该损失函数结合了L1和L2损失函数

<img src="https://latex.codecogs.com/svg.image?l(y,y^,)=\left\{\begin{matrix}|y-y^,|-\frac{1}{2}&if|y-y^,|>1\\\frac{1}{2}(y-y^,)^2&otherwise\end{matrix}\right." title="l(y,y^,)=\left\{\begin{matrix}|y-y^,|-\frac{1}{2}&if|y-y^,|>1\\\frac{1}{2}(y-y^,)^2&otherwise\end{matrix}\right." />

## 3 数据的处理

下载和处理Fashion-MNIST数据集
[查看链接](https://github.com/kxmust/Deep_learning_note/blob/main/2.1picture_classification_data.ipynb)

## 4 Softmax回归从0开始实现
详细的代码和讲解[点击](https://github.com/kxmust/Deep_learning_note/blob/main/2.2softmax_regression.ipynb)

## 5 使用pytorch简单实现Softmax回归
详细的代码和讲解[点击](https://github.com/kxmust/Deep_learning_note/blob/main/2.3softmax_regression_simple.ipynb)
