# RNN

Recurrent Neural Network 循环神经网络 学习笔记

***

## RNN

循环神经网络（Recurrent Neural Network, RNN）是一类具有内部环状连接的人工神经网络，用于处理序列数据。其最大特点是网络中存在着环，使得信息能在网络中进行循环，实现对序列信息的存储和处理。

### 模型结构

![r1](./img/rnn1.png)

首先我们来看RNN的网络结构，我们可以看出RNN和传统MLP的区别就是RNN有个自循环，而这个循环同时可以视为一个不断重复执行的模块，这个模块的输入有上个循环的结果 $H_{t-1}$ 和 传统MLP输入 $x_t$ ，输出则是 $H_t$ 。

而这个多出来的输入参数 $H_{t-1}$ 则被成为隐状态，即隐藏的状态，用于记录上个时间的信息。其方程如下：

$$
H_t=\phi(X_tW_{xh}+H_{t-1}W_{hh}+b_h)
$$

其中 $\phi$ 为激活函数一般为`tanh`函数， $X_t$ 为当前时刻输入， $H_{t-1}$ 为上一时刻隐状态， $W_{xh}$ 和 $W_{hh}$ 为权重矩阵， $b_h$ 为偏置。

由于在当前时间步中， 隐状态使用的定义与前一个时间步中使用的定义相同， 因此计算是循环的（recurrent）。 于是基于循环计算的隐状态神经网络被命名为循环神经网络（recurrent neural network）。 在循环神经网络中执行计算的层称为循环层（recurrent layer）。

而记录隐状态之后，还需要输出：

$$
O_t=H_tW_{hq}+b_q
$$

而输出则是当前隐状态值和权重的线性组合。

![r2](./img/rnn2.svg)

这样从公式看来，RNN和传统MLP相比就是多了个隐状态的线性组合。

此外需要注意的是，每个循环的各个权重矩阵和偏置都是相同的，变的部分只是输入和隐状态。

### 优缺点

- 优点
  - 能够处理不同长度的序列数据。
  - 能够捕捉序列中的时间依赖关系。
- 缺点
  - 对长序列的记忆能力较弱，可能出现梯度消失或梯度爆炸问题。
  - 训练可能相对复杂和时间消耗大。

***

## LSTM

***

## GRU

***

## Bi-RNN

***

## 参考

[ML Lecture 21-1: Recurrent Neural Network (Part I)](https://www.youtube.com/watch?v=xCGidAeyS4M&ab_channel=Hung-yiLee) - 李宏毅教授的详细RNN和LSTM讲解

[LSTM（长短期记忆神经网络）最简单清晰的解释来了！](https://www.bilibili.com/video/BV1zD421N7nA/) - StatQuest
的RNN和LSTM讲解动画

[动手学深度学习](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html)

[循环神经网络RNN完全解析：从基础理论到PyTorch实战](https://cloud.tencent.com/developer/article/2348483)