## 简介

这个项目是我在学习PyTorch及其运用的学习笔记和相关代码存放项目，大家可以自主取用和提意见。

本项目主要使用python和jupyter notebook和pytorch相关技术。

## 知识点集合

* NN 神经网络 DL 深度学习
  * 核心内容
    * 梯度下降
    * 反向回归
    * 自动微分
    * 损失函数
    * 优化器（梯度下降的方法）
  * MLP 全连接神经网络
    * 激活函数
  * CNN 卷积神经网络
    * 卷积层
    * 池化层
    * AlexNet
    * 残差网络
  * RNN 循环神经网络
    * LSTM
    * GRU
    * Seq2Seq RNN
    * 双向RNN
  * Transformer
    * 注意力机制
    * Encoder-Decoder
    * GPT
    * Bert
  * 运用
    * CV 计算机视觉
      * OD 目标识别
        * YOLO(CNN)
        * R-CNN(CNN)
        * DETR(Tranformer)
    * NLP 自然语音处理
      * 文本表示
        * 词向量
        * 独热编码
        * Word2Vec
        * Embbeding
        * Tokenize
      * RNN相关
      * Tranformer以及LLM相关

## 文件
* [nn.ipynb](./nn.ipynb) - torch.nn相关学习笔记，包含卷积层，池化层，激活函数，归一化，线性层，损失函数等内容
* [autograd.ipynb](./autograd.ipynb) - torch.autograd相关学习笔记
* tensor.ipynb - torch.tensor相关学习笔记
* cnn - 卷积神经网络相关练习项目代码和学习笔记
  * [od](./cnn/od/README.md) - 目标检测简介和相关内容
    * rcnn - RCNN学习笔记和代码
      * [rcnn.md](./cnn/od/rcnn/rcnn.md)
    * yolo - YOLO学习笔记和代码
      * [yolo.md](./cnn/od/yolo/yolo.md)
* rnn - 循环神经网络相关练习项目代码和学习笔记
  * [rnn.md](./rnn/rnn.md) RNN/LSTM/GRU学习笔记
* transformer - Tranformer相关练习项目代码和学习笔记
  * [transformer.md](./transformer/transformer.md) Transformer学习笔记和代码
  * transformer.py transformer模型实现代码
  * detr - DETR学习笔记和代码
    * [detr.md](./transformer/detr/detr.md)
