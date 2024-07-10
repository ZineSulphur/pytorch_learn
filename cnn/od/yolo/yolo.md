# YOLO

You Only Look Once

我只看一次就可以识别出来了，要的就是又快又准。

## 总体介绍

- YOLO系列的核心思想就是把目标检测转变为一个回归问题，利用整张图片作为网络的输入，通过神经网络，得到边界框的位置及其所属的类别。
- 经典 one-stage 方法

***

## YOLOv1

论文[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

### 处理流程

1. 输入图片resize为448x448x3
2. 将图片分割成SxS的网格，一般为7x7网格
3. 每个网格设置两个候选框，预设候选框大小(h,w)按照经验值设置，框的位置(x,y)默认为中心点
4. 得到真实值和候选框的IoU，并且选择更合适的框
5. 对候选框进行微调，即预测候选框的(x,y,h,w)和置信度c，以及物体的分类

![yolov1核心思想](./img/yolov1_2.png)

### 模型结构

![yolov1结构图](./img/yolov1_1.png)

- 主干网络（GoogleLeNet）：输入图像resize到448x448x3，经过多次卷积池化得到7x7x1024的特征图。
- 检测头：将特征图全连接展开，第一个全连接得到4096个特征，第二个全连接之后得到7x7x30的输出。其中7x7为图片对应网格数，30为每个网格的特征值，即10个为两个候选框的(x,y,w,h,c)，20为20个分类即每个类别的预测概率

### 损失函数

![yolov1损失函数](./img/yolov1_3.png)

### 优缺点

- 优点
  - 快速，简单
- 缺点
  - 每个cell只预测一个类别，重叠无法解决
  - 小物体检测效果一般，长宽比可选但单一

***

## YOLOv2

论文[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

### 改进内容

- Batch Normalization 
 
    归一化有助于解决梯度消失和梯度爆炸的问题，降低一些超参数的敏感性（比如学习率、网络参数的大小范围、激活函数的选择），并且起到了一定的正则化效果，从而能够获得更好的收敛速度和收敛效果。

- 高分辨率分类器

    和YOLOv1一样，他们在ImageNet以224x224的分辨率对模型进行了预训练。然而，这一次，他们在分辨率为448x448的ImageNet上对模型进行了10次微调，提高了网络在高分辨率输入下的性能。

- 全卷积层

    去除了全连接层，采用了全卷积结构。

- 锚框 Anchor Boxes

    受FasterRCNN启发YOLOv2采用锚框，即在每个网格中预先定义一组不同大小的边框，用来预测对象是否在框中和微调边框位置。由于YOLOv2的锚框数量多所以最终去除全连接层，直接使用锚框进行预测。

- 维度聚类 Dimension Clusters

    YOLOv2对训练集中标注的边框进行K-means聚类分析，以寻找尽可能匹配样本的边框尺寸。

    如果我们用标准的欧式距离的k-means，尺寸大的框比小框产生更多的错误。因为我们的目的是提高IOU分数，这依赖于Box的大小，所以距离度量的使用：

    $$d(box,centroid)=1-IoU(box,centroid)$$

    其中，centroid是聚类时被选作中心的边框，box就是其它边框，d就是两者间的“距离”，IOU越大，“距离”越近。通过实验结果的分析，选择k-means的k=5。

- 直接预测位置

    Yolov2网络在每一个网格单元中预测出5个边框，每个边框有五个值$t_x,t_y,t_w,t_h,t_o$如果单元格与图像左上角的偏移为$(c_x, c_y)$，并且边界框先验的宽度和高度为 $p_w,p_h$，则预测的结果：

    $$b_x=\sigma(t_x)+c_x \\ b_y=\sigma(t_y)+c_y \\ b_w=p_we^{t_w} \\ b_h=p_he^{t_h} \\ P_r(object)*IoU(b,object)=\sigma(t_o)$$

    ![YOLOv2锚框](./img/yolov2_1.png)

- 细粒度特征 Fine-Grained Features

    最后一层感受野太大了，小目标可能丢失了，需融合之前的特征。YOLOv2通过passthrough的方法获得细粒度特征。具体来说，就是在最后一个pooling之前，将其1拆4，直接传递（passthrough）到pooling后（并且又经过一组卷积）的特征图，两者叠加到一起作为输出的特征图。

    ![YOLOv2细粒度特征](./img/yolov2_3.png)

- 多尺度训练

    由于YOLOv2不使用全连接层，输入可以是不同的尺寸。为了使YOLOv2对不同的输入尺寸具有鲁棒性，作者随机训练模型，每10批改变尺寸（从320x320到608x608）。

### 模型结构

![YOLOv2模型结构](./img/yolov2_2.png)

Darknet-19
1. 使用了很多3x3卷积核；并且每一次池化后，下一层的卷积核的通道数=池化输出的通道 x 2
2. 在每一层卷积后，都增加了BN层进行预处理
3. 采用了降维的思想，把1x1的卷积置于3x3之间，用来压缩特征
4. 在网络最后的输出上增加了一个global average pooling层

***

## YOLOv3

论文[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

### 模型结构

Darknet-53

![Darknet53](./img/yolov3_1.png)

![YOLOv3模型结构](./img/yolov3_2.jpg)

1. CBL: Conv+BN+LeakyReLU。
2. Res unit: 借鉴Resnet网络中的残差结构，让网络可以构建的更深。
3. ResX: 由一个CBL和X个残差组件构成，是Yolov3中的大组件。每个Res模块前面的CBL都起到下采样的作用。
4. Concat: 张量拼接，会扩充两个张量的维度。Concat和cfg文件中的route功能一样。
5. add:张量相加，张量直接相加，不会扩充维度。add和cfg文件中的shortcut功能一样。

Darknet53的主要改进：

- 没有采用最大池化层，转而采用步长为2的卷积层进行下采样
- 去掉了全连接层
- 为了防止过拟合，在每个卷积层之后加入了一个BN层和一个Leaky ReLU
- 引入了残差网络的思想，目的是为了让网络可以提取到更深层的特征，同时避免出现梯度消失或爆炸
- 将网络的中间层和后面某一层的上采样进行张量拼接，达到多尺度特征融合的目的
- 网格大小有三种

### 改进内容

- 多尺度预测

    为了能够预测多尺度的目标，YOLOv3 选择了三种不同shape的Anchors，同时每种Anchors具有三种不同的尺度，一共9种不同大小的Anchors。

    ![YOLOv3多尺度预测](./img/yolov3_2.png)

- 特征金字塔网络(feature parymid network,FPN)

    ![YOLOv3特征金字塔网络](./img/yolov3_3.png)

    通过特征金字塔进行分层，使不同大小的特征图可以更好地预测不同大小的框。最小的特征图的感受野最大，可以预测大目标，反之则是感受野小细节多，可以预测小目标。

    通过上采样的方法，将不同大小特征图进行融合，从而使小特征图中的全局信息可以更好地帮助大的特征图进行预测。

- 残差连接

    ![YOLOv3残差连接](./img/yolov3_4.png)

    残差连接的核心思想是引入一个"shortcut"或"skip connection"，允许输入信号直接绕过一些层，并与这些层的输出相加。这样，网络不再需要学习将输入映射到输出的完整函数，而是学习一个残差函数，即输入与期望输出之间的差异。

- 先验框设计

    通过聚类得到9种锚框。

    |特征图|13x13|26x26|52x52|
    |---|---|---|---|
    |感受野|大|中|小|
    |先验框|(116x90)(156x198)(373x326)|(30x61)(62x45)(59x119)|(10x13)(16x30)(33x23)|
    
    ![YOLOv3先验框](./img/yolov3_5.png)

- 多标签分类

    YOLOv3在类别预测方面将YOLOv2的单标签分类改进为多标签分类，在网络结构中将YOLOv2中用于分类的softmax层修改为逻辑分类器。即是用logistic激活函数。

### 损失函数

$$Loss=\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}I_{ij}^{obj}[(x_i^j-\hat{x}_i^j)^2+(y_i^j-\hat{y}_i^j)^2] \\
+\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}I_{ij}^{obj}[(x_i^j-\hat{x}_i^j)^2+(y_i^j-\hat{y}_i^j)^2] \\
-\sum_{i=0}^{S^2}\sum_{j=0}^{B}I_{ij}^{obj}[\hat{C}_i^jlog(C_j^j)+(1-\hat{C}_i^j)log(1-C_j^j)] \\
-\lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B}I_{ij}^{noobj}[\hat{C}_i^jlog(C_j^j)+(1-\hat{C}_i^j)log(1-C_j^j)] \\
-\sum_{i=0}^{S^2}I_{ij}^{noobj}\sum_{c\in classes}[\hat{P}_i^j(C)log(P_i^j(C))+(1-\hat{P}_i^j(C))log(P_i^j(C))]$$

相比YOLOv1中的损失函数，置信度损失和类别预测的改为了交叉熵损失。

***

## YOLOv4

论文[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

### 模型结构

![YOLOv4](./img/yolov4_1.jpg)

- CBM：Yolov4网络结构中的最小组件，由Conv+Bn+Mish激活函数三者组成。
- CBL：由Conv+Bn+Leaky_relu激活函数三者组成。
- Res unit：借鉴Resnet网络中的残差结构，让网络可以构建的更深。
- CSPX：借鉴CSPNet网络结构，由卷积层和X个Res unint模块Concate组成。
- SPP：采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。

YOLOv4=CSODarknet53(backbone)+SPP模块(neck)+PANet(neck)+YOLOv3head(head)

![YOLOv4结构](./img/yolov4_2.png)

### 改进部分

作者分为了BOF和BOS两部分。

Bag of freebies: 只增加训练成本，但是能显著提高精度，并不影响推理速度

Bag of speacials: 增加稍许推断代价，单可以提高模型精度的方法

#### Bag of freebies

- mosaic数据增强

    将四张不同的训练图像随机拼接在一起，形成一张马赛克图像进行训练。这种方式可以帮助模型学习并适应不同的场景、目标形状和尺度变化。

    ![YOLOv4Mosaic数据增强](./img/yolov4_4.png)

- 数据增强
  - Random Erase: 用随机值或训练集的平均像素值替换图像区域
  - Hide and Seek: 根据概率设置随机隐藏一些patch
  - 等等

    ![YOLOv4数据增强](./img/yolov4_3.png)

- 自对抗训练(Self-Adversarial Training,SAT)

    SAT是一种自对抗训练数据增强方法，这一种新的对抗性训练方式。在第一阶段，神经网络改变原始图像而不改变网络权值。以这种方式，神经网络对自身进行对抗性攻击，改变原始图像，以制造图像上没有所需对象的欺骗。在第二阶段，用正常的方法训练神经网络去检测目标。

- DropBlock

    之前dropout随机选择点，现在随机选择一个区域

    ![YOLOv4DropBlock](./img/yolov4_5.png)

- Label Smoothing

    将类别标签进行平衡如(cat,dog)=(1,0)->[1,0]x(1-0.1)+0.1/2=[0.05,0.95]

    使用之后可以使簇内更紧密，簇间更分离。

- 损失函数

    ![IoU](./img/iou1.png)

    ![GIoU](./img/iou2.png)

    ![DIoU](./img/iou3.png)

    ![CIoU](./img/iou4.png)

#### Bag of specials

- DIoU-NMS和Soft-NMS

    之前采用NMS，选择采用DIoU-NMS，考虑Box中心点距离。

    Soft-NMS则是更柔和的NMS，将之前会直接删除的框，变为修改分数即$s_i<-s_if(iou(M,b_i))$

    $$s_i=\begin{cases}
        s_i,IoU-R_{DIoU}(M,B_i)\lt\varepsilon \\
        0,IoU-R_{DIoU}(M,B_i)\ge\varepsilon
    \end{cases} R_{DIoU}=\frac{\rho^2(b,b^{gt})}{c^2}$$

    M表示高置信度候选框，$B_i$是遍历各个框跟置信度高的框的重合情况

- SPPNet(Spatial Pyramid Pooling)
  - 增大感受野
  - 使用最大池化满足最终输入特征一致

- CSPNet(Cross Stage Partial Network)

    CSPNet的主要目的是能够实现更丰富的梯度组合，同时减少计算量。

    每个block按照特征图的channel分为两部分，一部分走正常网络，另一份直接concat到这个block的输出

    ![CSPNet](./img/yolov4_6.png)

- SAM(Spatial Attention Module)

    作者在原SAM(Spatial Attention Module)方法上进行了修改，将SAM从空间注意修改为点注意。通过引入SAM模块，YOLOv4能够自适应地调整特征图的通道注意力权重。

    ![SAM](./img/yolov4_7.png)

- PAN(Path Aggregation Network)

    YOLOv4引入了PAN（Path Aggregation Network）模块，用于在不同尺度的特征图之间进行信息传递和融合，以获取更好的多尺度特征表示。

    ![PAN](./img/yolov4_9.png)

    作者对原PAN(Path Aggregation Network)方法进行了修改, 使用张量连接(concat)代替了原来的快捷连接(shortcut connection)。

    ![PAN2](./img/yolov4_8.png)

- Mish激活函数

    $$f(x)=x \cdot tanh(ln(1+e^x))$$

***

## 参考文章和推荐

[YOLO系列算法全家桶——YOLOv1-YOLOv9详细介绍 ！！](https://cloud.tencent.com/developer/article/2406045)

[跟着迪哥学AI](https://space.bilibili.com/3493077589166602)

[目标检测——Yolo系列（YOLOv1/2/v3/4/5/x/6/7/8）](https://blog.csdn.net/zyw2002/article/details/125443226)

[深入浅出Yolo系列之Yolov3&Yolov4&Yolov5&Yolox核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)