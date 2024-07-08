# YOLO

You Only Look Once

我只看一次就可以识别出来了，要的就是又快又准。

## 总体介绍

- YOLO系列的核心思想就是把目标检测转变为一个回归问题，利用整张图片作为网络的输入，通过神经网络，得到边界框的位置及其所属的类别。
- 经典 one-stage 方法

## YOLOv1

论文[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

### 处理流程

1. 输入图片resize为448x448x3
2. 将图片分割成SxS的网格，一般为7x7网格
3. 每个网格设置两个候选框，预设候选框大小(h,w)按照经验值设置，框的位置(x,y)默认为中心点
4. 得到真实值和候选框的IoU，并且选择更合适的框
5. 对候选框进行微调，即预测候选框的(x,y,h,w)和置信度c，以及物体的分类

![yolov1核心思想](./img/yolov1_2.png)

### 网络结构

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

## 参考文章和推荐

[YOLO系列算法全家桶——YOLOv1-YOLOv9详细介绍 ！！](https://cloud.tencent.com/developer/article/2406045)

[跟着迪哥学AI](https://space.bilibili.com/3493077589166602)

[目标检测——Yolo系列（YOLOv1/2/v3/4/5/x/6/7/8）](https://blog.csdn.net/zyw2002/article/details/125443226)