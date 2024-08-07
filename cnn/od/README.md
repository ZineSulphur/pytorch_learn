# Object Detection 目标检测

目标检测（Object Detection）是计算机视觉领域中的一项核心技术，它旨在让计算机能够像人眼一样识别和定位图像中的物体。具体来说，它不仅需要识别出图像中有哪些对象，还要确定它们在图像中的位置（通常以边界框的形式表示）以及它们的类别。

目标检测的本质是对视觉信息进行解析，使计算机能够理解图像中的内容。这涉及到图像处理、特征提取、模式识别和机器学习等多个层面。在技术层面，目标检测要解决的问题包括但不限于：对象分类、位置估计、尺寸变化、遮挡处理、背景干扰以及实时处理能力等。

## 主要解决方式

- 两阶段检测方法：这种方法先从图像中提取出潜在的对象候选区域，然后对这些区域进行详细的分类和边界框精调。代表模型包括R-CNN系列（如Fast R-CNN, Faster R-CNN）和基于区域的全卷积网络（如Mask R-CNN）。
- 单阶段检测方法：这种方法直接在图像上预测对象的类别和位置，速度通常更快，但在准确度上可能略逊于两阶段方法。代表模型包括YOLO（You Only Look Once）、SSD（Single Shot MultiBox Detector）和RetinaNet等。
- Transformer架构也被引入到目标检测中，如DETR（Detection Transformer）模型

### 两阶段目标检测

两阶段目标检测器首先生成一系列候选的对象区域，然后对这些区域进行分类和边界框回归以确定最终的检测结果。这个过程通常分为两个主要阶段：

1. 区域提议（Region Proposal）：这单阶段生成可能包含目标的区域。例如，Faster RCNN使用区域提议网络（RPN）来高效生成提议。
2. 精确检测（Refinement）：提议的区域随后被送入网络进行进一步的分类和边界框的精调。Fast RCNN在此阶段应用了一种共享特征的方法来提高效率。

主要代表模型：
- RCNN：利用选择性搜索生成区域提议，然后将这些区域的特征通过CNN提取，并使用SVM分类。
- Fast RCNN：改进了RCNN的流程，通过共享卷积特征图来提高速度，并且在同一个网络中同时训练分类器和边界框回归器。
- Faster RCNN：引入RPN来端到端地生成区域提议，显著提高了生成提议的速度和效率。
- FPN（Feature Pyramid Networks）：通过一个自顶向下的结构和横向连接，为不同尺度的对象构建了高层次的语义特征，以改善多尺度目标的检测性能。

### 单阶段目标检测

单阶段目标检测方法的目标是简化检测流程，它们将目标的分类和边界框回归合并到一个步骤中完成。它们通常更快，但可能在精确度上略低于两阶段方法。

主要代表模型：
- YOLO（You Only Look Once）：将图像分割为网格，每个网格负责预测对象的边界框和分类概率，实现了非常快的检测速度。
- SSD（Single Shot MultiBox Detector）：在不同的特征图层级上同时检测不同尺度的对象，通过多尺度特征图和默认框来提高对小物体的检测能力。
- RetinaNet：解决了单阶段检测器中极端类别不平衡的问题，通过引入焦点损失（Focal Loss）来专注于困难样本，平衡了检测速度与准确性。

### 其它类型目标检测
- CornerNet：抛弃了使用锚框的传统方法，通过检测物体边界框的角点作为关键点来预测目标。
- CenterNet：进一步简化CornerNet的概念，只使用物体中心作为关键点，并回归所有相关属性。
- DETR（Detection Transformer）：利用Transformer结构将目标检测视为一种集合预测问题，摒弃了锚框的设计，通过全局特征来改进检测性能。

## 结构

- rcnn R-CNN学习内容
  - [rcnn.md](./rcnn/rcnn.md) R-CNN学习笔记
- yolo YOLO学习内容
  - [yolo.md](./yolo/yolo.md) YOLO学习笔记
- detr DETR学习内容
  - [detr.md](../../transformer/detr/detr.md) DETR学习笔记

## 参考文章

[目标检测简介](https://blog.csdn.net/qq_31463571/article/details/134692319)

[44 物体检测算法：R-CNN，SSD，YOLO【动手学深度学习v2】](https://www.bilibili.com/video/BV1Db4y1C71g)