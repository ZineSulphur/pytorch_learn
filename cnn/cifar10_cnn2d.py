# 利用CIFAR10数据集进行训练
import torch
import torchvision

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from cifar10_model import ImgRecNN

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 通过DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
imgRecNN = ImgRecNN()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(imgRecNN.parameters(), lr=learning_rate)

# 训练相关信息
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# tensorboard
writer = SummaryWriter("../logs")


for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 训练
    imgRecNN.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = imgRecNN(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if(total_train_step % 100 == 0):
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    imgRecNN.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = imgRecNN(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

            total_test_step += 1
            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    print("整体测试loss：{}".format(total_test_loss))
    print("整体测试准确率：{}".format(total_accuracy/test_data_size))

writer.close()
