from torch import nn

class ImgRecNN(nn.Module):
    def __init__(self):
        super(ImgRecNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), # 2d卷积3通道->32通道，卷积核5，步长1，填充2
            nn.MaxPool2d(2), #最大池化，池化核2
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(), # 展平成张量
            nn.Linear(64*4*4, 64), # 64个隐藏单元
            nn.Linear(64, 10) # 输出
        )

    def forward(self, x):
        x = self.model(x)
        return x