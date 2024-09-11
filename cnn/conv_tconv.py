import torch
from torch import nn


# 先卷积再反卷积，卷积核参数相同
class ConvMod(nn.Module):
    def __init__(self,W) -> None:
        super().__init__()
        self.W = W

    def trans_conv(self, X, W):
        h, w = W.shape
        Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i: i + h, j: j + w] += X[i, j] * W
        return Y

    def conv(self, X, W):
        h, w = W.shape
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * W).sum()
        return Y

    def forward(self, X):
        y1 = self.conv(X, W)
        y2 = self.trans_conv(y1, W)
        return y2
    
W = torch.rand(2,2,requires_grad=True)
convMod = ConvMod(W)
lr = 3e-2
optimizer = torch.optim.Adam([W], lr=lr)
loss_fn = nn.L1Loss()

min_loss = torch.inf
min_W = torch.zeros(W.shape)

for i in range(100000):
    X = torch.rand(3,3)*100
    Y = convMod(X)
    loss = loss_fn(Y,X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if min_loss > loss.item():
        min_loss = loss.item()
        min_W = W
    
    if (i + 1) % 10 == 0:
        print(f'epoch {i+1}, loss {loss:.3f}')

print(f'min_loss {min_loss}\n min_W {W}')