import torch
from torch import nn

from lstm_model import LSTM

# 初始化参数
input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1
model = LSTM(input_size, hidden_size, num_layers, output_size)

# 设置训练参数
learning_rate = 0.01
num_epochs = 100

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

