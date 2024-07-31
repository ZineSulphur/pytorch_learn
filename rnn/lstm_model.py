import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        init

        :param input_size: 每个时间步骤的输入特征数。
        :param hidden_size: 每个LSTM层中的隐藏单元数。
        :param num_layers: 堆叠的LSTM层数。
        :param output_size: 输出的大小（例如，预测值的数量）。
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #pytorch 实现的LSTM
        self.linear1 = nn.Linear(input_size,output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # out, (h0, c0) = self.lstm(x)
        # out = self.lstm(x, (h0, c0))[0]
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear1(out[:,-1,:])
        return out
