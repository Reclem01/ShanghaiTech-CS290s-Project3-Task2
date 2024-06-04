import torch
import torch.nn as nn

# 定义LSTM模型
# class StockPriceLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
#         super(StockPriceLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
#         return out


class StockPriceCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, seq_length):
        super(StockPriceCNNLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=(1, ))
        self.lstm = nn.LSTM(seq_length, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        # Apply 1D convolution
        # print(x.shape) # [25, 50, 30]
        x = self.conv1d(x)
        # print(x.shape) # [25, 1, 30]
        # x = x.squeeze(1)  # Remove the channel dimension
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out
