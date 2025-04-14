import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0,
        num_layers=1,
        output_dim=1,
        need_sigmoid=True,
        need_encoding=True,
        **kws
    ):
        super(LSTM_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.need_sigmoid = need_sigmoid
        self.need_encoding = need_encoding

        # 定义LSTM层
        if self.need_encoding: # 如果True则先过一层mlp降维
            self.lstm = nn.Sequential(
                nn.Linear(input_dim,hidden_dim),
                nn.ReLU(inplace=True),
                nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                        num_layers=num_layers, batch_first=True),)
            
        else:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                                num_layers=num_layers, batch_first=True)
        
        # 定义Batch Normalization层，注意BatchNorm1d期望的维度是NCL
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # 将LSTM层添加到模块中
        self.add_module('lstm', self.lstm)
        self.add_module('batch_norm', self.batch_norm)

        # 定义后续标准层
        net = nn.Sequential()
        if self.dropout is not None:
            net.add_module("dropout", nn.Dropout(self.dropout))
        net.add_module('relu',nn.ReLU(inplace=True))
        net.add_module('fc', nn.Linear(hidden_dim, output_dim))

        if self.need_sigmoid:
            net.add_module("sigmoid", nn.Sigmoid())
        self.net = net


    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # 通过LSTM层
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # 通过Batch Normalization层，注意需要转置维度
        lstm_out = self.batch_norm(lstm_out.transpose(1, 2))
        lstm_out = lstm_out.transpose(1, 2)  # 转置回(batch_size, seq_len, hidden_dim)
        
        # 取序列的最后一个时间步
        last_time_step = lstm_out[:, -1, :]

        # 通过后续标准层
        out = self.net(last_time_step)
        return out
