import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        units=[128, 64],
        dropout=0,
        need_sigmoid=True,
        is_bn_first=True,
        **kws
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.need_sigmoid = need_sigmoid
        net = nn.Sequential()
        if is_bn_first:
            net.add_module("bn0", nn.BatchNorm1d(input_size))
            net.add_module("linear0", nn.Linear(input_size, units[0]))
        else:
            net.add_module("linear0", nn.Linear(input_size, units[0]))
            net.add_module("bn0", nn.BatchNorm1d(units[0]))
        net.add_module("relu0", nn.ReLU())

        for i in range(1, len(units)):
            net.add_module("linear" + str(i), nn.Linear(units[i - 1], units[i]))
            net.add_module("bn" + str(i), nn.BatchNorm1d(units[i]))
            net.add_module("relu" + str(i), nn.ReLU())

        if dropout is not None:
            net.add_module("dropout", nn.Dropout(dropout))

        net.add_module(
            "linear" + str(len(units) + 1), nn.Linear(units[len(units) - 1], 1)
        )
        if self.need_sigmoid:
            net.add_module("sigmoid", nn.Sigmoid())


        self.net = net

    def forward(self, x):
        x = self.net(x)
        if self.need_sigmoid:
            x = x - 0.5
        return x