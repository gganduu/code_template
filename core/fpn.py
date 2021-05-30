import torch
from collections import OrderedDict

class FPN(torch.nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

    def forward(self, x):
        output = OrderedDict()
        for n, t in x.items():
            torch.nn.Conv2d()