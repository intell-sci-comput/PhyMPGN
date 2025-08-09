import torch.nn as nn

from src.utils.utils import build_net


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = layers
        self.net = build_net(layers)

    def forward(self, inputs):
        return self.net(inputs)


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = layers
        self.net = build_net(layers)

    def forward(self, h):
        return self.net(h)
