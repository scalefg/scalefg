import torch
from .gcnlayer import GCNLayer
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers

        # input layer
        self.layers.append(
            GCNLayer(in_feats, n_hidden, activation, dropout).layer)
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GCNLayer(n_hidden, n_hidden, activation, dropout).layer)
        # output layer
        self.layers.append(
            GCNLayer(n_hidden, n_classes, activation,
                     dropout).layer)  # activation None

    def forward(self, graph, inputs):
        h = inputs
        for l, layer in enumerate(self.layers):
            if l != 0:
                h = self.dropout(h)
            h = layer(graph, h, l, self.n_layers)
        return h