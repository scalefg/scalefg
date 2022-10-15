import torch
from .sagelayer import GraphSAGELayer
import torch.nn as nn


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.activation = activation

        # input layer
        self.layers.append(
            GraphSAGELayer(in_feats, n_hidden, activation, dropout,
                           aggregator_type).layer)
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden, activation, dropout,
                               aggregator_type).layer)
        # output layer
        self.layers.append(
            GraphSAGELayer(n_hidden, n_classes, activation, dropout,
                           aggregator_type).layer)  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h, l, self.n_layers)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
