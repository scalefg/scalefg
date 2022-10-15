import sys

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout):
        super(GCNLayer, self).__init__()
        self.layer = GraphConv(in_feats,
                               out_feats,
                               activation=activation,
                               allow_zero_in_degree=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, inputs, layer_id, n_layers):
        h = inputs
        if layer_id == 0:
            h = self.layer(graph, h)
        else:
            h = self.dropout(h)
            h = self.layer(graph, h)

        return h
