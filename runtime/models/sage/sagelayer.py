import sys

import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type):
        super(GraphSAGELayer, self).__init__()
        self.layer = SAGEConv(in_feats, out_feats, aggregator_type)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, inputs, layer_id, n_layers):
        if layer_id == n_layers - 1:
            h = self.layer(graph, inputs)
        elif layer_id == 0:
            h = self.dropout(inputs)
            h = self.layer(graph, h)
            h = self.activation(h)
            h = self.dropout(h)
        else:
            h = self.layer(graph, inputs)
            h = self.activation(h)
            h = self.dropout(h)

        return h