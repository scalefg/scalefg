import torch
from .gatlayer import GATLayer
import torch.nn as nn


class GAT(nn.Module):
    def __init__(self, n_layers, in_feats, n_hidden, n_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.activation = activation

        # input projection (no residual)
        layerid = 0
        self.layers.append(
            GATLayer(layerid, in_feats, n_hidden, heads, feat_drop, attn_drop,
                     negative_slope, False, self.activation).layer)

        layerid += 1
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GATLayer(i, n_hidden * heads[i - 1], n_hidden, heads[i],
                         feat_drop, attn_drop, negative_slope, residual,
                         self.activation).layer)
            layerid += 1

        # output projection
        self.layers.append(
            GATLayer(layerid, n_hidden * heads[-2], n_classes, heads[-1],
                     feat_drop, attn_drop, negative_slope, residual,
                     None).layer)  # activation None

    def forward(self, graph, inputs):
        h = inputs
        for l, layer in enumerate(self.layers):
            if l < len(self.layers) - 1:
                h = layer(graph, h, l, self.n_layers).flatten(1)
            else:
                # last layer
                h = layer(graph, h, l, self.n_layers).mean(1)

        return h
