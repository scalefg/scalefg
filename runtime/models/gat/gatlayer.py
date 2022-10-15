import sys

import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv


class GATLayer(nn.Module):
    def __init__(self, layerid, in_feats, out_feats, heads, feat_drop,
                 attn_drop, negative_slope, residual, activation):
        super(GATLayer, self).__init__()
        self.activation = activation

        # input layer
        if layerid == 0:
            self.layer = GATConv(in_feats, out_feats, heads[layerid],
                                 feat_drop, attn_drop, negative_slope, False,
                                 self.activation)
        # middle layer
        elif layerid < len(heads) - 1:
            self.layer = GATConv(in_feats, out_feats, heads[layerid],
                                 feat_drop, attn_drop, negative_slope,
                                 residual, self.activation)
        # output layer
        else:
            self.layer = GATConv(in_feats, out_feats, heads[layerid],
                                 feat_drop, attn_drop, negative_slope,
                                 residual, None)

    def forward(self, graph, inputs, layer_id, n_layers):
        if layer_id == n_layers - 1:
            # last layer
            h = self.layer(graph, inputs).mean(1)
        else:
            h = self.layer(graph, inputs).flatten(1)

        return h