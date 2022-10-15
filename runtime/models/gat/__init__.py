from .gatlayer import GATLayer
from .gat import GAT


def arch():
    return "gat"


def model(n_layers, in_feats, n_hidden, n_classes, heads, activation,
          feat_drop, attn_drop, negative_slope, residual, criterion):
    model = []
    i = 0
    model.append(
        (GATLayer(i, in_feats, n_hidden, heads, feat_drop, attn_drop,
                  negative_slope, False, activation), ["input0"], ["output0"]))
    i = 1
    while i < n_layers - 1:
        out0 = "output" + str(i)
        out1 = "output" + str(i + 1)
        model.append((GATLayer(i, n_hidden * heads[i - 1], n_hidden, heads,
                               feat_drop, attn_drop, negative_slope, residual,
                               activation), [out0], [out1]))
        i += 1
    out0 = "output" + str(i)
    out1 = "output" + str(i + 1)
    model.append(
        (GATLayer(i, n_hidden * heads[-2], n_classes, heads, feat_drop,
                  attn_drop, negative_slope, residual, None), [out0], [out1]))

    # # criterion
    # model.append((criterion, [out1], ["loss"]))

    return model


def full_model(n_layers, in_feats, n_hidden, n_classes, heads, activation,
               feat_drop, attn_drop, negative_slope, residual):
    return GAT(n_layers, in_feats, n_hidden, n_classes, heads, activation,
               feat_drop, attn_drop, negative_slope, residual)
