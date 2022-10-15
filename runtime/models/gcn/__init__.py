from .gcnlayer import GCNLayer
from .gcn import GCN


def arch():
    return "gcn"


def model(in_feats, n_hidden, n_classes, n_layers, activation, dropout,
          criterion):
    model = []
    model.append((GCNLayer(in_feats, n_hidden, activation,
                           dropout), ["input0"], ["output0"]))
    i = 1
    while i < n_layers - 1:
        out0 = "output" + str(i)
        out1 = "output" + str(i + 1)
        model.append((GCNLayer(n_hidden, n_hidden, activation,
                               dropout), [out0], [out1]))
        i += 1
    out0 = "output" + str(i)
    out1 = "output" + str(i + 1)
    model.append((GCNLayer(n_hidden, n_classes, activation,
                           dropout), [out0], [out1]))

    return model


def full_model(in_feats, n_hidden, n_classes, n_layers, activation, dropout):
    return GCN(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
