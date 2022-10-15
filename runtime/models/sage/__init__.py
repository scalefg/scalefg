from .sagelayer import GraphSAGELayer
from .sage import GraphSAGE


def arch():
    return "graphsage"


def model(in_feats, n_hidden, n_classes, n_layers, activation, dropout,
          aggregator_type, criterion):
    model = []
    model.append((GraphSAGELayer(in_feats, n_hidden, activation, dropout,
                                 aggregator_type), ["input0"], ["output0"]))
    i = 1
    while i < n_layers - 1:
        out0 = "output" + str(i)
        out1 = "output" + str(i + 1)
        model.append((GraphSAGELayer(n_hidden, n_hidden, activation, dropout,
                                     aggregator_type), [out0], [out1]))
        i += 1
    out0 = "output" + str(i)
    out1 = "output" + str(i + 1)
    model.append((GraphSAGELayer(n_hidden, n_classes, activation, dropout,
                                 aggregator_type), [out0], [out1]))

    return model


def full_model(in_feats, n_hidden, n_classes, n_layers, activation, dropout,
               aggregator_type):
    return GraphSAGE(in_feats, n_hidden, n_classes, n_layers, activation,
                     dropout, aggregator_type)
