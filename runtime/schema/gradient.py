import torch
import pickle


class Gradient(object):
    """
    G3 gradient definition
    """
    def __init__(self,
                 src_rank=-1,
                 dst_rank=-1,
                 layerID=-1,
                 subgraphID=-1,
                 tensor_name=None,
                 gradient=None):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.layerID = layerID
        self.subgraphID = subgraphID
        self.tensor_name = tensor_name
        self.gradient = gradient

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(self, b):
        self = pickle.loads(b)
        return self
