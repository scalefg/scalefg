import pickle


class Signal(object):
    """
    G3 Signal Definition
    """
    def __init__(self, src_rank=-1, dst_rank=-1, layerID=-1, subgraphID=-1):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.layerID = layerID
        self.subgraphID = subgraphID

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(self, b):
        self = pickle.loads(b)
        return self
