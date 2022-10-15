import pickle

from dgl.data.heterograph_serialize import HeteroGraphData


class Block(object):
    """
    G3 Block Definition,
    which contains src_rank, dst_rank, layerID, subgraphID, and block(subgraph)
    """
    def __init__(
            self,
            src_rank=-1,
            dst_rank=-1,
            epoch=-1,
            layerID=-1,
            partID=-1,
            is_train=True,
            fw_bw=True,
            nids=None,  # np.array
            node_feats=None,
            edge_feats=None,
            gradients=None,
            merge_nids=None,
            step=False,
            end=False,
            finish=False):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.epoch = epoch
        self.layerID = layerID
        self.partID = partID
        self.is_train = is_train
        self.fw_bw = fw_bw
        self.nids = nids
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.gradients = gradients
        self.merge_nids = merge_nids
        self.step = step
        self.end = end
        self.finish = finish

    def show(self, logger):
        logger.debug(f'src_rank: {self.src_rank} \
                        dst_rank: {self.dst_rank} \
                        epoch: {self.epoch} \
                        layerID: {self.layerID} \
                        partID: {self.partID} \
                        is_train: {self.is_train} \
                        FWBW: {self.fw_bw} \
                        nids: {self.nids} \
                        node_feats: {self.node_feats} \
                        edge_feats: {self.edge_feats} \
                        gradients: {self.gradients} \
                        merge_nids: {self.merge_nids} \
                        step: {self.step} \
                        end: {self.end} \
                        finish: {self.finish}')

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(self, b):
        self = pickle.loads(b)
        return self
