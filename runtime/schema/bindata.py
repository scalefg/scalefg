import dgl
import torch
from dgl.dataloading import NodeCollator
from dgl.base import NID, EID


class BinData(object):
    """
    Basic bin data
    Stores bin_input, bin_output, bin_block
    """
    def __init__(self,
                 logger=None,
                 id=-1,
                 layer=-1,
                 sampler=None,
                 graph=None,
                 lnids=None,
                 binset=None,
                 node_feats=None,
                 edge_feats=None):
        self.id = id
        self.logger = logger
        self.layer = layer
        if self.id == -1:
            self.bin_inlnids, self.bin_outlnids, self.bin_block = None, None, None
            self.node_feats, self.edge_feats = None, None
        else:
            self.bin_block = sampler.sample_blocks(graph, binset)
            self.bin_inlnids = self.bin_block[0].srcdata[NID]
            self.bin_outlnids = self.bin_block[-1].dstdata[NID]
            self.node_feats = node_feats
            self.edge_feats = edge_feats
