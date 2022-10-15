"""Modules for task definition"""
from enum import Enum, unique
import pickle
from runtime.schema.block import Block


@unique
class State(Enum):
    """
    Task status:
    PENDING_ASSIGN = "Pending Assignment"
    INGRESS = "Ingress"
    PENDING_EXEC = "Pending Execution"
    EXECUTING = "Executing"
    FINISH = "Finished"
    PRUNE = "Pruned"
    UPDATE = "Update"
    END_OF_EPOCH = "End of Epoch"
    END_OF_JOB = "End of Job"
    """
    PENDING_ASSIGN = 0
    INGRESS = 1
    PENDING_EXEC = 2
    EXECUTING = 3
    FINISH = 4
    PRUNE = 5
    UPDATE = 6  # Note: To update, each layer's gradients and tensors should be stored separately from store_blocks to prevent deleted by PRUNE
    END_OF_EPOCH = 7
    END_OF_JOB = 8


class Task(object):
    """
    G3 Task Definition
    """
    def __init__(self,
                 taskID=-1,
                 nodeID=-1,
                 layerID=-1,
                 subgraphID=-1,
                 fw_bw=True,
                 is_train=1,
                 layer_wsize=1,
                 required_blocks={}):
        self.taskID = taskID
        self.nodeID = nodeID
        self.layerID = layerID
        self.subgraphID = subgraphID
        self.fw_bw = fw_bw  # fw: True, bw: False
        self.is_train = is_train
        self.layer_wsize = layer_wsize

        self.required_blocks = required_blocks  #{(layerID(int), subgraphID(int), fw_bw(bool)): block(Block)}

        self.state = State.PENDING_ASSIGN

    def __lt__(self, other):
        return self.taskID < other.taskID

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(self, b):
        self = pickle.loads(b)
        return self

    def showall(self, logger):
        logger.debug(f'TaskID: {self.taskID} \
                        NodeID: {self.nodeID} \
                        LayerID: {self.layerID} \
                        SubgraphID: {self.subgraphID} \
                        FWBW: {self.fw_bw} \
                        Is_train: {self.is_train} \
                        Layer_wsize: {self.layer_wsize} \
                        State: {self.state} \
                        Required_blocks: ')
        for key, value in self.required_blocks.items():
            logger.debug(f'{key}')
