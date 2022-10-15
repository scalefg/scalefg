import torch as th
import torch.distributed as th_dist
import time
import sys
import os

from dgl import DGLHeteroGraph
from .tcpvan import TCPVan

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from runtime.schema.task import Task
from runtime.schema.base import QUEUE_QUERY_INERVAL
from runtime.schema.block import Block


class CommunicationHandler(object):
    """
    G3 communication handler, includes TCP and torch
    """
    def __init__(self, rank, clusterdict, logger):
        self._rank = rank
        self._clusterdict = clusterdict
        self.logger = logger
        self.initialize()

    def initialize(self):
        self._tcp = TCPVan(self._rank, self._clusterdict, self.logger)
        self._tcp.start()

        self.tensorDict = {
        }  # Received tensors stored in dict, {tag(int): tensor(torch.Tensor)}

    def stop(self):
        self._tcp.stop()

    def sendtask(self, dst_rank: int, task: Task):
        self._tcp.sendtask(dst_rank, task)

    def sendblock(self, dst_rank: int, block: Block):
        self._tcp.sendblock(dst_rank, block)

    def sendtensor(self, dst_rank: int, tensor: th.tensor, tag: int):
        """
        tag: taskID, match remote recv
        """
        th_dist.isend(tensor, dst_rank, tag=tag)

    def recvtensor(self, src_rank: int, tensor: th.tensor, tag: int):
        """
        tag: taskID, match remote send
        """
        th_dist.irecv(tensor, src_rank, tag=tag)

    def recvtask(self):
        """
        Blocking get recv_tasks. If get, then return task
        """
        task = None
        while True:
            try:
                task = self._tcp.taskQ.get_nowait()
            except Exception:
                pass
            if task == None:
                time.sleep(QUEUE_QUERY_INERVAL)
                continue
            else:
                break

        return task

    def non_blocking_recvtask(self):
        """
        Non-blocking get recv_tasks
        """
        task = None
        try:
            task = self._tcp.taskQ.get_nowait()
        except Exception:
            pass

        return task

    def recvblock(self):
        """
        Blocking get recv_blocks. If get, then return block
        """
        block = self._tcp.blockQ.get(block=True)
        return block

    def non_blocking_recvblock(self):
        """
        Non-blocking get recv_blocks
        """
        block = None
        try:
            block = self._tcp.blockQ.get_nowait()
        except Exception:
            pass

        return block