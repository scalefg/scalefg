"""Module for base types and utilities."""
from enum import Enum

Role = Enum('Role', ('rNotAssigned', 'rScheduler', 'rWorker'))

NTYPE = '_N'
ETYPE = '_E'

GID = '_G3'

SCHEDULER_RANK = 0
AGGREGATE_RANK = 1
SCHEDULE_INERVAL = 1 / 1000.0
QUEUE_QUERY_INERVAL = 1 / 1000.0


class Node(object):
    """
    Information of node
    """
    def __init__(self, ip="", port=-1, rank=-1, role=Role.rNotAssigned):
        self.ip = ip
        self.port = port

        self.rank = rank
        self.role = role