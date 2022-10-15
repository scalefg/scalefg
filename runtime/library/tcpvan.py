import socket
import logging
import threading
import os
import time
import sys
import queue
import struct

from .threadedtcp import ThreadedTCPServer, ThreadedTCPRequestHandler

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from runtime.schema.base import Node, Role
from runtime.schema.task import Task
# from runtime.schema.signal import Signal
from runtime.schema.block import Block
from runtime.utils import GetIP, dglgraph_serialize, dglgraph_deserialize

RECV_BUFFER = 1024


class TCPVan(object):
    """
    Handle TCP Communication between nodes.
    """
    def __init__(self, rank, clusterdict, logger):
        """ G3 Logging"""
        self.logger = logger
        """ Self-condition """
        self.__start_lock = threading.Lock()
        self._my_node_ = Node(rank=rank)
        """ Cluster information """
        self._is_scheduler_ = 0
        self._cluster_dict = clusterdict  # {rank:int, socket:Node}
        """ Receiving Data """
        self._receiver_ = None
        self._receiver_thread = None
        self.taskQ = queue.Queue()
        self.blockQ = queue.Queue()

    def start(self):
        self.__start_lock.acquire()

        # get my node info
        itf = os.environ[
            "G3_INTERFACE"]  # workers need to specify their interfaces
        ip = GetIP(itf)

        # check if rank exists
        if self._my_node_.rank not in self._cluster_dict:
            print("Error rank")
            sys.exit(-1)
        self._my_node_.ip = self._cluster_dict[self._my_node_.rank][0]
        self._my_node_.port = self._cluster_dict[self._my_node_.rank][1]

        if ip != self._my_node_.ip:
            print(ip, self._my_node_.ip)
            sys.exit(-2)

        self._is_scheduler_ = 1 if self._my_node_.rank == 0 else 0

        # start listen
        self._receiver_ = TCPVanServer(
            (self._my_node_.ip, self._my_node_.port),
            TCPVanHandler,
            taskQ=self.taskQ,
            blockQ=self.blockQ,
            logger=self.logger)
        self._receiver_thread = threading.Thread(
            target=self._receiver_.serve_forever)
        self._receiver_thread.daemon = True
        self._receiver_thread.start()

        self.__start_lock.release()

    def stop(self):
        # clear state
        self._receiver_.server_close()
        self._receiver_.shutdown()
        self._receiver_thread = None

        self.__start_lock = None
        self.__init_stage = 0
        self._my_node_ = None
        self._is_scheduler_ = -1
        self._cluster_dict.clear()

    def sendtask(self, dst_rank, task):
        if dst_rank < 0:
            self.logger.error("Error destination node: %d" % dst_rank)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(
            (self._cluster_dict[dst_rank][0], self._cluster_dict[dst_rank][1]))
        msg = b'0' + task.serialize()
        msg = struct.pack('>Q', len(msg)) + msg
        sock.sendall(msg)

        sock.close()

    def sendblock(self, dst_rank, block):
        if dst_rank < 0:
            self.logger.error("Error destination node %d" % dst_rank)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self._cluster_dict[dst_rank][0],
                          self._cluster_dict[dst_rank][1]))
        except Exception as e:
            self.logger.error("Error connect with {}: {}".format(dst_rank, e))
            sock.close()
            return -1

        msg = b'1' + block.serialize()
        msg = struct.pack('>Q', len(msg)) + msg

        self.logger.debug("msg.len={}".format(len(msg)))

        try:
            sock.sendall(msg)
        except Exception as e:
            self.logger.error(
                f"Error send block ({block.epoch},{block.layerID},{block.partID},{block.is_train},{block.fw_bw},{block.step},{block.end},{block.finish}) to {dst_rank}: {e}"
            )

        self.logger.debug(
            f"tcpvan send block: ({block.epoch},{block.layerID},{block.partID},{block.is_train},{block.fw_bw},{block.step},{block.end},{block.finish}) to {dst_rank}"
        )
        sock.close()
        return 0


class TCPVanHandler(ThreadedTCPRequestHandler):
    """
    Multi-threaded handle receiving messages, and put into taskQ and blockQ
    """
    def __init__(self, request, client_address, server):
        self.taskQ = server.taskQ
        self.blockQ = server.blockQ
        self.logger = server.logger
        super().__init__(request, client_address, server)

    def handle(self):
        raw_msglen = self.request.recv(8)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>Q', raw_msglen)[0]
        return self.recv_msg(msglen)

    def recv_msg(self, msglen):
        data = bytearray()
        while len(data) < msglen:
            packet = self.request.recv(msglen - len(data))
            if not packet:
                return None
            data.extend(packet)

        if data[0] == 48:
            # task
            newtask = Task().deserialize(data[1:])
            self.taskQ.put(newtask)
        elif data[0] == 49:
            # block
            try:
                newblock = Block().deserialize(data[1:])
            except Exception as e:
                self.logger.error("Can not deserialize block: {}".format(e))
            self.logger.debug(
                f"tcpvan receive: ({newblock.epoch},{newblock.layerID},{newblock.partID},{newblock.is_train},{newblock.fw_bw},{newblock.step},{newblock.finish})"
            )
            self.blockQ.put(newblock)
        else:
            self.logger.error("Unknown message received")


class TCPVanServer(ThreadedTCPServer):
    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 bind_and_activate=True,
                 taskQ=None,
                 blockQ=None,
                 logger=None):
        self.taskQ = taskQ
        self.blockQ = blockQ
        self.logger = logger
        ThreadedTCPServer.allow_reuse_address = True
        ThreadedTCPServer.request_queue_size = 3000
        super().__init__(server_address, RequestHandlerClass,
                         bind_and_activate)
