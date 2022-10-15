import collections
import itertools
import time
import threading
import importlib
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['DGLBACKEND'] = 'pytorch'
from multiprocessing.pool import ThreadPool
import numpy as np
import queue

import dgl
from dgl import DGLGraph
from dgl.distributed import load_partition
from dgl.base import NID, EID

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.multiprocessing import Pool

from sklearn.metrics import f1_score

from runtime.schema.base import Node, Role, NTYPE, ETYPE, SCHEDULER_RANK, AGGREGATE_RANK, SCHEDULE_INERVAL, QUEUE_QUERY_INERVAL
from runtime.schema.block import Block
from runtime.schema.bindata import BinData
from runtime.schema.module import ModulesWithDependencies
from runtime.library.communication import CommunicationHandler
import runtime.utils as rtu


class Worker(object):
    """
    G3 worker, includes runtime
    """
    def __init__(self, rank, partid, clusterfile, in_feats, n_classes,
                 activation, criterion, args, multilabel):
        self._rank = rank
        self._partid = partid
        self.logger = rtu.G3_logging(args, 'G3 Worker ' + str(self._rank))
        clusterdict, partid2rank = rtu.InitClusterDict(clusterfile)
        self._runtime = WorkerRuntime(rank, partid, clusterdict, partid2rank,
                                      self.logger, in_feats, n_classes,
                                      activation, criterion, args, multilabel)

    # args: backend (pytorch), module, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type, criterion, local_rank, part_id


class WorkerRuntime(object):
    """
    G3 worker runtime, handle communication, and task forward / backward operations
    Note: worker contains whole NN model
    """
    def __init__(self, rank, partid, clusterdict, partid2rank, logger,
                 in_feats, n_classes, activation, criterion, args, multilabel):
        self._rank = rank
        self._partid = partid
        self._cluster_dict = clusterdict
        self._partid2rank = partid2rank
        self.logger = logger
        self._comm_handler = CommunicationHandler(self._rank, clusterdict,
                                                  logger)
        self.args = args
        self.n_classes = n_classes
        self.multilabel = multilabel

        if self.args.num_gpus == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' +
                                       str(self._rank % self.args.num_gpus))

        try:
            self.initialize(in_feats, activation, criterion, args)

            self.fp_init()
            self.bp_init()

        except Exception as e:
            self.logger.exception('Initialize error: {}'.format(e))

    def initialize(self, in_feats, activation, criterion, args):
        self.activation = activation
        self.criterion = criterion
        self.in_feats = in_feats

        # init torch.dist
        master_addr, _, _, master_port = self._cluster_dict[SCHEDULER_RANK]
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(args.backend,
                                rank=self._rank,
                                world_size=len(self._cluster_dict))
        # init NN model in each worker
        module = importlib.import_module(args.module)  # module: models.sage
        self._arch = module.arch()
        if self._arch == "graphsage":
            model = module.model(in_feats, args.n_hidden, self.n_classes,
                                 args.n_layers, activation, args.dropout,
                                 args.aggregator_type, criterion)
        elif self._arch == "gcn":
            model = module.model(in_feats, args.n_hidden, self.n_classes,
                                 args.n_layers, activation, args.dropout,
                                 criterion)
        elif self._arch == "gin":
            model = module.model(in_feats, args.n_hidden, self.n_classes,
                                 args.n_layers, args.num_mlp_layers,
                                 args.dropout, args.learn_eps,
                                 args.graph_pooling_type,
                                 args.neighbor_pooling_type)
        elif self._arch == "gat":
            self.heads = ([args.num_heads] *
                          args.n_layers) + [args.num_out_heads]
            model = module.model(args.n_layers, in_feats, args.n_hidden,
                                 self.n_classes, self.heads, activation,
                                 args.in_drop, args.attn_drop,
                                 args.negative_slope, args.residual, criterion)

        self.modules_with_dependencies = ModulesWithDependencies(model)
        modules = self.modules_with_dependencies.modules()

        # load init model
        if self.args.load_init_model:
            for i in range(len(modules)):
                path = self.args.log_dir + '/' + str(
                    self._rank) + '-' + self.args.dataset + '-module_' + str(
                        i) + '-epoch_' + str(self.args.n_epochs - 1) + '.pth'
                load_module = torch.load(path)
                load_module['layer.fc_self.weight'] = load_module[
                    'module.layer.fc_self.weight']
                load_module['layer.fc_self.bias'] = load_module[
                    'module.layer.fc_self.bias']
                load_module['layer.fc_neigh.weight'] = load_module[
                    'module.layer.fc_neigh.weight']
                load_module['layer.fc_neigh.bias'] = load_module[
                    'module.layer.fc_neigh.bias']

                del load_module['module.layer.fc_self.weight']
                del load_module['module.layer.fc_self.bias']
                del load_module['module.layer.fc_neigh.weight']
                del load_module['module.layer.fc_neigh.bias']
                modules[i].load_state_dict(load_module)

        self.cuda()

        dev_id = self._rank % self.args.num_gpus
        for i in range(len(modules)):
            modules[i] = torch.nn.parallel.DistributedDataParallel(
                modules[i], device_ids=[dev_id], output_device=dev_id)

        # Init optimizer per layer
        self.optimizer = []
        for i in range(len(modules)):
            optimizer = optim.Adam(modules[i].parameters(), self.args.lr)
            optimizer.zero_grad()
            self.optimizer.append(optimizer)

        # Init receive block thread
        self._featureQ = queue.Queue()
        self._gradientQ = queue.Queue()
        self._featurebuffer = []
        self._gradientbuffer = []

        if self._rank == AGGREGATE_RANK:
            self._notifyQ = queue.Queue()
            self._notify_blocks = {}  # {layerID(int): set(int)}
        else:
            self._stepQ = queue.Queue()
            self._step_blocks = {}  # {layerID(int): block(Block)}

        if self._rank == AGGREGATE_RANK:
            self._notify_endQ = queue.Queue()
            self._notify_end_blocks = []  # {layerID(int): set(int)}
        else:
            self._step_endQ = queue.Queue()

        self.merge_dict = dict()
        self.send_dict = dict()

        self.stop_receive_block_thread = False
        self._receive_block_thread = threading.Thread(
            target=self.receive_block_helper,
            args=(lambda: self.stop_receive_block_thread, ))

        self._receive_block_thread.start()

        # Load subgraph
        self.g, self.node_feats, self.edge_feats, self.gpb, result = rtu.process(
            self._partid, self.args, os.cpu_count())

        self.train_nid = torch.nonzero(
            self.node_feats['{}/train_mask'.format(NTYPE)], as_tuple=True)[0]
        self.val_nid = torch.nonzero(
            self.node_feats['{}/val_mask'.format(NTYPE)], as_tuple=True)[0]
        self.test_nid = torch.nonzero(
            ~(self.node_feats['{}/train_mask'.format(NTYPE)]
              | self.node_feats['{}/val_mask'.format(NTYPE)]),
            as_tuple=True)[0]

        _, self.Dnid, self.Inid, self.Hnid, self.local_nc, self.all_nc, _ = result

        # self.DInid is numpy.array
        self.DInid = np.sort(
            np.append(np.array(self.Dnid), np.array(self.Inid)))
        self.DIlnid = get_indices(self.g.ndata[NID].numpy(),
                                  np.array(self.DInid))

        self.npHnid = np.array(self.Hnid)
        self.Hlnid = get_indices(self.g.ndata[NID].numpy(), self.npHnid)

        # init input features
        self.init_feats = torch.zeros([self.all_nc, self.in_feats],
                                      dtype=torch.float32)
        self.init_feats[self.DIlnid] = self.node_feats['{}/features'.format(
            NTYPE)]

        # Store partid nid dict
        mergetensor = self.g.ndata['part_id']
        for partid in range(len(self._cluster_dict)):
            is_partid = torch.nonzero(mergetensor == partid, as_tuple=True)[0]
            if len(is_partid):
                self.merge_dict[partid] = self.g.ndata[NID][is_partid].numpy()
                if partid == self._partid:
                    pass
                else:
                    merge_block = Block(src_rank=self._rank,
                                        dst_rank=self._partid2rank[partid],
                                        partID=self._partid,
                                        merge_nids=self.merge_dict[partid])
                    self._comm_handler.sendblock(merge_block.dst_rank,
                                                 merge_block)
                    del merge_block
            else:
                # Guarantee all nodes have synced merge_dict
                null_merge_block = Block(src_rank=self._rank,
                                         dst_rank=self._partid2rank[partid],
                                         partID=self._partid,
                                         merge_nids=torch.tensor([]))
                self._comm_handler.sendblock(null_merge_block.dst_rank,
                                             null_merge_block)
                del null_merge_block

        # Barrier until receive all merge_blocks
        while True:
            if len(self.send_dict) < len(self._cluster_dict) - 1:
                time.sleep(1)
            else:
                # Remove null_merge_block
                for k in list(self.send_dict):
                    if len(self.send_dict[k]) == 0:
                        del self.send_dict[k]

                break

        # DEBUG
        self.logger.debug("Rank-{} merge_dict: {}".format(
            self._rank, self.merge_dict))
        self.logger.debug("Rank-{} send_dict: {}".format(
            self._rank, self.send_dict))

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].to(self.device)

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def optim_zero_grad(self):
        for optim in self.optimizer:
            optim.zero_grad()

    def module_train(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def module_eval(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def run(self, epoch):
        try:
            if self.args.is_train:
                train_acc, val_acc, train_time, val_time = self.train(epoch)
                tr_speed = self.local_nc / train_time
                val_speed = len(self.train_nid) / train_time
                print(
                    'Epoch {:05d} | Train Time {:.4f} | Train Acc {:.4f} | Train Tput (nodes/s) {:.4f} | Val Time {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Val Tput (nodes/s) {:.4f}'
                    .format(epoch, train_time,
                            train_acc.item(), tr_speed, val_time,
                            self.loss.item(), val_acc.item(), val_speed),
                    flush=True)
                if epoch == self.args.n_epochs - 1:
                    start = time.time()
                    acc = self.eval(epoch)
                    end = time.time()
                    speed = self.local_nc / (end - start)
                    print(
                        'Epoch {:05d} | Test Time {:.4f} | Test Loss {:.4f} | Test Acc {:.4f} | Test Tput (nodes/s) {:.4f}'
                        .format(epoch, end - start, self.loss.item(),
                                acc.item(), speed),
                        flush=True)
            else:
                start = time.time()
                with torch.no_grad():
                    acc = self.eval(epoch)
                end = time.time()
                speed = self.local_nc / (end - start)
                print(
                    'Epoch {:05d} | Time {:.4f} | Loss {:.4f} | Test Acc {:.4f} | Speed (samples/sec) {:.4f}'
                    .format(epoch, end - start, self.loss.item(), acc.item(),
                            speed),
                    flush=True)

                block = Block(src_rank=self._rank,
                              dst_rank=AGGREGATE_RANK,
                              epoch=epoch,
                              partID=self._partid,
                              end=True)
                self._comm_handler.sendblock(block.dst_rank, block)
                del block
                self.logger.debug("Barrier end block sent.")

        except Exception as e:
            self.logger.exception('Train / eval error: {}'.format(e))

    def refresh(self, epoch):
        """
        Refresh before starting epoch training
        """
        self.logger.debug(f"Refresh starts")

        try:
            self.fp_refresh()
            self.bp_refresh()
        except Exception as e:
            self.logger.exception('Refresh error: {}'.format(e))

        self.logger.debug(f"Refresh ends")

    def fp_refresh(self):
        self.bin_gen_counter.clear()  # FP bin_gen_counter at each layer
        self.bin_comp_counter.clear()  # FP bin_comp_counter at each layer
        self.bin_feats_dict.clear(
        )  # FP intermediate tensors. {layerID(int): list(aggregated bin_feats(torch.tensor that stored in GPU))}
        self.in_tensors.clear(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}
        self.in_lnids.clear(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}
        self.out_tensors.clear(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}
        self.out_lnids.clear(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}

        # for i in range(self.args.n_layers + 1):
        for i in range(self.args.n_layers):
            self.bin_comp_counter[i] = 0
            self.in_tensors[i] = list()
            self.in_lnids[i] = list()
            self.out_tensors[i] = list()
            self.out_lnids[i] = list()

        self.loss = 0.0

        self.last_feats = None

        # init bin-packing mechanism
        self.bin_thread_list.clear()  # temporarily store bin threads list
        self.binQ.queue.clear()
        self.agg_featsQ.queue.clear(
        )  # enqueue aggregated feats when finishes layer computation

    def bp_refresh(self):
        # self.zero_grad()
        self.optim_zero_grad()
        self.step_layerID = self.args.n_layers - 1

    def fp_init(self):
        self.bin_gen_counter_lock = threading.Lock()
        self.bin_gen_counter = list()  # FP bin_gen_counter at each layer
        self.bin_comp_counter = dict()  # FP bin_comp_counter at each layer
        self.bin_feats_dict = dict(
        )  # FP intermediate tensors. {layerID(int): list(aggregated bin_feats(torch.tensor that stored in GPU))}
        self.in_tensors = dict(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}
        self.in_lnids = dict(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}
        self.out_tensors = dict(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}
        self.out_lnids = dict(
        )  # FP intermediate tensors. {layerID(int): list(intermediate_tensor(torch.tensor that stored in GPU))}

        # for i in range(self.args.n_layers + 1):
        for i in range(self.args.n_layers):
            self.bin_comp_counter[i] = 0
            self.in_tensors[i] = list()
            self.in_lnids[i] = list()
            self.out_tensors[i] = list()
            self.out_lnids[i] = list()

        self.loss = 0.0

        self.last_feats = None

        # init bin-packing mechanism
        self.loader_sem = threading.BoundedSemaphore(self.args.co_dataloader)
        self.bin_thread_list = list()  # temporarily store bin threads list
        self.binQ = queue.Queue()
        self.agg_featsQ = queue.Queue(
        )  # enqueue aggregated feats when finishes layer computation

    def bp_init(self):
        self.optim_zero_grad()
        self.step_layerID = self.args.n_layers - 1

    def start(self, epoch):
        self.logger.debug("start function starts")
        # refresh parameters, settle garbage
        self.refresh(epoch)

        # Init step block helper thread in rank=0
        if self.args.is_train:
            if self._rank == AGGREGATE_RANK:
                self._notify_block_thread = threading.Thread(
                    target=self.notify_block_helper, args=())

                self._notify_block_thread.start()
            else:
                self._step_block_thread = threading.Thread(
                    target=self.step_block_helper, args=())

                self._step_block_thread.start()
        else:
            if self._rank == AGGREGATE_RANK:
                self._notify_end_thread = threading.Thread(
                    target=self.notify_end_helper, args=())

                self._notify_end_thread.start()
            else:
                self._step_end_thread = threading.Thread(
                    target=self.step_end_helper, args=())

                self._step_end_thread.start()

        try:
            self.run(epoch)
            # store model
            if self.args.store_model:
                if epoch == self.args.n_epochs - 1:
                    modules = self.modules_with_dependencies.modules()
                    for i in range(len(modules)):
                        path = self.args.log_dir + '/' + str(
                            self._rank
                        ) + '-' + self.args.dataset + '-module_' + str(
                            i) + '-epoch_' + str(epoch) + '.pth'
                        torch.save(modules[i].state_dict(), path)
        except Exception as e:
            self.logger.exception('Run error: {}'.format(e))

        # barrier until all layers finished updating
        if self.args.is_train:
            if self._rank == AGGREGATE_RANK:
                self._notify_block_thread.join()
            else:
                self._step_block_thread.join()
        else:
            if self._rank == AGGREGATE_RANK:
                self._notify_end_thread.join()
            else:
                self._step_end_thread.join()

        self.logger.debug("start function ends")

    def receive_block_helper(self, stop):
        self.logger.debug(f'start receive_block_helper')
        while True:
            block = self._comm_handler.recvblock()
            self.logger.debug(
                f"Receive_block_helper: ({block.epoch},{block.layerID},{block.partID},{block.is_train},{block.fw_bw},{block.step},{block.finish})"
            )
            if block.finish:
                self.logger.debug(
                    f'Finish block received. Exit receive_block_helper')
                break
            # Receive blocks from other workers
            # Update block dst status, store block in self._recv_blocks
            if block.step:
                self.logger.debug(
                    'Receive step requests from w{} to w{}. Step Layer-{}'.
                    format(block.src_rank, block.dst_rank, block.layerID))
                # step block
                if self._rank == AGGREGATE_RANK:
                    self._notifyQ.put(block)
                else:
                    if block.src_rank == AGGREGATE_RANK:
                        self._stepQ.put(block)
                    else:
                        self.logger.exception(
                            'Error step block received from worker-{}!'.format(
                                block.src_rank))
            elif block.end:
                self.logger.debug('Receive end requests')
                # step block
                if self._rank == AGGREGATE_RANK:
                    self._notify_endQ.put(block)
                else:
                    if block.src_rank == AGGREGATE_RANK:
                        self._step_endQ.put(block)
                    else:
                        self.logger.exception(
                            'Error end block received from worker-{}!'.format(
                                block.src_rank))

            elif block.merge_nids is not None:
                self.logger.debug("Receive merge nids from w{}".format(
                    block.src_rank))
                # merge_dicts update
                self.send_dict[block.partID] = block.merge_nids
            else:
                if block.fw_bw:
                    self._featureQ.put(block)
                else:
                    self._gradientQ.put(block)

            if stop():
                break

    def step_block_helper(self):
        """
        Loop quests if self.step_layerID step block has been received.
        If yes, optimizer.step()
        Loop breaks when all layers has been stepped
        """
        self.logger.debug(f'start step_block_helper')
        while True:
            if self.step_layerID in self._step_blocks.keys():
                # check to avoid step block out-of-order
                # received step block, step
                self.optimizer[self.step_layerID].step()
                self.step_layerID -= 1

                if self.step_layerID < 0:
                    self._step_blocks.clear()
                    self.logger.debug(f'Exit step_block_helper')
                    break
                else:
                    self.logger.debug("Loop in step_block_helper")
            else:
                block = self._stepQ.get()
                self._step_blocks[block.layerID] = block

                if self.step_layerID in self._step_blocks.keys():
                    # received step block, step
                    self.optimizer[self.step_layerID].step()
                    self.step_layerID -= 1

                    if self.step_layerID < 0:
                        self._step_blocks.clear()
                        self.logger.debug(f'Exit step_block_helper')
                        break
                    else:
                        self.logger.debug("Loop in step_block_helper")

    def notify_block_helper(self):
        """
        Loop quest if all workers has finished self.step_layerID.
        If yes, send step blocks to each worker, then step rank=0 itself.
        Loop breaks when all layers has been steped
        """
        self.logger.debug(f'start notify_block_helper')
        while True:
            if self.step_layerID in self._notify_blocks.keys() and len(
                    self._notify_blocks[self.step_layerID]) == len(
                        self._cluster_dict):
                # check to avoid step block out-of-order
                # all step blocks received, send step block, then step rank=0 itself
                for dst in self._notify_blocks[self.step_layerID]:
                    if dst == self._rank:
                        continue
                    block = Block(src_rank=self._rank,
                                  dst_rank=dst,
                                  layerID=self.step_layerID,
                                  step=True)
                    self._comm_handler.sendblock(block.dst_rank, block)
                    self.logger.debug(
                        f"Notify_block_helper sends: (layer:{block.layerID},rank:{block.dst_rank},step:{block.step})"
                    )
                    del block

                self.optimizer[self.step_layerID].step()
                self.step_layerID -= 1

                if self.step_layerID < 0:
                    self._notify_blocks.clear()
                    self.logger.debug(f'Exit notify_block_helper')
                    break
                else:
                    self.logger.debug("Loop in notify_block_helper")
            else:
                block = self._notifyQ.get()
                key = block.layerID
                self.logger.debug(
                    'Receive step requests from w{} to w{}. Step Layer-{}'.
                    format(block.src_rank, block.dst_rank, block.layerID))
                if key in self._notify_blocks.keys():
                    self._notify_blocks[key].add(block.src_rank)
                else:
                    self._notify_blocks[key] = set()
                    self._notify_blocks[key].add(block.src_rank)

                # Send step block, excluding self._rank
                if key == self.step_layerID:
                    if len(self._notify_blocks[self.step_layerID]) == len(
                            self._cluster_dict):
                        # all step blocks received, send step block, then step rank=0 itself
                        for dst in self._notify_blocks[self.step_layerID]:
                            if dst == self._rank:
                                continue
                            block = Block(src_rank=self._rank,
                                          dst_rank=dst,
                                          layerID=self.step_layerID,
                                          step=True)
                            self._comm_handler.sendblock(block.dst_rank, block)
                            self.logger.debug(
                                f"Notify_block_helper sends: (layer:{block.layerID},rank:{block.dst_rank},step:{block.step})"
                            )
                            del block

                        self.optimizer[self.step_layerID].step()
                        self.step_layerID -= 1

                        if self.step_layerID < 0:
                            self._notify_blocks.clear()
                            self.logger.debug(f'Exit notify_block_helper')
                            break
                        else:
                            self.logger.debug("Loop in notify_block_helper")
                    else:
                        self.logger.debug("Current len:{}, keys:{}".format(
                            len(self._notify_blocks[self.step_layerID]),
                            self._notify_blocks[self.step_layerID]))

    def step_end_helper(self):
        self.logger.debug(f'start step_end_helper')
        while True:
            block = self._step_endQ.get()
            break

    def notify_end_helper(self):
        self.logger.debug(f'start notify_end_helper')
        while True:
            if len(self._notify_end_blocks) == len(self._cluster_dict):
                # all step blocks received, send step block, then step rank=0 itself
                for dst in self._notify_end_blocks:
                    if dst == self._rank:
                        continue
                    block = Block(src_rank=self._rank, dst_rank=dst, end=True)
                    self._comm_handler.sendblock(block.dst_rank, block)
                    self.logger.debug(
                        f"Notify_end_helper sends: (rank:{block.dst_rank},end:{block.end})"
                    )
                    del block

                self._notify_end_blocks.clear()
                self.logger.debug(f'Exit notify_end_helper')
                break
            else:
                block = self._notify_endQ.get()
                self.logger.debug(
                    'Receive end requests from w{} to w{}.'.format(
                        block.src_rank, block.dst_rank))
                self._notify_end_blocks.append(block.src_rank)

                if len(self._notify_end_blocks) == len(self._cluster_dict):
                    # all step blocks received, send step block, then step rank=0 itself
                    for dst in self._notify_end_blocks:
                        if dst == self._rank:
                            continue
                        block = Block(src_rank=self._rank,
                                      dst_rank=dst,
                                      end=True)
                        self._comm_handler.sendblock(block.dst_rank, block)
                        self.logger.debug(
                            f"Notify_end_helper sends: (rank:{block.dst_rank},end:{block.end})"
                        )
                        del block

                    self._notify_end_blocks.clear()
                    self.logger.debug(f'Exit notify_end_helper')
                    break

    def train(self, epoch):
        """
        model.train()
        run_forward / run_backward
        """
        self.module_train()
        self.logger.debug("Train starts")

        tr_start = time.time()
        try:
            final_feats = self.run_forward(epoch, self.train_nid, True)
        except Exception as e:
            self.logger.exception('Train run_forward error: {}'.format(e))

        try:
            self.run_backward(epoch, True)
        except Exception as e:
            self.logger.exception('Train run_backward error: {}'.format(e))
        tr_end = time.time()

        # Calculate train results
        final_feats = final_feats[self.train_nid].to(self.device)
        labels = self.node_feats['{}/labels'.format(NTYPE)][self.train_nid].to(
            self.device)
        tr_acc = compute_acc(final_feats, labels, self.multilabel)

        del final_feats, labels

        self.fp_refresh()

        self.logger.debug("Start val...")
        self.module_eval()
        val_start = time.time()
        try:
            final_feats = self.run_forward(epoch, self.val_nid, False)
        except Exception as e:
            self.logger.exception('Eval run_forward error: {}'.format(e))
        val_end = time.time()

        # Calculate evaluation results
        final_feats = final_feats[self.val_nid].to(self.device)
        labels = self.node_feats['{}/labels'.format(NTYPE)][self.val_nid].to(
            self.device)
        val_acc = compute_acc(final_feats, labels, self.multilabel)

        del final_feats, labels
        self.logger.debug("Train ends")
        return tr_acc, val_acc, tr_end - tr_start, val_end - val_start

    def eval(self, epoch):
        self.module_eval()
        self.fp_refresh()
        self.logger.debug("Evaluation starts")

        try:
            final_feats = self.run_forward(epoch, self.test_nid, False)
        except Exception as e:
            self.logger.exception('run_forward error: {}'.format(e))

        # Calculate evaluation results
        final_feats = final_feats[self.test_nid].to(self.device)
        labels = self.node_feats['{}/labels'.format(NTYPE)][self.test_nid].to(
            self.device)

        del final_feats, labels
        self.logger.debug("Evaluation ends")
        return compute_acc(final_feats, labels, self.multilabel)

    def run_forward(self, epoch, mask_tensor, is_train):
        self.logger.debug(f'run_forward starts')

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        modules = self.modules_with_dependencies.modules()

        self._bin_gen_thread = threading.Thread(target=self.bin_generator,
                                                args=(
                                                    epoch,
                                                    sampler,
                                                    is_train,
                                                ))
        self._bin_gen_thread.start()

        # loop FP computation
        while True:
            bin = self.binQ.get()
            if bin.id == -1:
                break
            self.logger.debug('Processing bin-{} at layer-{}'.format(
                bin.id, bin.layer))

            layer = bin.layer

            # create new output tensor
            if self._arch == "gat":
                # Because local nodes are the first N nodes in the graph
                new_feats = torch.zeros([
                    self.local_nc, self.args.n_hidden * self.heads[layer]
                    if layer < self.args.n_layers - 1 else self.n_classes
                ],
                                        dtype=torch.float32)
            else:
                new_feats = torch.zeros([
                    self.local_nc, self.args.n_hidden
                    if layer < self.args.n_layers - 1 else self.n_classes
                ],
                                        dtype=torch.float32)

            # process FP computation
            bin_block = bin.bin_block[0].int().to(self.device)
            bin_in = bin.node_feats[bin.bin_inlnids].to(self.device)

            with torch.autograd.graph.save_on_cpu():
                bin_out = modules[layer](bin_block, bin_in, layer,
                                         self.args.n_layers)

            bin_inc = bin_in.to(torch.device("cpu"))
            bin.bin_inclnids = bin.bin_inlnids.to(torch.device("cpu"))
            self.in_tensors[layer].append(bin_inc)
            self.in_lnids[layer].append(bin.bin_inclnids)

            bin_outc = bin_out.to(torch.device("cpu"))
            bin.bin_outclnids = bin.bin_outlnids.to(torch.device("cpu"))

            self.out_tensors[layer].append(bin_outc)
            self.out_lnids[layer].append(bin.bin_outclnids)

            del bin_block, bin_in, bin_out

            new_feats[bin.bin_outclnids] = bin_outc.clone().detach()

            self.bin_comp_counter[layer] += 1

            # send features
            if layer < self.args.n_layers - 1:
                # if not last layer, send features
                send_feats_thread = threading.Thread(
                    target=self.send_features,
                    args=(
                        epoch,
                        new_feats[bin.bin_outlnids],
                        self.g.ndata[NID][bin.bin_outlnids].numpy(),
                        layer + 1,
                        is_train,
                    ))
                send_feats_thread.start()

            # if exists, aggregate new_feats
            if layer in self.bin_feats_dict.keys():
                self.bin_feats_dict[layer][bin.bin_outlnids] = new_feats[
                    bin.bin_outlnids]
            else:
                # else append new_feats
                self.bin_feats_dict[layer] = new_feats

            self.bin_gen_counter_lock.acquire()
            if self.bin_comp_counter[layer] == self.bin_gen_counter[
                    layer] and layer < self.args.n_layers - 1:
                self.agg_featsQ.put(self.bin_feats_dict[layer])
            self.bin_gen_counter_lock.release()

            self.logger.debug('Finish processing bin-{}'.format(bin.id))
            del bin
            torch.cuda.empty_cache()

        # guarantee that bin_generator thread successfully ends
        self._bin_gen_thread.join()

        self.last_feats = self.bin_feats_dict[self.args.n_layers - 1].to(
            self.device)

        if is_train:
            self.last_feats.requires_grad = True

        if self.multilabel:
            self.loss = self.criterion(
                self.last_feats[mask_tensor],
                self.node_feats['{}/labels'.format(NTYPE)][mask_tensor].to(
                    self.device),
                reduction='sum')
        else:
            self.loss = self.criterion(
                self.last_feats[mask_tensor],
                self.node_feats['{}/labels'.format(NTYPE)][mask_tensor].to(
                    self.device, dtype=torch.long))

        if not is_train:
            self.last_feats = self.last_feats.to("cpu")

        self.logger.debug(f"run_forward ends")
        return self.bin_feats_dict[self.args.n_layers - 1]

    def run_backward(self, epoch, is_train):
        self.logger.debug(f'run_backward starts')

        inputs = {}  # {name(str): input_tensor(torch.tensor)}
        outputs = {}  # {name(str): Doutput_tensor(torch.tensor)}
        input_gradients = {
        }  # {name(str): gradients(torch.tensor)}, to register_hook
        last_gradients = {}  # {name(str): gradients(torch.tensor)}
        output_gradients = {}  # {name(str): gradients(torch.tensor)}

        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        # init bin gradients
        for layer in range(self.args.n_layers - 1, -1, -1):
            # set input_gradients and output_gradients
            input_name = all_input_names[layer][0]
            output_name = all_output_names[layer][0]

            inputs[input_name] = list()
            input_gradients[input_name] = list()
            last_gradients[output_name] = list()

        # hook to record input gradients
        def hook_wrapper(grad_dict, input_name):
            def hook(input_gradient):
                grad_dict[input_name].append(input_gradient)

            return hook

        # perform output layer BP
        output_name = all_output_names[-1][0]
        if self.last_feats.requires_grad:
            self.last_feats.register_hook(
                hook_wrapper(last_gradients, output_name))
        try:
            torch.autograd.backward(self.loss, grad_tensors=None)
            output_gradients[output_name] = last_gradients[output_name][-1]

            del self.last_feats, self.loss
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.exception('Error loss backward: {}'.format(e))

        for layer in range(self.args.n_layers - 1, -1, -1):
            self.logger.debug('Backwarding layer {}'.format(layer))
            # set input_gradients and output_gradients
            input_name = all_input_names[layer][0]
            output_name = all_output_names[layer][0]

            if layer < self.args.n_layers - 1 and layer >= 0:
                self._send_gradients_thread.join()
                self._merge_gradients_thread.join()
                output_gradients[output_name] = recv_grads

                output_gradients[output_name] += cur_grads[self.DIlnid]
                output_gradients[output_name] = output_gradients[
                    output_name].to(self.device)

            if layer - 1 >= 0:
                if self._arch == "gat":
                    cur_grads = torch.zeros(
                        [self.all_nc, self.args.n_hidden * self.heads[layer]],
                        dtype=torch.float32)
                    recv_grads = torch.zeros([
                        self.local_nc,
                        self.args.n_hidden * self.heads[layer - 1]
                    ],
                                             dtype=torch.float32)
                else:
                    cur_grads = torch.zeros([self.all_nc, self.args.n_hidden],
                                            dtype=torch.float32)
                    recv_grads = torch.zeros(
                        [self.local_nc, self.args.n_hidden],
                        dtype=torch.float32)

                self._merge_gradients_thread = threading.Thread(
                    target=self.merge_gradients,
                    args=(
                        epoch,
                        recv_grads,
                        layer,
                        is_train,
                    ))
                self._merge_gradients_thread.start()

            # Loop bin BP
            for binidx in range(len(self.in_tensors[layer])):
                bin_out = self.out_tensors[layer][binidx]
                bin_out = bin_out.to(self.device)

                out_lnids = self.out_lnids[layer][binidx]

                # perform backward pass, only one input/output tensor each layer
                try:
                    torch.autograd.backward(
                        bin_out,
                        grad_tensors=output_gradients[output_name][out_lnids])
                except Exception as e:
                    self.logger.exception(
                        'Error backward for binidx-{} at layer-{}. {}'.format(
                            binidx, layer, e))

                # # if not first layer, merge bin input gradients
                if layer - 1 >= 0:
                    pass

                # del bin_in, bin_out
                del bin_out
                torch.cuda.empty_cache()

            del output_gradients[output_name]
            torch.cuda.empty_cache()

            # send aggregated gradients
            if layer - 1 >= 0:
                self._send_gradients_thread = threading.Thread(
                    target=self.send_gradients,
                    args=(
                        epoch,
                        cur_grads,
                        layer,
                        is_train,
                    ))
                self._send_gradients_thread.start()

            # send STEP block to rank=AGGREGATE_RANK
            block = Block(src_rank=self._rank,
                          dst_rank=AGGREGATE_RANK,
                          epoch=epoch,
                          layerID=layer,
                          partID=self._partid,
                          step=True)
            self._comm_handler.sendblock(block.dst_rank, block)
            self.logger.debug(
                "Send step block from w{} to w{} step layer-{}".format(
                    block.src_rank, block.dst_rank, block.layerID))

            del block

            self.logger.debug('Finish backward layer {}'.format(layer))

        self.logger.debug(f'run_backward ends')

    def bin_dataloader(self, ID, layer, sampler, nodes, binset, node_feats,
                       edge_feats):
        self.loader_sem.acquire()
        self.logger.debug("Loading bin-{}".format(ID))

        b = BinData(self.logger, ID, layer, sampler, self.g, self.DIlnid,
                    binset, node_feats, edge_feats)
        self.binQ.put(b)

        self.logger.debug("Finish loading bin-{}".format(b.id))
        self.loader_sem.release()

    def g3_bin_packing(self, epoch, layer, method="prio", eager=True):
        def select(a, size=-1):
            if size == -1:
                yield a
                return
            l = 0
            while l < len(a):
                selected = a[l:l + size]
                l += size
                yield selected

        bin_size = self.args.bin_size
        nid, lnid = self.DInid, self.DIlnid
        N = len(lnid)

        if epoch == 0 and layer == 0:
            self.Dlnid = self.Dnid - nid[0]
            self.Ilnid = self.Inid - nid[0]

        # this part is equivalent to the original (naive) bin packing implemented
        if method == "naive":
            for x in select(lnid, bin_size):
                yield x
            return

        if layer != 0:
            bin_size *= int(self.in_feats / self.args.n_hidden)

        diter = select(self.Dlnid)
        iiter = select(self.Ilnid, bin_size)

        if eager:
            while self._merge_features_thread[layer].is_alive():
                try:
                    # Because bin dataloading is non-blocking, this eager
                    # compuation algo may send MANY I-nodes bins to eager mode,
                    # delaying the computation on ready D-nodes, essentially breaking
                    # rule of node priority, which slows down the whole process.
                    # We can restrict eager computation to ONE bin at a time
                    # by join the I-node bin's thread.
                    # Current method works well with cora, though.
                    yield next(iiter)
                except:
                    break
        # interlayer pp: node priority
        for x in diter:
            yield x
        # intralayer pp: adaptive bin packing
        # method not performing well on small graph (cora), removed for now
        for x in iiter:
            yield x

    def bin_generator(self, epoch, sampler, is_train):
        self.logger.debug("bin_generator starts")

        # init features, features_bak (next layer's merged input)
        features = self.init_feats.clone()
        # intermediate features have the same shape
        zero_feats = torch.zeros([self.all_nc, self.args.n_hidden],
                                 dtype=torch.float32)

        self._merge_features_thread = [None] * self.args.n_layers
        # send 1st layer's features and merge
        self.send_features(epoch, features, self.DInid, 0, is_train)
        self._merge_features_thread[0] = threading.Thread(
            target=self.merge_features, args=(
                epoch,
                features,
                0,
                is_train,
            ))
        self._merge_features_thread[0].start()

        binid = 0

        eager_enabled = self.args.eager
        bin_method = self.args.bin_method

        for layer in range(self.args.n_layers):
            self.logger.debug("Generating layer-{} bin".format(layer))

            if not eager_enabled:
                self._merge_features_thread[layer].join()

            # if not layer-0, assign received values from bak to features, output values from previous layer to features
            if layer != 0:
                # block until all previous layer's features has been computed
                new_feats = self.agg_featsQ.get()
                features = features_bak
                features[self.DIlnid] = new_feats

            if layer < self.args.n_layers - 1:
                # if not last layer, start merge thread
                if self._arch == "gat":
                    features_bak = torch.zeros(
                        [self.all_nc, self.args.n_hidden * self.heads[layer]],
                        dtype=torch.float32)
                else:
                    features_bak = zero_feats.clone()

                self._merge_features_thread[layer + 1] = threading.Thread(
                    target=self.merge_features,
                    args=(
                        epoch,
                        features_bak,
                        layer + 1,
                        is_train,
                    ))
                self._merge_features_thread[layer + 1].start()

            self.bin_gen_counter_lock.acquire()
            self.bin_gen_counter.append(0)
            self.bin_gen_counter_lock.release()

            # select bin
            # method: naive or prio
            for binset in self.g3_bin_packing(epoch,
                                              layer,
                                              method=bin_method,
                                              eager=eager_enabled):
                # generate FP bin
                t = threading.Thread(target=self.bin_dataloader,
                                     args=(
                                         binid,
                                         layer,
                                         sampler,
                                         self.DIlnid,
                                         self.DIlnid[binset],
                                         features,
                                         None,
                                     ))
                if layer == self.args.n_layers - 1:  # if last layer, get all threads for join
                    self.bin_thread_list.append(t)
                t.start()
                binid += 1
                self.bin_gen_counter_lock.acquire()
                self.bin_gen_counter[-1] += 1
                self.bin_gen_counter_lock.release()

            self.logger.debug("Finish generating layer-{} bin".format(layer))

        # join all last layer's threads to guarantee fin_bin is the last bin
        for t in self.bin_thread_list:
            t.join()

        # insert fin_bin that indicates ending FP epoch
        fin_bin = BinData()
        self.binQ.put(fin_bin)
        self.logger.debug("bin_generator ends")

    def merge_features(self, epoch, feats, layer, is_train):
        """
        merge features in self.block.node_feats & self.block.edge_feats
        in_place change feats
        feats.size = [self.all_nc, self.embedding_in_each_layer]
        """
        self.logger.debug("Merge_features starts.")
        allhnidset = set(self.Hnid)

        # Obtain, merge, and clear buffer
        for block in reversed(self._featurebuffer):
            if layer == block.layerID and epoch == block.epoch and is_train == block.is_train:
                self.logger.debug(
                    "Merge out-of-order features from worker-{}".format(
                        block.partID))
                localhnids = get_indices(self.g.ndata[NID].numpy(), block.nids)
                feats[localhnids] = block.node_feats
                allhnidset.difference_update(set(block.nids.tolist()))

                self._featurebuffer.remove(block)
                del block
            else:
                self.logger.debug(
                    "Worker-{}: layer: {}-{}, epoch: {}-{}, is_train: {}-{}".
                    format(block.partID, block.layerID, layer, block.epoch,
                           epoch, block.is_train, is_train))

        if len(allhnidset) > 0:
            # Process merge when receiving features
            while True:
                block = self._featureQ.get()
                if layer == block.layerID and epoch == block.epoch and is_train == block.is_train:
                    localhnids = get_indices(self.g.ndata[NID].numpy(),
                                             block.nids)
                    feats[localhnids] = block.node_feats
                    allhnidset.difference_update(set(block.nids.tolist()))

                    del block

                    if len(allhnidset) == 0:
                        break
                    else:
                        pass
                else:
                    self._featurebuffer.append(block)
                    self.logger.debug(
                        "Unordered feature block received for merging. Block layer {} from worker-{} received in Layer {}, is_train {}-{}. Buffer.len: {}"
                        .format(block.layerID, block.partID, layer,
                                block.is_train, is_train,
                                len(self._featurebuffer)))

        self.logger.debug("Merge_features ends.")

    def merge_gradients(self, epoch, grads, layer, is_train):
        """
        If not receive all required nodes, sleep and re-check
        merge gradients in self.block.gradients
        in-place change grads
        """
        self.logger.debug("Merge_gradients starts.")
        # Check if all required partition received
        merge_ids = list(self.send_dict.keys())
        if self._partid in merge_ids:
            merge_ids.remove(self._partid)

        # Obtain, merge, and clear buffer
        for block in reversed(self._gradientbuffer):
            if layer == block.layerID and epoch == block.epoch and is_train == block.is_train:
                merge_ids.remove(block.partID)
                localnids = get_indices(self.g.ndata[NID].numpy(), block.nids)
                # in-place add
                grads[localnids] += block.gradients

                self._gradientbuffer.remove(block)
                del block

        if len(merge_ids) > 0:
            while True:
                block = self._gradientQ.get()
                if layer == block.layerID and epoch == block.epoch and is_train == block.is_train:
                    merge_ids.remove(block.partID)
                    localnids = get_indices(self.g.ndata[NID].numpy(),
                                            block.nids)
                    grads[localnids] += block.gradients

                    del block

                    if len(merge_ids) == 0:
                        break
                    else:
                        self.logger.debug(
                            "Not received gradient hosts: {}".format(
                                merge_ids))
                else:
                    self._gradientbuffer.append(block)
                    self.logger.debug(
                        "Unordered gradient block received for merging. Block layer {} received in Layer {}. Buffer.len: {}"
                        .format(block.layerID, layer,
                                len(self._gradientbuffer)))

        self.logger.debug("Merge_gradients ends.")

    def send_features(self, epoch, feats, nids, layer, is_train):
        """
        feats.size = [self.all_nc, self.embedding_in_each_layer]
        # localnids - index in all_nc
        nids
        """
        self.logger.debug("Send_features starts.")
        fw_bw = True
        try:
            for key, value in self.send_dict.items():
                if key == self._partid:
                    continue
                host_nids = np.intersect1d(value, nids)
                lnids_host_idx = get_indices(nids, host_nids)

                if len(lnids_host_idx):
                    block = Block(src_rank=self._rank,
                                  dst_rank=self._partid2rank[key],
                                  epoch=epoch,
                                  layerID=layer,
                                  partID=self._partid,
                                  is_train=is_train,
                                  fw_bw=fw_bw,
                                  nids=nids[lnids_host_idx],
                                  node_feats=feats[lnids_host_idx])
                    self._comm_handler.sendblock(block.dst_rank, block)
                    self.logger.debug(
                        f"Send features block: ({block.epoch},{block.layerID},{block.dst_rank},{block.is_train})"
                    )
                    del block

        except Exception as e:
            self.logger.exception("Error in send_features: {}".format(e))

        self.logger.debug("Send_features ends.")

    def send_gradients(self, epoch, grads, layer, is_train):
        self.logger.debug("Send_gradients starts.")
        fw_bw = False
        for key, value in self.merge_dict.items():
            if key == self._partid:
                continue
            localnids = get_indices(self.g.ndata[NID].numpy(), value)
            block = Block(src_rank=self._rank,
                          dst_rank=self._partid2rank[key],
                          epoch=epoch,
                          layerID=layer,
                          partID=self._partid,
                          is_train=is_train,
                          fw_bw=fw_bw,
                          nids=value,
                          gradients=grads[localnids])
            self._comm_handler.sendblock(block.dst_rank, block)
            self.logger.debug(
                f"Send gradients block: ({block.epoch},{block.layerID},{block.dst_rank},{block.is_train})"
            )
            del block

        self.logger.debug("Send_gradients ends.")

    def stop(self):
        """
        Stop epoch training
        """
        fin_block = Block(finish=True)
        self._comm_handler.sendblock(self._rank, fin_block)
        del fin_block
        self._receive_block_thread.join()

        self.loss = 0.0
        self.zero_grad()
        self.optim_zero_grad()
        self.step_layerID = self.args.n_layers - 1

        self.last_feats = None

        self._featurebuffer.clear()
        self._gradientbuffer.clear()
        self.bin_thread_list = list()  # temporarily store bin threads list

        # close comm_handler
        self._cluster_dict.clear()
        self._comm_handler.stop()

    def store_tensor(self, layer, epoch, tensor, name):
        path = self.args.log_dir + '/' + str(
            self._rank
        ) + '-' + self.args.dataset + '-layer_' + str(layer) + '-bin_' + str(
            self.args.bin_size) + '-epoch_' + str(epoch) + '-' + name + '.pth'
        torch.save(tensor, path)


def compute_acc(pred, labels, multilabel):
    """
    Compute the accuracy of prediction given the labels.
    """
    if not multilabel:
        labels = labels.long()
        return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        pred[pred > 0] = 1
        pred[pred <= 0] = 0
        return f1_score(labels.detach().cpu(),
                        pred.detach().cpu(),
                        average="micro")


def get_indices(x, y):
    """
    x is NID, y is subset of x
    """
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices
