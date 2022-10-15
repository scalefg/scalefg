"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import sys
import argparse
import time
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['DGLBACKEND'] = 'pytorch'
import random
import numpy as np
import faulthandler

from runtime.worker import Worker

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

gdataset = {
    "simple": (1, 2),
    "karate": (1, 2),
    "cora": (1433, 7),
    "reddit": (602, 41),
    "citeseer": (3703, 6),
    "pubmed": (500, 3),
    "ogb-product": (100, 47),
    "ogbn-products": (100, 47),
    "ogbn-papers100M": (128, 172),
    "ogb-paper100M": (128, 172),
    "amazon": (200, 107),
    "twitter-2010": (128, 10)
}
multilabel_data = set(['amazon'])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_seeds(args):
    if args.seed == 0:
        return
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    # Can be fast when NN structure and input are fixed
    torch.backends.cudnn.benchmark = True


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def run_worker(args):
    # Initiate activation, criterion
    activation = F.relu
    multilabel = args.dataset in multilabel_data
    if multilabel:
        criterion = F.binary_cross_entropy_with_logits
    else:
        criterion = F.cross_entropy
    for key in gdataset.keys():
        if key in args.dataset:
            in_feats, n_classes = gdataset[key]
            break

    if args.in_feats != -1:
        in_feats = args.in_feats

    # Initiate worker
    worker = Worker(args.rank, args.partid, args.clusterfile, in_feats,
                    n_classes, activation, criterion, args, multilabel)
    runtime = worker._runtime

    print(f'Worker {args.rank} starts...', flush=True)
    for i in range(args.n_epochs):
        print(f'Running epoch {i}')
        runtime.start(i)

    runtime.stop()
    print(f'Worker {args.rank} finishes...', flush=True)


def main(args):
    init_seeds(args)
    run_worker(args)


if __name__ == '__main__':
    faulthandler.enable()

    parser = argparse.ArgumentParser(description='G3 GraphSAGE')
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="PyTorch distributed rank")
    parser.add_argument("--partid", type=int, help="DGL partID")
    parser.add_argument("--clusterfile",
                        type=str,
                        default="nodes.txt",
                        help="Distributed cluster settings")
    parser.add_argument("--logging-level",
                        type=str,
                        default="debug",
                        help="G3 logging level")
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        help="PyTorch backend")
    parser.add_argument('--load-init-model',
                        action='store_true',
                        help='load init model')
    parser.add_argument('--store-model',
                        action='store_true',
                        help='store model in the last epoch')
    parser.add_argument("--module",
                        type=str,
                        default="runtime.models.sage",
                        help="Model importlib")
    parser.add_argument("--n-epochs",
                        type=int,
                        default=10,
                        help="Total number of epochs")
    parser.add_argument("--n-hidden",
                        type=int,
                        default=16,
                        help="Dataset n_hidden feature size")
    parser.add_argument("--n-layers",
                        type=int,
                        default=2,
                        help="GNN num_layer")

    # GIN parameters
    parser.add_argument("--num-mlp-layers",
                        type=int,
                        default=2,
                        help="GIN num_mlp_layer")
    parser.add_argument("--learn_eps",
                        action="store_true",
                        help="learn the epsilon weighting")
    parser.add_argument('--graph_pooling_type',
                        type=str,
                        default="sum",
                        choices=["sum", "mean", "max"],
                        help='type of graph pooling: sum, mean or max')
    parser.add_argument('--neighbor_pooling_type',
                        type=str,
                        default="sum",
                        choices=["sum", "mean", "max"],
                        help='type of neighboring pooling: sum, mean or max')

    # GAT parameters
    parser.add_argument("--num-heads",
                        type=int,
                        default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads",
                        type=int,
                        default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual",
                        action="store_true",
                        default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop",
                        type=float,
                        default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop",
                        type=float,
                        default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope',
                        type=float,
                        default=0.2,
                        help="the negative slope of leaky relu")

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument("--aggregator-type",
                        type=str,
                        default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--local-rank",
                        type=int,
                        default=0,
                        help="Local rank of process")
    parser.add_argument("--num_gpus", type=int, default=-1, help="num of gpus")
    parser.add_argument("--dataset",
                        type=str,
                        default="cora",
                        help="Training / Inference dataset")
    parser.add_argument("--out-path",
                        type=str,
                        default="../data",
                        help="Output partition path")
    parser.add_argument("--is-train",
                        type=str2bool,
                        default=False,
                        help="Is train?")
    parser.add_argument('--bin_size', type=int, default=1000)
    parser.add_argument('--co-dataloader', type=int, default=1)
    parser.add_argument("--eager",
                        type=str2bool,
                        default=False,
                        help="Run eager?")
    parser.add_argument("--bin_method",
                        type=str,
                        default="prio",
                        help="bin method to use")
    parser.add_argument("--lr",
                        type=float,
                        default=0.003,
                        help="learning rate")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed value. By default 0 and not fixed")
    parser.add_argument("--log-dir",
                        type=str,
                        default='../log',
                        help="Log-dir to store weights/gradients")
    parser.add_argument("--in-feats", type=int, help="input feature size")
    args = parser.parse_args()
    print(args)

    main(args)
