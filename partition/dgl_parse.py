#!/usr/bin/python3
import dgl
import numpy as np
import torch as th
import argparse
import time

from dgl_load import load_karate, load_cora, load_reddit, load_ogb, load_amzn, load_yelp, create_simple_graph, load_webgraph

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        '--dataset',
        type=str,
        default='reddit',
        help='datasets: cora, reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--method',
                           type=str,
                           default='g3',
                           help='partitioning method')
    argparser.add_argument('--num_parts',
                           type=int,
                           default=4,
                           help='number of partitions')
    argparser.add_argument('--undirected',
                           action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--selfloop',
                           action='store_true',
                           help='turn the graph into a self-looped graph.')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'simple':
        g = create_simple_graph(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'karate':
        g, _ = load_karate(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'cora':
        g, _ = load_cora(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'reddit':
        g, _ = load_reddit(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogb-paper100M':
        g, _ = load_ogb('ogbn-papers100M',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogb-proteins':
        g, _ = load_ogb('ogbn-proteins',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogb-mag':
        g, _ = load_ogb('ogbn-mag', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogb-arxiv':
        g, _ = load_ogb('ogbn-arxiv', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogbn-products':
        g, _ = load_ogb('ogbn-products',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogbn-papers100M':
        g, _ = load_ogb('ogbn-papers100M',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogbn-proteins':
        g, _ = load_ogb('ogbn-proteins',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogbn-mag':
        g, _ = load_ogb('ogbn-mag', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogbn-arxiv':
        g, _ = load_ogb('ogbn-arxiv', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'amazon':
        g, _ = load_amzn(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'yelp':
        g, _ = load_yelp(prob=args.prob, feat_size=args.feat_size)
    else:
        g = load_webgraph(args.dataset,
                          prob=args.prob,
                          feat_size=args.feat_size)

    print('load {} takes {:.3f} seconds'.format(args.dataset,
                                                time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(
        th.sum(g.ndata['train_mask']), th.sum(g.ndata['val_mask']),
        th.sum(g.ndata['test_mask'])))

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    if args.selfloop:
        print("adding self loops")
        g = dgl.add_self_loop(g)

    f = open("dataset/{}/{}.parts.{}.{}".format(args.dataset, args.dataset,
                                                args.num_parts, args.method))
    loaded_parts = [int(x) for x in f.readlines()]

    # the default DGL implementation forces a choice for partition method that generates `node_parts`
    # we alter the method here to make it take a `node_parts` directly from the input
    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        "dataset/%s/dgl" % args.dataset,
        part_method="g3",
        node_parts=loaded_parts,
    )
