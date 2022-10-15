import netifaces as ni
import pickle
import logging
import sys
from tqdm import tqdm
from pathlib import Path
import os

import dgl
from dgl.data.heterograph_serialize import HeteroGraphData
from torch.multiprocessing import Pool

CLUSTERFILE = 'nodes.txt'  # ip, sshport, tcpport, pytorchport, rank
# engine = 'dgl'


def GetIP(interface):
    ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
    return ip


def InitClusterDict(clusterfile=CLUSTERFILE):
    cluster_dict = {}
    partid2rank_dict = {}
    f = open(clusterfile, 'r')
    lines = f.readlines()

    for line in lines:
        ip, snd_port, recv_port, th_port, partid, rank = line.split()
        cluster_dict[int(rank)] = (ip, int(snd_port), int(recv_port), th_port)
        partid2rank_dict[int(partid)] = int(rank)

    return cluster_dict, partid2rank_dict


def load_subgraph(args, part_id):
    part_config_path = args.out_path + "/" + args.dataset + ".json"
    g, node_feats, edge_feats, gpb, _ = dgl.distributed.load_partition(
        part_config_path, part_id)
    return g, node_feats, edge_feats, gpb


def G3_logging(args, name):
    logger = logging.getLogger('[' + name + ']: ')
    fmt = logging.Formatter('%(name)s %(levelname)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if args.logging_level == 'info' or args.logging_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif args.logging_level == 'debug' or args.logging_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif args.logging_level == 'error' or args.logging_level == 'ERROR':
        logger.setLevel(logging.ERROR)
    else:
        logging.error('Unknown G3 logging level')
        sys.exit(1)

    return logger


def dglgraph_serialize(graph):
    """
    Serialize DGLHeteroGraph
    """
    gdata = HeteroGraphData.create(graph)
    return pickle.dumps(gdata)


def dglgraph_deserialize(gbytes):
    """
    Deserialize DGLHeteroGraph
    """
    gdata = pickle.loads(gbytes)
    g = HeteroGraphData.get_graph(gdata)  # TODO(wxc): test if this could work
    return g


def find_node_d(partid, u, v, nidmap, prange, wrange, wid):
    def isLocal(x):
        return prange[partid] <= x and x < prange[partid + 1]

    def findp(x):
        for i in range(1, len(prange)):
            if prange[i] > x:
                return i - 1

    dsets = [set() for i in range(len(prange) - 1)]
    for i in range(len(u))[wrange[0]:wrange[1]]:
        x = nidmap[u[i]]
        if isLocal(x):
            continue
        y = nidmap[v[i]]
        dsets[findp(x)].add(int(y))
    return dsets


def process(partid, args, workers):
    try:
        graph, node_feats, edge_feats, book, _, _, _ = dgl.distributed.load_partition(
            '{}/{}.json'.format(args.out_path, args.dataset), partid)
    except Exception as e:
        print('Error load_partition, exception:' + str(e))

    local_node_count = book.metadata()[partid]["num_nodes"]
    print("\npartition #{}: nodes {} (local {}), edges {}".format(
        partid, graph.num_nodes(), local_node_count, graph.num_edges()))

    try:
        print("loading from cache file...")
        cachef = open("%s/%s.%s.cache" % (args.out_path, args.dataset, partid),
                      "r")
        print("cache file found, loading...")
        cache = cachef.readlines()
        dlist = [int(x) for x in cache[0].strip("[]\n").split(", ")]
        ilist = [int(x) for x in cache[1].strip("[]\n").split(", ")]
        hlist = [int(x) for x in cache[2].strip("[]\n").split(", ")]
        pdcount = [int(x) for x in cache[3].strip("[]\n").split(", ")]
        print("preloading cache hit\n")
        print(
            "PartID:{}, Len(dset):{}, Len(iset):{}, Len(haloset):{} local:{}, Num_node:{}, pdcount:{}"
            .format(partid, len(dlist), len(ilist), len(hlist),
                    local_node_count, graph.num_nodes(), pdcount))
        return graph, node_feats, edge_feats, book, (partid, dlist, ilist,
                                                     hlist, local_node_count,
                                                     graph.num_nodes(),
                                                     pdcount)
    except Exception as e:
        print(e)
        print("preloading cache miss\n")
        cachef = open("%s/%s.%s.cache" % (args.out_path, args.dataset, partid),
                      "w")

    u, v = graph.edges(form="uv", order="srcdst")
    nidmap = graph.ndata["_ID"]
    partitions = len(book.metadata())
    prange = [0]
    for x in book.metadata():
        prange += [prange[-1] + x["num_nodes"]]

    dsets = [set() for x in range(partitions)]
    with Pool(processes=workers) as pool:
        workload = int(len(u) / workers) + 1
        splits = list(
            zip(range(0, len(u), workload),
                range(workload,
                      len(u) + workload, workload)))
        works = []
        for wid, wrange in enumerate(splits):
            works.append(
                pool.apply_async(find_node_d,
                                 (partid, u, v, nidmap, prange, wrange, wid)))
        for x in works:
            r = x.get()
            for i in range(partitions):
                dsets[i].update(r[i])

    pdsets = [set() for x in range(partitions)]
    for i in range(partitions):
        pdsets[i].update(dsets[i])
        for j in range(partitions):
            if i != j:
                pdsets[i].difference_update(dsets[j])
    pdcount = [len(x) for x in pdsets]

    dset = set().union(*dsets)
    iset = set(range(prange[partid], prange[partid + 1])) - dset
    haloset = set(nidmap.tolist()) - dset - iset

    print(
        "PartID:{}, Len(dset):{}, Len(iset):{}, Len(haloset):{} local:{}, Num_node:{}, pdcount:{}"
        .format(partid, len(dset), len(iset), len(haloset), local_node_count,
                graph.num_nodes(), pdcount))

    cachef.write("%s\n" % list(dset))
    cachef.write("%s\n" % list(iset))
    cachef.write("%s\n" % list(haloset))
    cachef.write("%s\n" % pdcount)
    cachef.close()
    return graph, node_feats, edge_feats, book, (partid, list(dset),
                                                 list(iset), list(haloset),
                                                 local_node_count,
                                                 graph.num_nodes(), pdcount)