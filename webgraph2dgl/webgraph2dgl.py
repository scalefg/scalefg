import networkx as nx
import dgl
import torch
import os
import argparse, time
import pdb
import numpy as np
from scipy.sparse import coo_matrix

from dgl.data.utils import load_graphs, save_graphs

webgraph_baseurl = "http://data.law.di.unimi.it/webdata"
arclist_suffix = "edges"
low_label = 0
high_label = 10
dgl_bin_dir = "../dataset"


def download(args):
    if os.path.isdir('./{}'.format(args.name)):
        print("Directory exists, skip download")
    else:
        print("Start download")

        os.mkdir('./{}'.format(args.name))
        cmd = 'wget -c {}/{}/{}.graph -P {}/'.format(webgraph_baseurl,
                                                     args.name, args.name,
                                                     args.name)
        os.system(cmd)

        cmd = 'wget -c {}/{}/{}.properties -P {}/'.format(
            webgraph_baseurl, args.name, args.name, args.name)
        os.system(cmd)

        print("Finish download")


def transform(args):
    if os.path.exists('{}/{}.{}'.format(args.name, args.name, arclist_suffix)):
        print("Edgelist exists, skip transformation")
    else:
        print("Start transform")
        cmd = 'java -cp "deps/*" it.unimi.dsi.webgraph.ArcListASCIIGraph {}/{} {}/{}.{}'.format(
            args.name, args.name, args.name, args.name, arclist_suffix)
        os.system(cmd)
        print("Finish transform")

    return '{}/{}.{}'.format(args.name, args.name, arclist_suffix)


def load_from_edgelist(args, arcpath):
    src = []
    dst = []
    with open(arcpath) as f:
        for line in f:
            edge = line.split()
            src.append(int(edge[0]))
            dst.append(int(edge[1]))

    print(len(src), len(dst))
    return np.array(src), np.array(dst)


def parse_edges(args, arcpath):
    srcpath = '{}/src_{}.npy'.format(args.name, args.name)
    dstpath = '{}/dst_{}.npy'.format(args.name, args.name)

    if os.path.exists(srcpath) and os.path.exists(dstpath):
        print("npy files exists, skip load_from_edgelist")
        print("Start loading np.array...")
        src_npy = np.load(srcpath)
        dst_npy = np.load(dstpath)
    else:
        print("Start loading edgelist...")
        # 1. Read file into src and dst
        src_npy, dst_npy = load_from_edgelist(args, arcpath)

        print("Start storing np.arrays...")
        # 2. Store into src.npy and dst.npy
        np.save(srcpath, src_npy)
        np.save(dstpath, dst_npy)

    return src_npy, dst_npy


def load2nx2dgl(arcpath):
    print("Start load graph")
    NG = nx.read_edgelist(arcpath, create_using=nx.DiGraph)
    print("Start DGLGraph transformation")
    G = dgl.from_networkx(NG)
    print("Finish load2dgl")
    return G


def load2scipy2dgl(src_npy, dst_npy):
    data = np.ones(len(src_npy), dtype=bool)
    num_nodes = max(max(src_npy), max(dst_npy)) + 1
    print("Start constructing scipy.sparse.matrix...")
    mat = coo_matrix((data, (src_npy, dst_npy)), shape=(num_nodes, num_nodes))

    print("Finish sparse matrix. Start DGLGraph transformation")
    G = dgl.from_scipy(mat)
    print("Finish load2scipy2dgl")
    return G


def save_dgl_bin(args, G, dgl_bin_dir):
    # Save
    filepath = "{}.bin".format(args.name)
    dgl_bin_path = dgl_bin_dir + "/{}".format(filepath)
    node_label = torch.randint(low_label,
                               high_label, (G.number_of_nodes(), ),
                               dtype=torch.float32)
    label_dict = {'labels': node_label}
    print("Saving to {}".format(dgl_bin_path))
    save_graphs(dgl_bin_path, G, label_dict)
    print("Graph {} saved in {}".format(args.name, dgl_bin_path))


def main(args):
    """
    Main function for webgraph
    """

    # 1. Download dataset
    download(args)

    # 2. Transform
    arcpath = transform(args)

    # 3. Parse edgelist
    src_npy, dst_npy = parse_edges(args, arcpath)

    # Option 4. Load arclist to nx to DGL
    # G = load2nx2dgl(arcpath, dgl_bin_dir)

    # Option 5. Load npy to scipy to DGL
    G = load2scipy2dgl(src_npy, dst_npy)

    # 6. Store DGL bin
    save_dgl_bin(args, G, dgl_bin_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='webgraph to dglgraph bin')
    parser.add_argument('--name', type=str, help='webgraph name')

    args = parser.parse_args()

    print(args)
    main(args)
