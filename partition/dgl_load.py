import dgl
from dgl.base import NID, EID

import torch as th
import scipy
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import pdb
from dgl.data.utils import load_graphs

OGB_DIR = "/data/dataset"
LOAD_GRAPH = ["ogbn-papers100M", "ogb-paper100M"]
WEBGRAPH_DIR = "/dataset"
AMAZON_DIR = "/dataset/amazon"
YELP_DIR = "/dataset/yelp"

MAX_FEATURESIZE = 128


def create_simple_graph(prob=None, feat_size=-1):
    g = dgl.DGLGraph()
    g.add_nodes(9)
    g.add_edge([0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 3, 3, 4])
    g.add_edge([8, 8, 8, 7, 7, 6, 5], [7, 6, 5, 6, 5, 5, 4])

    # features
    if feat_size == -1:
        g.ndata['feat'] = th.arange(9)
    else:
        g.ndata['feat'] = th.arange(9)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['label'] = th.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
    g.ndata['labels'] = g.ndata['label']

    # mask
    g.ndata['train_mask'] = th.tensor(
        [True, True, True, False, False, False, False, False, False])
    g.ndata['val_mask'] = th.tensor(
        [False, False, False, True, True, True, False, False, False])
    g.ndata['test_mask'] = th.tensor(
        [False, False, False, False, False, False, True, True, True])

    if prob is not None:
        g.ndata[prob] = th.randn(g.number_of_nodes(), dtype=th.float)
        g.edata[prob] = th.randn(g.number_of_edges(), dtype=th.float)
    return g


def load_karate(prob=None, feat_size=-1):
    from dgl.data import KarateClubDataset

    # load kerate data
    data = KarateClubDataset()
    g = data[0]
    if feat_size == -1:
        g.ndata['features'] = th.tensor(
            [[x] for x in range(len(g.ndata['label']))], dtype=th.float32)
    else:
        g.ndata['features'] = th.tensor(
            [[x] for x in range(len(g.ndata['label']))], dtype=th.float32)
    g.ndata['labels'] = g.ndata['label']

    split = int(len(g.ndata['labels']) / 3)
    train_mask = th.zeros((g.number_of_nodes(), ), dtype=th.bool)
    train_mask[[x for x in range(0, split)]] = True
    val_mask = th.zeros((g.number_of_nodes(), ), dtype=th.bool)
    val_mask[[x for x in range(split, split * 2)]] = True
    test_mask = th.zeros((g.number_of_nodes(), ), dtype=th.bool)
    test_mask[[x for x in range(split * 2, len(g.ndata['labels']))]] = True

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    if prob is not None:
        g.ndata[prob] = th.randn(g.number_of_nodes(), dtype=th.float)
        g.edata[prob] = th.randn(g.number_of_edges(), dtype=th.float)
    return g, len(g.ndata['labels'])


def load_cora(prob=None, feat_size=-1):
    from dgl.data import CoraGraphDataset

    # load cora data
    data = CoraGraphDataset()
    g = data[0]

    if feat_size == -1:
        pass
    else:
        g.ndata['feat'] = th.randn([len(g.ndata['label']), feat_size],
                                   dtype=th.float32)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    # pdb.set_trace()
    if prob is not None:
        g.ndata[prob] = th.randn(g.number_of_nodes(), dtype=th.float)
        g.edata[prob] = th.randn(g.number_of_edges(), dtype=th.float)

    for etype in g.canonical_etypes:
        print("g.edges[{}].data: {}".format(etype, g.edges[etype].data))
    # pdb.set_trace()
    return g, data.num_labels


def load_reddit(prob=None, feat_size=-1):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]

    if feat_size == -1:
        pass
    else:
        g.ndata['feat'] = th.randn([len(g.ndata['label']), feat_size],
                                   dtype=th.float32)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    if prob is not None:
        g.ndata[prob] = th.randn(g.number_of_nodes(), dtype=th.float)
        g.edata[prob] = th.randn(g.number_of_edges(), dtype=th.float)
    return g, data.num_labels


def load_ogb(name, prob=None, feat_size=-1):
    from ogb.nodeproppred import DglNodePropPredDataset
    print('prob:', prob)
    print('load', name)
    if name in LOAD_GRAPH:
        data = DglNodePropPredDataset(name=name, root=OGB_DIR)
    else:
        data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    labels = labels[:, 0]

    # pdb.set_trace()
    if feat_size == -1:
        pass
    else:
        graph.ndata['feat'] = th.randn([len(graph.ndata['label']), feat_size],
                                       dtype=th.float32)

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx[
        'valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(), ), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(), ), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(), ), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)

    if prob is not None:
        print("Generate prob tensors in node and edge")
        graph.ndata[prob] = th.randn(graph.number_of_nodes(), dtype=th.float)
        graph.edata[prob] = th.randn(graph.number_of_edges(), dtype=th.float)
    return graph, num_labels


def load_amzn(prob=None, feat_size=-1):
    adj_full = scipy.sparse.load_npz(
        '{}/adj_full.npz'.format(AMAZON_DIR)).astype(np.bool)
    g = dgl.from_scipy(adj_full)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    num_nodes = g.num_nodes()

    adj_train = scipy.sparse.load_npz(
        '{}/adj_train.npz'.format(AMAZON_DIR)).astype(np.bool)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    role = json.load(open('{}/role.json'.format(AMAZON_DIR)))
    mask = np.zeros((num_nodes, ), dtype=bool)
    train_mask = mask.copy()
    train_mask[role['tr']] = True
    val_mask = mask.copy()
    val_mask[role['va']] = True
    test_mask = mask.copy()
    test_mask[role['te']] = True

    feats = np.load('{}/feats.npy'.format(AMAZON_DIR))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    class_map = json.load(open('{}/class_map.json'.format(AMAZON_DIR)))
    class_map = {int(k): v for k, v in class_map.items()}

    # Multi-label binary classification
    num_classes = len(list(class_map.values())[0])
    class_arr = np.zeros((num_nodes, num_classes))
    for k, v in class_map.items():
        class_arr[k] = v

    g.ndata['feat'] = th.tensor(feats, dtype=th.float)
    g.ndata['label'] = th.tensor(class_arr, dtype=th.float)
    g.ndata['train_mask'] = th.tensor(train_mask, dtype=th.bool)
    g.ndata['val_mask'] = th.tensor(val_mask, dtype=th.bool)
    g.ndata['test_mask'] = th.tensor(test_mask, dtype=th.bool)

    if feat_size == -1:
        pass
    else:
        g.ndata['feat'] = th.randn([len(g.ndata['label']), feat_size],
                                   dtype=th.float32)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    if prob is not None:
        g.ndata[prob] = th.randn(g.number_of_nodes(), dtype=th.float)
        g.edata[prob] = th.randn(g.number_of_edges(), dtype=th.float)

    return g, num_classes


def load_yelp(prob=None, feat_size=-1):
    adj_full = scipy.sparse.load_npz(
        '{}/adj_full.npz'.format(YELP_DIR)).astype(np.bool)
    g = dgl.from_scipy(adj_full)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    num_nodes = g.num_nodes()

    adj_train = scipy.sparse.load_npz(
        '{}/adj_train.npz'.format(YELP_DIR)).astype(np.bool)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    role = json.load(open('{}/role.json'.format(YELP_DIR)))
    mask = np.zeros((num_nodes, ), dtype=bool)
    train_mask = mask.copy()
    train_mask[role['tr']] = True
    val_mask = mask.copy()
    val_mask[role['va']] = True
    test_mask = mask.copy()
    test_mask[role['te']] = True

    feats = np.load('{}/feats.npy'.format(YELP_DIR))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    class_map = json.load(open('{}/class_map.json'.format(YELP_DIR)))
    class_map = {int(k): v for k, v in class_map.items()}
    num_classes = len(list(class_map.values())[0])
    class_arr = np.zeros((num_nodes, num_classes))
    for k, v in class_map.items():
        class_arr[k] = v

    g.ndata['feat'] = th.tensor(feats, dtype=th.float)
    g.ndata['label'] = th.tensor(class_arr, dtype=th.long)
    g.ndata['train_mask'] = th.tensor(train_mask, dtype=th.bool)
    g.ndata['val_mask'] = th.tensor(val_mask, dtype=th.bool)
    g.ndata['test_mask'] = th.tensor(test_mask, dtype=th.bool)

    if feat_size == -1:
        pass
    else:
        g.ndata['feat'] = th.randn([len(g.ndata['label']), feat_size],
                                   dtype=th.float32)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    if prob is not None:
        g.ndata[prob] = th.randn(g.number_of_nodes(), dtype=th.float)
        g.edata[prob] = th.randn(g.number_of_edges(), dtype=th.float)

    return g, num_classes


def load_webgraph(gname, prob=None, feat_size=-1):
    g, label_dict = load_graphs("{}/{}.bin".format(WEBGRAPH_DIR, gname))
    g = g[0]
    labels = label_dict['labels']
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    g.ndata['label'] = labels

    # features
    if feat_size == -1:
        g.ndata['feat'] = th.randn([len(g.ndata['label']), MAX_FEATURESIZE],
                                   dtype=th.float32)
    else:
        g.ndata['feat'] = th.randn([len(g.ndata['label']), feat_size],
                                   dtype=th.float32)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    # mask
    node_size = g.number_of_nodes()
    train_size = int(node_size * 0.7)
    val_size = int(node_size * 0.2)
    test_size = node_size - train_size - val_size

    train_tensor = th.zeros(node_size, dtype=th.bool)
    val_tensor = th.zeros(node_size, dtype=th.bool)
    test_tensor = th.zeros(node_size, dtype=th.bool)

    train_tensor[:train_size] = 1
    val_tensor[train_size:-test_size] = 1
    test_tensor[-test_size:] = 1

    g.ndata['train_mask'] = train_tensor
    g.ndata['val_mask'] = val_tensor
    g.ndata['test_mask'] = test_tensor

    return g


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
