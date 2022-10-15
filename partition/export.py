#!/usr/bin/python3
import dgl, sys
from tqdm import tqdm, trange

dataset = sys.argv[1]
graph = None
addloop = False
if "loop" in dataset:
    addloop = True
    dataset = dataset.split("+")[0]

if dataset == "cora":
    graph = dgl.data.CoraGraphDataset()[0]
elif dataset == "reddit":
    graph = dgl.data.RedditDataset()[0]
else:
    import ogb
    from ogb.nodeproppred import DglNodePropPredDataset
    dataset_loader = DglNodePropPredDataset(name=dataset)
    graph, _ = dataset_loader[0]

if addloop:
    graph = dgl.add_self_loop(graph)
    dataset = dataset + "+loop"

u, v = graph.edges(order="srcdst", form="uv")

node = -1
f = open("dataset/"+dataset+"/"+dataset+".graph", "w")
f.write("{} {:.0f}".format(graph.num_nodes(), graph.num_edges()/2))
for i in trange(len(u)):
    if node != u[i]:
        while node != u[i]:
            node += 1
            f.write("\n")
    f.write("{} ".format(int(v[i])+1))
f.write("\n")
f.close()


