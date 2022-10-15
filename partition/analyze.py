#!/usr/bin/python3
import dgl
from tqdm import tqdm
import sys, os
from torch.multiprocessing import Pool, RLock
from multiprocessing.pool import ThreadPool

dataset = 'reddit' if len(sys.argv)<2 else sys.argv[1]  
partitions = 5 if len(sys.argv)<3 else int(sys.argv[2])

def find_node_d(partid, u, v, nidmap, prange, wrange, wid):
    def isLocal(x):
        return prange[partid]<=x and x<prange[partid+1]
    def findp(x):
        for i in range(1, len(prange)):
            if prange[i]>x:
                return i-1
    dsets = [set() for i in range(len(prange)-1)]
    for i in tqdm(range(len(u))[wrange[0]:wrange[1]], desc='#{}-{}'.format(partid, wid+1), position=wid, leave=False):
        x = nidmap[u[i]]
        if isLocal(x):
           continue
        y = nidmap[v[i]]
        dsets[findp(x)].add(int(y))
    return dsets

def process(partid, dataset, workers):
    graph, node_feats, edge_feats, book, graph_name, ntypes, etypes = dgl.distributed.load_partition('dataset/{}/dgl/{}.json'.format(dataset, dataset), partid)
    local_node_count = book.metadata()[partid]["num_nodes"]
    print("\npartition #{}: nodes {} (local {}), edges {}".format(partid, graph.num_nodes(), local_node_count, graph.num_edges()))
    
    u, v = graph.edges(form="uv", order="srcdst")
    nidmap = graph.ndata["_ID"]
    partitions = len(book.metadata())
    prange = [0]
    for x in book.metadata():
        prange += [prange[-1] + x["num_nodes"]]

    dsets = [set() for x in range(partitions)]
    with Pool(processes=workers) as pool:
        workload = int(len(u)/workers)+1
        splits = list(zip(range(0, len(u), workload), range(workload,len(u)+workload,workload)))
        works = []
        for wid, wrange in enumerate(splits):
            works.append(pool.apply_async(find_node_d, 
                    (partid, u, v, nidmap, prange, wrange, wid)))
        for x in works:
            r = x.get()
            for i in range(partitions):
                dsets[i].update(r[i])
    
    dset = set().union(*dsets)
    iset = set(range(prange[partid], prange[partid+1])) - dset
    
    return (partid, 
            len(dset),
            local_node_count,
            graph.num_nodes())

if __name__ == '__main__':
    tqdm.set_lock(RLock())
    results = []
    futures = []
    processes_limit = 2
    workers = os.cpu_count()
    with ThreadPool(processes=processes_limit) as pool:
        for i in range(partitions):
            futures.append(pool.apply_async(process, (i, dataset, workers)))
        results.extend([x.get() for x in futures])
    print()
    for result in results:
        partid, countd, countlocal, countall = result
        print("partition #{}: local {}, type-d {} ({:.2f}%), remote {}".format(partid, countlocal, countd, countd*1.0/countlocal*100, countall-countlocal))
