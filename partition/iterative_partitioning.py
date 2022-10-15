#!/usr/bin/python3
import sys
from collections import Counter
from tqdm import tqdm, trange
import random
from torch.multiprocessing import Pool

dataset = sys.argv[1] if len(sys.argv)>1 else "cora"
k = int(sys.argv[2]) if len(sys.argv)>2 else 10

def load_graph(dataset, k):
    f = open("dataset/{}/{}.graph".format(dataset, dataset))
    n, m = f.readline().split(" ")
    n, m = int(n), int(m)
    g = []
    for i in trange(n, desc="loading graph"):
        l = f.readline().strip()
        nset = set([int(x)-1 for x in l.split(" ")]) if len(l)>0 else set()
        g.append(nset)
    f = open("dataset/{}/{}.parts.{}".format(dataset, dataset, k))
    passign = [int(x) for x in f.readlines()]
    return g, passign

def init_partitions(passign):
    k = max(passign)+1
    p = [set() for x in range(k)]
    pdc = [Counter() for x in range(k)]
    for i, x in tqdm(list(enumerate(passign)), desc="initializing partition"):
        p[x].add(i)
        pdc[x].update(g[i])
    for i in range(k):
        for x in p[i]:
            if x in pdc[i]:
                pdc[i].pop(x)
    return p, pdc

def output_partition(p, dataset, k):
    r = []
    for i in range(len(p)):
        for x in p[i]:
            r.append((x,i))
    r = sorted(r)
    f = open("dataset/{}/{}.parts.{}.g3".format(dataset,dataset, k), "w")
    for x in r:
        f.write("{}\n".format(x[1]))
    f.close()

def log_partition(pdc, it_num):
    pd = [set(x.elements()) for x in pdc]
    dc = sorted([(len(x), i) for i, x in enumerate(pd)])
    print("iteration {} - ".format(it_num), end="")
    for x in dc:
        print("#{}: {} ".format(x[1], x[0]), end="")
    print()
    return dc[0][0]/dc[-1][0]

def find_swap_out(g, p, cset):
    most = -(1<<30)
    ns = -1
    nsset = []
    for x in p:
        xset = g[x]&cset
        if len(xset) > most:
            most = len(xset)
            ns = x
            nsset = xset
    return most, ns, nsset

def find_swap_in(g, p, psd):
    least = 1<<30
    nt = -1
    for x in p:
        xset = g[x] - psd
        if len(xset) < least:
            least = len(xset)
            nt = x
    return least, nt

def partition_swap(g, p, pdc):
    pd = [set(x.elements()) for x in pdc]
    dc = sorted([(len(x), i) for i, x in enumerate(pd)])
    ps, pt = dc[-1][1], dc[0][1]

    c = pdc[ps]
    cset = set([x for x in c if c[x]==1])
    most, ns, nsset = find_swap_out(g, p[ps], cset)

    #workers = 1
    #with Pool(processes=workers) as pool:
    #    nodes = list(p[ps])
    #    workload = int(len(nodes)/workers)+1
    #    splits = list(zip(range(0, len(nodes), workload), range(workload,len(nodes)+workload,workload)))
    #    works = []
    #    for w in splits:
    #        works.append(pool.apply_async(find_swap_out,
    #            (g, nodes[w[0]:w[1]], cset)))
    #    for x in works:
    #        _most, _ns, _nsset = x.get()
    #        if _most > most:
    #            most, ns, nsset = _most, _ns, _nsset
   
    psd = ((pd[ps]-set(nsset))&p[ps])-{ns}
    least, nt = find_swap_in(g, p[pt], psd)
    #with Pool(processes=workers) as pool:
    #    nodes = list(p[pt])
    #    workload = int(len(nodes)/workers)+1
    #    splits = list(zip(range(0, len(nodes), workload), range(workload,len(nodes)+workload,workload)))
    #    works = []
    #    for w in splits:
    #        works.append(pool.apply_async(find_swap_in,
    #            (g, nodes[w[0]:w[1]], psd)))
    #    for x in works:
    #        _least, _nt = x.get()
    #        if _least < least:
    #            least, nt = _least, _nt

    p[ps].remove(ns)
    pdc[ps].subtract(g[ns]-p[ps])
    p[ps].add(nt)
    pdc[ps].update(g[nt]-p[ps])
    if nt in pdc[ps]:
        pdc[ps].pop(nt)
    pdc[ps][ns] = len(g[ns]&p[ps])

    p[pt].remove(nt)
    pdc[pt].subtract(g[nt]-p[pt])
    p[pt].add(ns)
    pdc[pt].update(g[ns]-p[pt])
    if ns in pdc[pt]:
        pdc[pt].pop(ns)
    pdc[pt][nt] = len(g[nt]&p[pt])
    

if __name__ == '__main__':
    g, passign = load_graph(dataset, k)
    p, pdc = init_partitions(passign)

    last_rate = 0
    for i in range(10000):
        partition_swap(g, p, pdc)
        rate = log_partition(pdc, i)
        if i % 10 == 0: 
            if rate <= last_rate:
                output_partition(p, dataset, k)
                exit(0)
            last_rate = rate 
    output_partition(p, dataset, k)
