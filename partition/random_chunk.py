#!/usr/bin/python3
import argparse
import random

nmap = {
        "reddit": 232965,
        "ogbn-products": 2449029,
    }

argparser = argparse.ArgumentParser("Partition builtin graphs")
argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: cora, reddit, ogb-product, ogb-paper100M')
argparser.add_argument('--method', type=str, default='random',
                           help='partitioning method')
argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
args = argparser.parse_args()

fn = "dataset/{}/{}.parts.{}.{}".format(args.dataset, args.dataset, args.num_parts, args.method)
f = open(fn, "w")

N = nmap[args.dataset]

if args.method == "random":
    for x in range(N):
        f.write("%s\n"%random.randint(0,args.num_parts-1))

if args.method == "chunk":
    size = N / args.num_parts
    for x in range(N):
        f.write("%s\n"%(int(x/size)))

f.close()

