import os
import sys
import time
import subprocess
import argparse
from threading import Thread
from os import path

algs = ["sage"]
datasets = ["ogbn-products"]
partitions = [16]
methods = ["metis"]
layers = [2]
eagers = ["True"]
bin_methods = ["prio"]
hiddens = [16]
in_feats = [-1]

WORKSPACE = "/g3"
LOG_LEVEL = "info"
N_EPOCHS = 100
BIN_SIZE = 1000

CO_DATALOADER = 5
IS_TRAIN = "True"


def check_status(logdir):
    d = "{}/tests/{}".format(WORKSPACE, logdir)
    if path.exists(d):
        cmd = "cd {} && tail -n 1 0.out".format(d)
        return "finishes" in os.popen(cmd).read()
    else:
        # pass this job
        print("{} not exists. Pass.".format(d))
        return True


def clean_previous_job():
    cmd = "cd {}/tests && python3 dist_kill.py --nodefile 16nodes.txt --username root".format(
        WORKSPACE)
    print(cmd)
    os.system(cmd)


def run_new_job(data, part, alg, method, layer, bin_size, eager, bin_method,
                hidden, in_feat, logdir):
    cmd = "cd {}/tests && ./test_g3_exp.sh ".format(WORKSPACE)

    args = " {} {} {} {}nodes.txt {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} ".format(
        WORKSPACE, "root", "0", part, LOG_LEVEL, "/dataset", data, method, 1,
        layer, N_EPOCHS, alg, 1, bin_size, CO_DATALOADER, IS_TRAIN, eager,
        bin_method, hidden, in_feat, logdir)

    cmd += args
    print(cmd)

    os.system(cmd)


def main(args):
    for part_idx, part in enumerate(partitions):
        for data_idx, data in enumerate(datasets):
            for method in methods:
                for layer in layers:
                    for eager in eagers:
                        for bin_method in bin_methods:
                            for hidden in hiddens:
                                for in_feat in in_feats:
                                    for alg in algs:
                                        # first check and start a job
                                        bin_size = BIN_SIZE

                                        LOGDIR = f"{data}.{method}.{part}.{layer}layers.{alg}.bin{bin_size}.{bin_method}.eager_{eager}.hidden_{hidden}.f{in_feat}.log"
                                        d = "{}/tests/{}".format(
                                            WORKSPACE, LOGDIR)
                                        if path.exists(d):
                                            print(
                                                "Logdir exists. Pass this job."
                                            )
                                            continue

                                        clean_start = check_status(LOGDIR)
                                        while True:
                                            if clean_start:
                                                print(
                                                    "\n===============================\nPrevious task has finished. First sleep for a while...."
                                                )
                                                clean_previous_job()
                                                print(
                                                    "Next, run experiment with algrithm: {} on dataset: {} with partitions: {}, method: {}, eager: {}, bin_method: {}, hidden: {}, in_feats: {}"
                                                    .format(
                                                        alg, data, part,
                                                        method, eager,
                                                        bin_method, hidden,
                                                        in_feat))
                                                time.sleep(5)
                                                print(
                                                    "\n###################################\nStart new job using algrithm: {} on dataset: {} with partitions: {}, method: {}, eager: {}, bin_method: {}, hidden: {}, in_feats: {}. Now is {}"
                                                    .format(
                                                        alg, data, part,
                                                        method, eager,
                                                        bin_method, hidden,
                                                        in_feat, time.ctime()))
                                                # start new job
                                                run_new_job(
                                                    data, part, alg, method,
                                                    layer, bin_size, eager,
                                                    bin_method, hidden,
                                                    in_feat, LOGDIR)
                                                # time.sleep(1 * 10)
                                                break
                                            else:
                                                print("Wait for cleaning...")
                                                time.sleep(1 * 30)
                                                clean_start = check_status(
                                                    LOGDIR)
                                        while True:
                                            finish = check_status(LOGDIR)
                                            if finish:
                                                print(
                                                    "Job of Algorithm: {} on dataset: {} with partitions: {}, method has finished\n###################################"
                                                    .format(
                                                        alg, data, part,
                                                        method))
                                                clean_previous_job()
                                                break
                                            else:
                                                print(
                                                    "Process is still running now... Current job is run algorithm: {} on dataset: {} with partitions: {}, method: {}, eager: {}, bin_method: {}, hidden: {}, in_feats: {}....... Now is: {}"
                                                    .format(
                                                        alg, data, part,
                                                        method, eager,
                                                        bin_method, hidden,
                                                        in_feat, time.ctime()))
                                                time.sleep(1 * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run expriments on AWS")
    parser.add_argument(
        "--alg",
        type=str,
        default="graphsage",
        help="Algorithm to run: graphsage, gcn, gin",
    )
    parser.add_argument(
        "--train_full",
        action="store_true",
        help="Train GNN on full batch",
    )
    parser.add_argument("--sample",
                        action="store_true",
                        help="Train GNN with sampling")

    args = parser.parse_args()
    main(args)