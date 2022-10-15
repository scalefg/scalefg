#!/usr/bin/python
"""
Launch a distributed job
"""
import argparse
import os, sys
import signal
import logging
import subprocess
from multiprocessing import Pool, Process
from threading import Thread


def get_hosts_from_file(filename):
    with open(filename) as f:
        tmp = f.readlines()
    assert len(tmp) > 0
    hosts = []
    for h in tmp:
        if len(h.strip()) > 0:
            # parse addresses of the form ip:port
            h = h.split()
            i = h[0]
            # sp = h[1]
            sp = '22'
            p = h[2]
            r = h[3]
            hosts.append((i, sp, p, r))
    return hosts


def start_ssh(prog, node, sshport, port, rank, username, fname):
    def run(prog):
        subprocess.check_call(prog, shell=True)

    if username is not None:
        prog = 'ssh -o StrictHostKeyChecking=no ' + ' -l ' + username \
               + ' ' + node + ' -p ' + sshport + ' \"' + prog + '\"'
    else:
        prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + sshport + ' \"' + prog + '\"'

    print(prog)

    thread = Thread(target=run, args=(prog, ))
    thread.setDaemon(True)
    thread.start()
    return thread


def submit(args):
    nodes = get_hosts_from_file(args.nodefile)
    num_nodes = len(nodes)
    assert num_nodes >= 1
    print('Launch %d nodes' % (num_nodes))

    username = ''
    if args.username is not None:
        username = args.username

    threads = []
    for i, (node, sshport, port, rank) in enumerate(nodes):
        threads.append(
            start_ssh(
                "ps -ef | grep {} | awk '{{print \$2}}' | xargs kill -9".
                format(args.file), node, sshport, port, rank, username,
                'node' + str(i)))

    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser(
        description='Kill python jobs in distributed G3')
    parser.add_argument(
        '-N',
        '--nodefile',
        required=True,
        type=str,
        help='the nodefile of all nodes which will kill the job.')
    parser.add_argument('--file',
                        type=str,
                        default='train_g3.py',
                        help='Run file')
    parser.add_argument('--username',
                        type=str,
                        default='xinchen',
                        help='the username for ssh')

    args = parser.parse_args()

    # check necessary args
    assert args.nodefile

    submit(args)


def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)


if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()
