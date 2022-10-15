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


def preprocess_envs(args_envs):
    envs_map = {}
    for item in args_envs:
        i = item.find(":")
        if i != -1:
            key = item[:i]
            val = item[i + 1:]
        envs_map[key] = val
    return envs_map


def get_env(envs_map):
    envs = []
    # get system envs
    keys = ['OMP_NUM_THREADS', 'KMP_AFFINITY']
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            envs.append('export ' + k + '=' + v + ';')
    # get ass_envs
    for k, v in envs_map.items():
        envs.append('export ' + str(k) + '=' + str(v) + ';')
    return (' '.join(envs))


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
            tp = h[3]
            partid = h[4]
            r = h[5]
            hosts.append((i, sp, p, tp, partid, r))
    return hosts


def start_ssh(dirname, prog, node, sshport, port, th_port, username, fname):
    def run(prog):
        subprocess.check_call(prog, shell=True)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    pname = dirname + '/' + fname
    if username is not None:
        prog = 'ssh -o StrictHostKeyChecking=no ' + ' -l ' + username \
               + ' ' + node + ' -p ' + sshport + ' \'' + prog + '\'' \
               + ' > ' + pname + '.out' + ' 2>' + pname + '.err&'
    else:
        prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + sshport + ' \'' + prog + '\'' \
               + ' > ' + pname + '.out' + ' 2>' + pname + '.err&'

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

    # common env
    pass_envs = preprocess_envs(args.env)
    pass_envs['G3_NUM_NODES'] = str(num_nodes)
    pass_envs['G3_INTERFACE'] = str(args.interface)
    pass_envs['NCCL_DEBUG'] = args.nccl_debug
    pass_envs['NCCL_IB_DISABLE'] = args.nccl_ib_disable
    pass_envs['NCCL_SOCKET_IFNAME'] = args.nccl_socket_ifname

    username = ''
    if args.username is not None:
        username = args.username

    threads = []
    for i, (node, sshport, port, th_port, partid, rank) in enumerate(nodes):
        prog = get_env(pass_envs) + (' '.join(
            args.command)) + ' --partid ' + partid + ' --rank ' + rank
        threads.append(
            start_ssh(args.log_dir, prog, node, sshport, port, th_port,
                      username, str(i)))

    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser(
        description='Launch a distributed training job for G3')
    parser.add_argument(
        '-N',
        '--nodefile',
        required=True,
        type=str,
        help='the nodefile of all nodes which will run the job.')
    parser.add_argument('--interface',
                        type=str,
                        default='enp59s0f0',
                        help='the network interface to use')
    parser.add_argument(
        '--env',
        action='append',
        default=[],
        help='Given a pair of environment_variable:value, sets this value of \
                        environment variable for all workers and servers. Example OMP_NUM_THREADS:3'
    )
    parser.add_argument('--username',
                        type=str,
                        default='xinchen',
                        help='the username for ssh')
    parser.add_argument('--nccl-debug',
                        type=str,
                        default='INFO',
                        help='NCCL_DEBUG')
    parser.add_argument('--nccl-ib-disable',
                        type=int,
                        default=1,
                        help='NCCL_IB_DISABLE')
    parser.add_argument('--nccl-socket-ifname',
                        type=str,
                        default='enp59s0f0',
                        help='NCCL_SOCKET_IFNAME')
    parser.add_argument('--log-dir',
                        type=str,
                        default='g3log',
                        help='path to store logs')
    parser.add_argument('command',
                        nargs='+',
                        help='command for launching the program')

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
