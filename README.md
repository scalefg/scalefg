# Scalable and Efficient Full-Graph GNN Training for Large Graphs
Source code for Scalable and Efficient Full-Graph GNN Training for Large Graphs.

## Directory Structure
```
├── model_log       # model checkpoints
├── partition       # partition code
├── runtime         # core runtime
│   ├── library     # communication library
│   ├── models      # GNN models
│   │   ├── gat
│   │   ├── gcn
│   │   └── sage
│   ├── optimizer   # layer optimizder
│   ├── schema      # data structures
│   └── worker      # executor process
├── tests           # test scripts
└── webgraph2dgl    # transform webgraph to DGL binary
    └── deps        # webgraph required java dependencies
```

## Setup
### Environment
#### Hardware Dependencies
- 700GB host memory for graph partition
- 1 GPU in each worker

#### Software Dependencies
- Ubuntu 18.04
- Python 3.6.9
- CUDA 11.1
- PyTorch 1.10.1
- DGL 0.6.1
- OGB 1.2.1

You can install these requirements by using `conda install` or `pip3 install`.

### Execution
#### 1. Graph partition
First you should use balance partition strategy to partition graph. Please refer to [README.md](partition/README.md) for partitioning details. We also support other partition strategies, e.g., 'metis', 'chunk', and 'random'. You can use `--method` argument in [dgl_parse.py](partition/dgl_parse.py) to adopt them.

After partitioning, you should distribute each subgraph to the corresponding worker. We recommend you use NFS to allow all workers accessing the same directory simultaneously.
#### 2. Run scripts
##### Run single job
`test_g3.sh` is an example script that helps run GNN training over G3. Here we briefly describe its key parameters
```
WORKSPACE="/g3"                                 # Main workspace. All workers should maintain the same.
USER="root"                                     # User name used to run command in every worker
INTERFACE="0"                                   # Network interface name
NODEFILE="16nodes.txt"                          # Cluster info filename
LOG_LEVEL="info"                                # G3_LOG_LEVEL, by default use "info"

DATASETPATH="/dataset"                          # Path to subgraph
DATASET="reddit"                                # Subgraph name 
NUM_PARTS=$(wc -l ${NODEFILE}|awk '{print $1}') # Number of subgraph, should equals to cluster size

PART_METHOD="g3"                                # Partition strategy, can be "g3", "metis", "chunk", and "random"
NUM_HOPS=1                                      # Number of hop for halo nodes, should be 1
N_LAYERS=2                                      # Number of layers
N_EPOCHS=100                                    # Number of epochs to run

OUTPUT="${DATASETPATH}/${DATASET}.${PART_METHOD}.${NUM_PARTS}"  # Subgraph path

LOGDIR="${DATASET}.${PART_METHOD}.${NUM_PARTS}.${N_LAYERS}layers.log"   # Log path
MODEL_LOGDIR="${WORKSPACE}/model_log"           # Checkpoint path

MODEL="sage"                                    # Model type, can be "sage", "gcn", "gat"
LOG_EVERY=1                                     # Log every X iterations
BIN_SIZE=1000                                   # Bin size
CO_DATALOADER=5                                 # Concurrent dataloader

```

Current [test_g3.sh](tests/test_g3.sh) trains GraphSAGE with ogbn-products dataset in a 16-node cluster. You may specify other tasks accordingly.

##### Run batch of jobs
We support running a batch of jobs using [run_exp.py](test/run_exp.py).

```
algs = ["sage"]                     # Models
datasets = ["ogbn-products"]        # Datasets
partitions = [16]                   # Cluster sizes
methods = ["g3"]                    # Partition strategies
layers = [2]                        # Number of layers
eagers = ["True"]                   # Use inter-pipeline or not
bin_methods = ["prio"]              # Use intra-pipeline or not
hiddens = [16]                      # Hidden dimensions
in_feats = [-1]                     # Input features, -1 means use default features of datasets
```

You may modify the above lists to run batch of jobs. Note that you must run [run_exp.py](test/run_exp.py) in Worker-0 such that it can check if the previous process has finished or not.