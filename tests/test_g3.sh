#!/bin/bash

WORKSPACE="/g3"
USER="root"
INTERFACE="0"

NODEFILE="16nodes.txt"
LOG_LEVEL="info"

DATASETPATH="/dataset"
DATASET="ogbn-products"

NUM_PARTS=$(wc -l ${NODEFILE} | awk '{print $1}')

PART_METHOD="metis"
NUM_HOPS=1
N_LAYERS=2
N_EPOCHS=100

OUTPUT="${DATASETPATH}/${DATASET}.${PART_METHOD}.${NUM_PARTS}"

LOGDIR="${DATASET}.${PART_METHOD}.${NUM_PARTS}.${N_LAYERS}layers.log"
MODEL_LOGDIR="${WORKSPACE}/model_log"
MODEL="sage"
LOG_EVERY=1

BIN_SIZE=1000
CO_DATALOADER=5

python3 dist_launcher.py \
    --nodefile ${NODEFILE} \
    --username ${USER} \
    --interface ${INTERFACE} \
    --nccl-debug INFO \
    --nccl-ib-disable 1 \
    --nccl-socket-ifname ${INTERFACE} \
    --log-dir ${LOGDIR} \
    " cd ${WORKSPACE}/tests; PYTHONFAULTHANDLER=1 \
    DMLC_LOG_FATAL_THROW=1 \
    DMLC_LOG_BEFORE_THROW=0 \
    DMLC_LOG_STACK_TRACE=0 \
    DMLC_LOG_DEBUG=0 \
    python3 train_g3.py \
--clusterfile ${NODEFILE} \
--logging-level ${LOG_LEVEL} \
--backend gloo \
--module runtime.models.${MODEL} \
--n-epochs ${N_EPOCHS} \
--n-hidden 16 \
--n-layers ${N_LAYERS} \
--num-mlp-layers 2 \
--learn_eps \
--graph_pooling_type \"sum\" \
--neighbor_pooling_type \"sum\" \
--dropout 0 \
--aggregator-type mean \
--local-rank 0 \
--num_gpus 1 \
--dataset ${DATASET} \
--out-path ${OUTPUT} \
--is-train True \
--bin_size ${BIN_SIZE} \
--co-dataloader ${CO_DATALOADER} \
--eager False \
--lr 0.003 \
--seed 1 \
--log-dir ${MODEL_LOGDIR} "
