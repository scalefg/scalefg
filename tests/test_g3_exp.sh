#!/bin/bash

WORKSPACE=$1
USER=$2
INTERFACE=$3

NODEFILE=$4
LOG_LEVEL=$5

DATASETPATH=$6
DATASET=$7

NUM_PARTS=$(wc -l ${NODEFILE} | awk '{print $1}')

PART_METHOD=$8
NUM_HOPS=$9
N_LAYERS=${10}
N_EPOCHS=${11}

OUTPUT="${DATASETPATH}/${DATASET}.${PART_METHOD}.${NUM_PARTS}"

MODEL_LOGDIR="${WORKSPACE}/model_log"
MODEL=${12}
LOG_EVERY=${13}

BIN_SIZE=${14}
CO_DATALOADER=${15}

IS_TRAIN=${16}
EAGER=${17}
BIN_METHOD=${18}

HIDDEN=${19}
IN_FEATS=${20}

LOGDIR=${21}

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
--n-hidden ${HIDDEN} \
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
--is-train ${IS_TRAIN} \
--bin_size ${BIN_SIZE} \
--co-dataloader ${CO_DATALOADER} \
--eager ${EAGER} \
--bin_method ${BIN_METHOD} \
--lr 0.003 \
--seed 1 \
--in-feats ${IN_FEATS} \
--log-dir ${MODEL_LOGDIR} "
