#!/bin/bash

NAME=$1
GPU=$2
GPU_IDX=$3
DATASET=$4
MASTER_PORT=$5
LOG_DIR=logs/$DATASET/$NAME
LOG_FILE=$LOG_DIR/train.log
mkdir -p $LOG_DIR

mkdir -p $LOG_DIR/python
cp -f *.py $LOG_DIR/python/
mkdir -p $LOG_DIR/python/data
cp -f ./data/*.py $LOG_DIR/python/data/
mkdir -p $LOG_DIR/python/utils
cp -f ./utils/*.py $LOG_DIR/python/utils/
mkdir -p $LOG_DIR/python/options
cp -f ./options/*.py $LOG_DIR/python/options/
mkdir -p $LOG_DIR/python/motion_loaders
cp -f ./motion_loaders/*.py $LOG_DIR/python/motion_loaders/
mkdir -p $LOG_DIR/python/models
cp -f ./models/*.py $LOG_DIR/python/models/
mkdir -p $LOG_DIR/python/models/vq
cp -f ./models/vq/*.py $LOG_DIR/python/models/vq
mkdir -p $LOG_DIR/python/models/transformer
cp -f ./models/transformer/*.py $LOG_DIR/python/models/transformer

echo NAME $1 | tee $LOG_FILE
echo GPU $2 | tee $LOG_FILE -a
echo GPU cuda:$GPU_IDX | tee $LOG_FILE -a
echo LOG_DIR $LOG_DIR | tee $LOG_FILE -a
echo LOG_FILE $LOG_FILE | tee $LOG_FILE -a
echo EXTRA ${@:5} | tee $LOG_FILE -a

# CUDA_VISIBLE_DEVICES=$GPU_IDX \
# python -m torch.distributed.launch --master_port 11234 --nproc_per_node=$GPU train_mask_transformer_ddp.py --name $NAME --dataset_name $DATASET  \
#     ${@:5} | tee $LOG_DIR/train.log -a

CUDA_VISIBLE_DEVICES=$GPU_IDX \
python -m torch.distributed.launch --master_port $MASTER_PORT --nproc_per_node=$GPU train_mask_transformer_ddp.py --name $NAME --dataset_name $DATASET  \
    ${@:6} | tee $LOG_DIR/train.log -a


# bash run_mtrans.sh mtrans_test 2 humanml3d --vq_name rvq_test --batch_size 64 --max_epoch 2000 --milestones 1000000 --attnj --attnt
