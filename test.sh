#!/bin/sh
PARTITION=Segmentation

exp_name=$1 # split0 - split3
dataset=$2  # pascal coco
gpu=$3      # 0~7
nlp=$4      # word2vec clip
weight=$5   # ckp path
net=resnet50 # vgg resnet50

config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml

CUDA_VISIBLE_DEVICES=${gpu} python3 -u test.py \
        --config=${config} \
        --nlp=${nlp} \
        --weight=${weight}
