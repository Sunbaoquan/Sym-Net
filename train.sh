#!/bin/sh
PARTITION=Segmentation


exp_name=$1
dataset=$2 # pascal coco
gpu=$3
nlp=$4
resume=$5

if [ ${gpu} -eq 2 ]; then
  GPU_ID=0,1
elif [ ${gpu} -eq 1 ]; then
  GPU_ID=0
elif [ ${gpu} -eq 4 ]; then
  GPU_ID=0,1,2,3
else
  echo "Only 1, 2 and 4 gpu number are supperted"
  exit 1
fi

net=resnet50 # vgg resnet50
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=${gpu} --master_port=1234 train.py \
        --config=${config} --nlp=${nlp} --resume=${resume}