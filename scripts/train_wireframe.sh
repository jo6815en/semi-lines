#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='wireframe'
method='supervised'
exp='supervised_1_16'
split='1_16'


config=mlsd_pytorch/wireframe.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split


mkdir -p $save_path

torchrun \
    --standalone\
    --nnodes=1\
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 --config $config --is_wireframe True| tee $save_path/$now.log

