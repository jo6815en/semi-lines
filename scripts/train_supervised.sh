#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# split: ['all', '1_2', '1_4', '1_8', '1_16']

dataset='finnwoods'
method='supervised'
exp='supervised_large_1_2_2'
split='1_2'


config=mlsd_pytorch/trees_supervised.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
path_prev_best=exp/finnwoods/supervised/$prev_exp/$split
# --prev_best $path_prev_best

mkdir -p $save_path

torchrun \
    --standalone\
    --nnodes=1\
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 --config $config | tee $save_path/$now.log

