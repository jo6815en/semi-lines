#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# split: ['all', '1_2', '1_4'. '1_8', '1_16', '1_32', '1_64']

dataset='wireframe'
method='semisup'
exp='unimatch_1_2'
split='1_2'
pre_exp='supervised_1_2'
pre_split='1_2'

config=mlsd_pytorch/wireframe.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
path_pre_best=exp/wireframe/supervised/$pre_exp/$pre_split
#path_prev_best=berzelius_jobs/wireframe/supervised/$prev_exp/$prev_split


mkdir -p $save_path

torchrun \
    --standalone\
    --nnodes=1\
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 --config $config --prev_best $path_pre_best --is_wireframe True | tee $save_path/$now.log

