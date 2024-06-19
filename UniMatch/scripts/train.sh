#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='finnwoods'
method='unimatch'
exp='unimatch_weibull_1_2_test2'
split='1_2'
#split='sam_segs_1_2'
prev_split='1_2'
prev_exp='supervised_ber_1_2'
#prev_exp='supervised_ber_1_2'


config=mlsd_pytorch/trees_tiny.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
#path_prev_best=exp/finnwoods/supervised/$prev_exp/$prev_split
path_prev_best=berzelius_jobs/finnwoods/supervised/$prev_exp/$prev_split
#path_prev_best=berzelius_exp/$prev_exp/$prev_split
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
    --save-path $save_path --port $2 2>&1 --config $config --prev_best $path_prev_best| tee $save_path/$now.log
