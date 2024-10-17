#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# split: ['all', '1_2', '1_4', '1_8', '1_16']
#        ['skrylle_1_2', 'skrylle_1_4', 'skrylle_1_8', 'skrylle_1_16']
#        ['snoge_1_2', 'snoge_1_4', 'snoge_1_8', 'snoge_1_16']


dataset='finnwoods'
method='semisup'
exp='unimatch_large_1_2'
split='1_2'
prev_split='1_2'
prev_exp='supervised_large_1_2'


config=mlsd_pytorch/trees_tiny.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
path_prev_best=exp/finnwoods/supervised/$prev_exp/$prev_split
#path_prev_best=berzelius_jobs/finnwoods/supervised/$prev_exp/$prev_split
#path_prev_best=berzelius_exp/$prev_exp/$prev_split

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
