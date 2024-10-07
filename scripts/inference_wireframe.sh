#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# method: ['unimatch', 'supervised']

dataset='wireframe'
method='supervised'
exp='supervised_1_2'
#exp='supervised_1_16_2'
split='1_2'

#exp='unimatch_no_cutmix_1_16_2'
#exp='unimatch_1_2'
#exp='supervised_large_1_2'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
# save_path=exp/$dataset/$method/$exp/$split
#save_path=berzelius_exp/$dataset/$method/$exp/$split
save_path=berzelius_jobs/$dataset/$method/$exp/$split



torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    inference.py \
    --uni_config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 --test_set wireframe --config mlsd_pytorch/wireframe.yaml| tee
