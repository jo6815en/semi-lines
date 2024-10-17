#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# test_set: ['finn', 'snoge', 'skrylle']
# method: ['supervised', 'semisup']

dataset='finnwoods'
method='supervised'
split='all'
exp='supervised_weibull_test'
test_set='finn'

#config=mlsd_pytorch/trees_tiny.yaml
config=mlsd_pytorch/trees_supervised.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
#save_path=berzelius_exp/$exp/$split
#save_path=berzelius_jobs_OLD/$dataset/$method/$exp/$split

torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    inference.py \
    --uni_config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 --test_set $test_set --config  $config | tee
