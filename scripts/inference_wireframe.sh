#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# test_set: ['wireframe', 'finn', 'snoge', 'skrylle']
# method: ['supervised', 'semisup']

dataset='wireframe'
method='semisup'
exp='unimatch_1_16_finn'
split='finn_1_16'
test_set='finn'



config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
#save_path=berzelius_exp/$dataset/$method/$exp/$split
#save_path=berzelius_jobs/$dataset/$method/$exp/$split



torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    inference.py \
    --uni_config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 --test_set $test_set --config mlsd_pytorch/wireframe.yaml| tee
