#!/bin/sh


#SBATCH --gpus 1

#SBATCH -t 24:00:00

# job name
#SBATCH -J WireUni_1
#
# Remap stdout and stderr to write to these files
#SBATCH -o WireframeUni_%A_%a.out
#SBATCH -e WireframeUni_%A_%a.out
#
# Get notified by email when the job starts, ends or fails
#SBATCH --mail-user=johanna.engman@math.lth.se
#SBATCH --mail-type=ALL

cat $0

cd /proj/berzelius-2023-228/users/x_joeng/UniMatch

export SLURM_NTASKS=1

apptainer exec --nv -B /proj/berzelius-2023-228/users/x_joeng/UniMatch:/home2/johannae/semi-lines/UniMatch uni.sif sh /home2/johannae/semi-lines/UniMatch/scripts/train_uni_wireframe.sh 1 1233

