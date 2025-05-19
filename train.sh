#!/bin/bash
#SBATCH --account=a-large-sc
#SBATCH --job-name=LSAI_project
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:06:00
#SBATCH --output=/users/%u/scratch/LSAI_project/logs/output_%j.out
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --signal=USR1@120
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc/environment/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue

TRAINING_CMD=" --sequence-length 2048 \
               --batch-size 1 \
               --learning-rate 5e-5 \
               --lr-warmup-steps 100 \
               --training-steps 1000 \
               --raise-error \
               --error-step 600"

if [ -n "$1" ]; then
    TRAINING_CMD="$TRAINING_CMD \
     --checkpoint-id $1"
fi
export WORKDIR="/users/$USER/scratch/LSAI_project"

exec srun --unbuffered python $WORKDIR/train.py  $TRAINING_CMD
