#!/bin/bash
#SBATCH -o train_adam.log-%j
# SBATCH --job-name=AdamTrain
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=40
# SBATCH --mem=250G
# SBATCH --time=23:59:00

source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.4
module load nccl/2.10.3-cuda11.4

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Set up rendezvous parameters
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 29000-29999 -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NODELIST: $SLURM_NODELIST"

# Use srun to ensure the job is distributed
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_adam.py \
    config/train_gpt2_small_adam.py \
    --batch_size=12 \
    --gradient_accumulation_steps=5