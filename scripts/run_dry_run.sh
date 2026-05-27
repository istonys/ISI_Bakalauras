#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/dry_run_%j.out
#SBATCH -e logs/dry_run_%j.err

set -e
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
mkdir -p logs

source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m src.train --config configs/dry_run.yaml
