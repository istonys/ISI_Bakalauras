#!/bin/bash
#SBATCH -p main
#SBATCH -t 06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -o logs/liepa_prep_%j.out
#SBATCH -e logs/liepa_prep_%j.err

set -e
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
source .venv/bin/activate

python -m src.prepare_liepa_demand \
  --liepa_root  /scratch/lustre/home/$USER/data/LIEPA \
  --demand_root /scratch/lustre/home/$USER/data/DEMAND \
  --output_dir  /scratch/lustre/home/$USER/data/LIEPA_DEMAND \
  --matched_hours 0
