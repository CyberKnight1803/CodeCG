#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH -t 0-10:00
#SBATCH --job-name="codemix"
#SBATCH -o /scratch/aruna/codecg-outs/nl2nl-%j.out
#SBATCH -e /scratch/aruna/codecg-outs/nl2nl-%j.err
#SBATCH --mail-user=f20190083@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=NONE

# Load modules
spack unload 
spack load gcc@11.2.0
spack load cuda@11.7.1%gcc@11.2.0 arch=linux-rocky8-zen2
spack load python@3.9.13%gcc@11.2.0 arch=linux-rocky8-zen2


# Activate Environment
source ~/omkar/envs/codemix/bin/activate

# Run 
cd /home/aruna/omkar/codecg

srun ~/omkar/envs/codemix/bin/python main.py \
--run_name nl2nl-run-1 \
--logger wandb \
--path_logs "/scratch/aruna/codecg-logs" \
--path_base_models "/scratch/aruna/codecg-base-models" \
--path_cache_datasets "/scratch/aruna/codecg-dataset" \
--workers 8 \
--path_save_nl_encoder "/scratch/aruna/codecg-saved-models/codecg-nl-encoder" \
--path_save_nl_decoder "/scratch/aruna/codecg-saved-models/codecg-nl-decoder" \
--epochs 5 \