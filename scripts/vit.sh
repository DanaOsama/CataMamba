#!/bin/bash
#SBATCH --job-name=test             # Job name
#SBATCH --output=outputs/logs/train_clips_resnet101_gru.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=24          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

cd Cataracts_Multi-task/
module load conda
# module load cuda10.2/toolkit/10.2.89
eval "$(conda shell.bash hook)"
conda activate multitask # name of the conda environment

python train.py --architecture ViT --epochs 200 --optimizer AdamW --num_clips 8 \
    --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 0.003 \
    --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm True

# python train.py --run_name learning_rate_5e-3 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False