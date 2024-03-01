#!/bin/bash
#SBATCH --job-name=test             # Job name
#SBATCH --output=outputs/logs/train_clips_resnet50_lstm.%A_%a.txt   # Standard output and error log
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

# All defaults are correct
python train.py 
# python train.py --hidden_size 256 --epochs 100 --num_clips -1 --cnn_model resnet18 --rnn_model gru --weighted_loss True
# python train.py --hidden_size 256 --epochs 100 --num_clips -1 --cnn_model resnet18 --weighted_loss True
# python train.py --architecture CNN --weighted_loss True --num_clips -1
# python train.py --architecture ViT --epochs 150 
# python train.py --architecture Mamba --epochs 1