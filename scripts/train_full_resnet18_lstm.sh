#!/bin/bash
#SBATCH --job-name=test             # Job name
#SBATCH --output=outputs/logs/train_full_resnet18_lstm.%A_%a.txt   # Standard output and error log
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
wandb online 
# python train.py --num_clips -1 --rnn_model lstm --cnn_model resnet18 --hidden_size 256
python train.py --num_clips -1 --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --random_int 804145 --wandb_run_id eq89uao2 --resume_training True