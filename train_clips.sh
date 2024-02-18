#!/bin/bash
#SBATCH --job-name=test             # Job name
#SBATCH --output=outputs/logs/full_video_train.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=24          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

# parser.add_argument("--num_classes", type=int, help="Number of classes, default=10", default=10)
# parser.add_argument("--num_clips", type=int, help="Number of clips to sample from each video", default=2)
# parser.add_argument("--clip_size", type=int, help="Number of frames in each clip", default=20)
# parser.add_argument("--step_size", type=int, help="Number of frames to skip when sampling clips", default=1)


cd Cataracts_Multi-task/
conda activate multitask # name of the conda environment
python train.py --num_clips 8 --epochs 50 --clip_size 20 --step_size 1