# #!/bin/bash
# #SBATCH --job-name=test             # Job name
# #SBATCH --output=outputs/logs/train_clips_resnet101_gru.%A_%a.txt   # Standard output and error log
# #SBATCH --nodes=1                   # Run all processes on a single node    
# #SBATCH --ntasks=1                  # Run on a single CPU
# #SBATCH --mem=40G                   # Total RAM to be used
# #SBATCH --cpus-per-task=24          # Number of CPU cores
# #SBATCH --gres=gpu:1                # Number of GPUs (per node)
# #SBATCH -p gpu                      # Use the gpu partition
# #SBATCH --time=12:00:00             # Specify the time needed for your experiment
# #SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

# cd Cataracts_Multi-task/
# module load conda
# # module load cuda10.2/toolkit/10.2.89
# eval "$(conda shell.bash hook)"
# conda activate multitask # name of the conda environment
python train.py --run_name d_state=8,no_grad_clip --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 8 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False
python train.py --run_name d_state=16,no_grad_clip --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 16 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False
python train.py --run_name d_state=32,no_grad_clip --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False

python train.py --run_name d_state=8 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 8 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name d_state=16 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 16 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name d_state=32 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
