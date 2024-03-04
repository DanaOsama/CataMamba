


# best_mamba refers to d_state=32, d_conv=4, lr=5e-5
# This one tests weight decay, scheduler, and label smoothing and the resnet model
python train.py --run_name good_mamba_round1 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name good_mamba_r1_resnet50 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name good_mamba_r1_wd-0.001 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name good_mamba_r1_wd-0.00001 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.00001 --learning_rate 5e-5  --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
