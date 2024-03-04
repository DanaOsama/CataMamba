python train.py --run_name final3_INCASE --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 25 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name final4_INCASE --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 64 --d_conv 4 --expand 2 --epochs 25 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True