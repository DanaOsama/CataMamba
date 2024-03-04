


# python train.py --run_name CATARACTS_amazing_wd=0.0 --num_classes 18 --dataset 9_CATARACTS --architecture Cata-Mamba --mamba_num_blocks 1 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 1 --num_clips 20 --clip_size 20 --optimizer AdamW --scheduler Cosine --weight_decay 0.0000 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True

python train.py --run_name e3_amazing_wd=0.0 --architecture Cata-Mamba --mamba_num_blocks 1 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0000 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
python train.py --run_name e3_amazing_lr=5e-6 --architecture Cata-Mamba --mamba_num_blocks 1 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 3 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-6 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
