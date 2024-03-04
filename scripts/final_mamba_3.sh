


python train.py --run_name test_f4_Cata --seed 0 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 1 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18

python train.py --run_name e50_final2_CATA_seed0 --seed 0 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
python train.py --run_name e50_final2_CATA_seed1 --seed 1 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
python train.py --run_name e50_final2_CATA_seed2 --seed 2 --architecture Cata-Mamba --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
