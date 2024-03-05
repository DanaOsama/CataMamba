# Mambas
# python train.py --run_name Cata-Mamba-v2_expand=3_epochs_25 --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 3 --epochs 20 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
# March 5: 3:59 PM: All are running 
# March 6: 1:00 AM seed 0 and 2 crashed, restarting them. Tmux 52
python train.py --run_name Test1_CLIPS_Cata-Mamba-v2_resnet18_expand_3_dilation_3_seed0 --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id b5vr17sf
# python train.py --run_name Test1_CLIPS_Cata-Mamba-v2_resnet18_expand_3_dilation_3_seed1 --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2
python train.py --run_name Test1_CLIPS_Cata-Mamba-v2_resnet18_expand_3_dilation_3_seed2 --seed 2 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id 2do9hrwa

############ TO BE DONE
# March 6: 2:00 AM, running seed 0 on tmux54 
python train.py --run_name Test2_CLIPS_seed0 --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2
python train.py --run_name Test2_CLIPS_seed1 --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2
python train.py --run_name Test2_CLIPS_seed2 --seed 2 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2

# LSTMS
# March 5: 3:59 PM: seed 0 is running
# March 6 1:00 AM: seed 0 is done. Name changed from epochs=40 to epochs=50 because it ran for 50 epochs
# They are running now on tmux53
# python train.py --run_name Test1_seed=0_epochs=50_numclips_9_clipsize_30 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 
python train.py --run_name Test1_seed=1_epochs=50_numclips_9_clipsize_30 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 
python train.py --run_name test1_seed=2_epochs=50_numclips_9_clipsize_30 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 

## CATARACTS
python train.py --run_name seed=0_epochs=40_numclips_9_clipsize_30 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
python train.py --run_name seed=1_epochs=40_numclips_9_clipsize_30 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
python train.py --run_name seed=2_epochs=40_numclips_9_clipsize_30 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
