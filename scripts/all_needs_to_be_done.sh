# FLAGS {"TODO": Not run at all, "RUNNING": currently running, "DONE": finished running, "NEED TO RESUME": need to be resumed}

# Mambas
# DONE
# python train.py --run_name Cata-Mamba-v2_expand=3_epochs_25 --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 3 --epochs 20 --num_clips -1 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True
# March 5: 3:59 PM: All are running 
# March 6: 1:00 AM seed 0 and 2 crashed, restarting them. Tmux 52
# python train.py --run_name Test1_CLIPS_Cata-Mamba-v2_resnet18_expand_3_dilation_3_seed0 --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id b5vr17sf
# python train.py --run_name Test1_CLIPS_Cata-Mamba-v2_resnet18_expand_3_dilation_3_seed1 --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2
# python train.py --run_name Test1_CLIPS_Cata-Mamba-v2_resnet18_expand_3_dilation_3_seed2 --seed 2 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id 2do9hrwa
# =================================================================#

# Running Test1 on CATARACTS dataset == March 6, 7:05 AM
# March 6, 7:05 AM => Seed 0 is running on tmux 52: https://wandb.ai/danaosama/Cata-Mamba-v2/runs/kr32euhe
# March 6, 7:05 AM => Seed 1 is running on tmux 54: https://wandb.ai/danaosama/Cata-Mamba-v2/runs/1lk4shc5
# March 6, 7:28 AM => Seed 2 is running on tmux 59: https://wandb.ai/danaosama/Cata-Mamba-v2/runs/3tu3sjq3
# RUNNING: Resume training those three runs = > tmux 58 @ 10:30 AM (ALL OF THE SEEDS ARE RESUMED at tmux 58)

# DONE 
# python train.py --run_name Test1_seed0_CLIPS_Cata-Mamba-v2_resnet18_CATARACTS --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18 --resume_training True --wandb_run_id kr32euhe
# python train.py --run_name Test1_seed1_CLIPS_Cata-Mamba-v2_resnet18_CATARACTS --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18 --resume_training True --wandb_run_id 1lk4shc5
# DONE
# python train.py --run_name Test1_seed2_CLIPS_Cata-Mamba-v2_resnet18_CATARACTS --seed 2 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18 --resume_training True --wandb_run_id 3tu3sjq3


############ TO BE DONE
# March 6: 2:00 AM, running seed 0 on tmux54 -- DONE
# March 6: 7:27 AM, running seeds 1 and 2 on tmux 59
# python train.py --run_name Test2_CLIPS_seed0 --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2

# MARCH 6, 7:11 PM
# NEED TO RESUME
# This is the best cata-mamba but with Resnet50
# DONE
# python train.py --run_name Test2_CLIPS_seed1 --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id otwf2zuu
# Running on tmux 61. --wandb_run_id 8y3wio8e
# RESUMED ON TMUX 66 - March 7, 7:00 AM
python train.py --run_name Test2_CLIPS_seed2 --seed 2 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id 8y3wio8e

# This is the best cata-mamba but with Resnet50 and running on CATARACTS dataset
# RUNNING ON TMUX64 - 10:28 PM
# DONE March 7, 5:44 AM
# python train.py --run_name Test2_CLIPS_seed0_CATARACTS --seed 0 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18

# DONE
# python train.py --run_name Test2_CLIPS_seed1_CATARACTS --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18
# Not yet started but scheduled on tmux64 - 5:45 AM
# Will need to be resumed
python train.py --run_name Test2_CLIPS_seed2_CATARACTS --seed 1 --architecture Cata-Mamba-v2 --mamba_num_blocks 2 --dilation_levels 3 --cnn_model resnet50 --d_state 32 --d_conv 4 --expand 2 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18 --resume_training True --wandb_run_id 511hljue

# Need to run the FC model for ablation study - Wed 6 March 10:15 AM
# RUNNING ON TMUX 62: 10:20 PM
# Done: March 7, 5:45 AM
# python train.py --run_name Test1_CLIPS_Cata-Mamba-fc_cat101_seed1 --seed 1 --architecture Cata-Mamba-fc --mamba_num_blocks 2 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id idttmdsi

# Running on tmux 62: 10:20 PM. Will need to resume
python train.py --run_name Test1_CLIPS_Cata-Mamba-fc_cat101_seed2 --seed 2 --architecture Cata-Mamba-fc --mamba_num_blocks 2 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --resume_training True --wandb_run_id 6790evnn

# RUNNING Tmux 65 - 10:20 PM
# Done: March 7, 5:45 AM
# python train.py --run_name Test1_CLIPS_Cata-Mamba-fc_CATARACTS_seed0 --seed 0 --architecture Cata-Mamba-fc --mamba_num_blocks 2 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18
# Still running on tmux 65: --wandb_run_id x9v35wqx
# python train.py --run_name Test1_CLIPS_Cata-Mamba-fc_CATARACTS_seed1 --seed 1 --architecture Cata-Mamba-fc --mamba_num_blocks 2 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18
# RUNNING on tmux 65, will need to be resumed
python train.py --run_name Test1_CLIPS_Cata-Mamba-fc_CATARACTS_seed2 --seed 2 --architecture Cata-Mamba-fc --mamba_num_blocks 2 --cnn_model resnet18 --d_state 64 --d_conv 4 --expand 3 --epochs 50 --optimizer AdamW --scheduler Cosine --weight_decay 0.0001 --learning_rate 5e-5 --weighted_loss False --label_smoothing 0.1 --clip-grad-norm True --num_clips 9 --clip_size 30 --batch_size 2 --dataset 9_CATARACTS --num_classes 18 --resume_training True --wandb_run_id facgpseo

#=================================================================================================================#
## LSTMSSSSS
# Since using ResNet18 with Mamba, should also test ResNet18 with LSTM
# March 6, 7:31 AM: Running on tmux 60 all seeds for cat101
# RUNNING: tmux 60
# DONE
# python train.py --run_name test2_seed=0_resnet18_cat101_numclips_9_clipsize_30 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 
# NEED TO RESUME: 25xm8wvr
# DONE
# python train.py --run_name test2_seed=1_resnet18_cat101_numclips_9_clipsize_30 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001









# DONE ON MY BIOMEDIA MACHINE - March 6, 9:06 PM

# python train.py --run_name test2_seed=2_resnet18_cat101_numclips_9_clipsize_30 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 
# python train.py --run_name test2_seed=0_resnet18_CATARACTS_numclips_9_clipsize_30 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# python train.py --run_name test2_seed=1_resnet18_CATARACTS_numclips_9_clipsize_30 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# python train.py --run_name test2_seed=2_resnet18_CATARACTS_numclips_9_clipsize_30 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet18 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18


# LSTMS
# March 5: 3:59 PM: seed 0 is running
# March 6 1:00 AM: seed 0 is done. Name changed from epochs=40 to epochs=50 because it ran for 50 epochs
# They are running now on tmux53
# python train.py --run_name Test1_seed=0_epochs=50_numclips_9_clipsize_30 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 

# March 6, 7:09 AM => Seed 1 will finish soon but seed 2 will need to be restarted.
# python train.py --run_name Test1_seed=1_epochs=50_numclips_9_clipsize_30 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 
# March 6, 10:20 AM, seed 2 needs to be re-run: https://wandb.ai/danaosama/CNN_RNN/runs/cu8ryx6w
# Resumed on tmux 58 - 10:30 AM
# DONE
# python train.py --run_name test1_seed=2_epochs=50_numclips_9_clipsize_30 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --resume_training True --wandb_run_id cu8ryx6w

## CATARACTS - Running on BIOMEDIA NUMAN right now (March 6, 7:09 AM)
# TODO: Check if everything is okay
# python train.py --run_name seed=0_epochs=50_numclips_9_clipsize_30 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# python train.py --run_name seed=1_epochs=50_numclips_9_clipsize_30 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# python train.py --run_name seed=2_epochs=50_numclips_9_clipsize_30 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 9 --clip_size 30 --batch_size 1 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
