
# python train.py --run_name 101-seed0 --seed 0 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 
# python train.py --run_name 101-seed1 --seed 1 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 
# python train.py --run_name 101-seed2 --seed 2 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 
# # python train.py --run_name CATARACTS-seed0 --seed 0 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# # python train.py --run_name CATARACTS-seed1 --seed 1 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# # python train.py --run_name CATARACTS-seed2 --seed 2 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18

# python train.py --run_name 101-seed3 --seed 3 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001
# python train.py --run_name 101-seed4 --seed 4 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001
python train.py --run_name AdamW_B=2_mai_CATARACTS-seed0 --seed 0 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18 --batch_size 2 
python train.py --run_name AdamW_B=2_mai_CATARACTS-seed1 --seed 1 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18 --batch_size 2 
python train.py --run_name AdamW_B=2_mai_CATARACTS-seed2 --seed 2 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18 --batch_size 2 

# # python train.py --run_name CATARACTS-seed3 --seed 3 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
# python train.py --run_name CATARACTS-seed4 --seed 4 --architecture CNN --cnn_model resnet50 --epochs 50 --optimizer Adam --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --dataset 9_CATARACTS --num_classes 18
