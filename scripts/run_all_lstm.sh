
# python train.py --run_name test --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 1 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine --dataset 9_CATARACTS --num_classes 18


python train.py --run_name 101-seed0 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine
python train.py --run_name 101-seed1 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine
python train.py --run_name 101-seed2 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine


python train.py --run_name CATARACTS-seed0 --seed 0 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine --dataset 9_CATARACTS --num_classes 18
python train.py --run_name CATARACTS-seed1 --seed 1 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine --dataset 9_CATARACTS --num_classes 18
python train.py --run_name CATARACTS-seed2 --seed 2 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine --dataset 9_CATARACTS --num_classes 18

python train.py --run_name 101-seed3 --seed 3 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine
python train.py --run_name 101-seed4 --seed 4 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine

python train.py --run_name CATARACTS-seed3 --seed 3 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine --dataset 9_CATARACTS --num_classes 18
python train.py --run_name CATARACTS-seed4 --seed 4 --architecture CNN_RNN --rnn_model lstm --cnn_model resnet50 --hidden_size 256 --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 20 --step_size 2 --learning_rate 0.001 --label_smoothing 0.11 --scheduler Cosine --dataset 9_CATARACTS --num_classes 18
