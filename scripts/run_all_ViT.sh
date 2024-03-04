

python train.py --run_name test --seed 0 --architecture ViT --epochs 1 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False


python train.py --run_name 101-seed0 --seed 0 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False
python train.py --run_name 101-seed1 --seed 1 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False
python train.py --run_name 101-seed2 --seed 2 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False

python train.py --run_name CATARACTS-seed0 --seed 0 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
python train.py --run_name CATARACTS-seed1 --seed 1 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
python train.py --run_name CATARACTS-seed2 --seed 2 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18

python train.py --run_name 101-seed3 --seed 3 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False
python train.py --run_name 101-seed4 --seed 4 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False

python train.py --run_name CATARACTS-seed3 --seed 3 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
python train.py --run_name CATARACTS-seed4 --seed 4 --architecture ViT --epochs 50 --optimizer AdamW --num_clips 8 --clip_size 18 --step_size 2 --weight_decay 0.3 --learning_rate 5e-5 --label_smoothing 0.11 --scheduler Cosine --clip-grad-norm False --dataset 9_CATARACTS --num_classes 18
