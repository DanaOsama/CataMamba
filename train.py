from models.cnn_rnn import CNN_RNN_Model
import os
import torch
from torchvision.transforms import v2
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders import Cataracts_101_21_v2
from utils import * 



# LOGGING
import wandb

import logging
import sys
from logging import StreamHandler, Formatter, FileHandler
from prettytable import PrettyTable
import argparse

parser = argparse.ArgumentParser(description="MultiTask Cataracts Training Script")

# Define command-line arguments
parser.add_argument("--num_classes", type=int, help="Number of classes, default=10", default=10)
parser.add_argument("--root_dir", type=str, help="Path containing the downloaded dataset folder Cataracts_Multitask", default="/l/users/dana.mohamed")
parser.add_argument("--json_path", type=str, help="Path to the json file containing the dataset labels", default="/home/dana.mohamed/MultiTask_Video_Data_Preprocessing/2_NEW_dataset_level_labels.json")
parser.add_argument("--checkpoint_path", type=str, help="Path to save and load the model checkpoints", default="/l/users/dana.mohamed/checkpoints/")
parser.add_argument("--num_clips", type=int, help="Number of clips to sample from each video", default=4)
parser.add_argument("--clip_size", type=int, help="Number of frames in each clip", default=16)
parser.add_argument("--step_size", type=int, help="Number of frames to skip when sampling clips", default=2)
parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer", default=0.001)
parser.add_argument("--epochs", type=int, help="Number of epochs for training the model", default=10)
parser.add_argument("--batch_size", type=int, help="Batch size for training the model", default=2)
parser.add_argument("--hidden_size", type=int, help="Hidden size for the RNN", default=512)
parser.add_argument("--loss_function", type=str, help="Loss function to use for training the model", default="CrossEntropyLoss")
parser.add_argument("--optimizer", type=str, help="Optimizer to use for training the model", default="Adam")
parser.add_argument("--task", type=str, help="Task to train the model on", default="Phase_detection")
parser.add_argument("--dataset", type=str, help="Dataset to train the model on", default="2_Cataracts-101")
parser.add_argument("--model", type=str, help="Model to use for training", default="CNN_RNN")

# Parse the command-line arguments
args = parser.parse_args()

# Set the seed for reproducibility
torch.manual_seed(0)

# Set the device to GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # check if NVIDIA device is visible to torch

# Parameters
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes

## Directories
root_dir = args.root_dir
json_path = args.json_path
checkpoint_path = args.checkpoint_path


# TODO: Add more processing for those two parameters
model = CNN_RNN_Model(num_classes=10, hidden_size=512, num_layers=1, bidirectional=False)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss() if args.loss_function == "CrossEntropyLoss" else None
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# #################################

# number of parameters in the model
print("[INFO] number of parameters in the model: {}".format(sum(p.numel() for p in model.parameters())))

# Transforms
transform = v2.Compose(
        [
            v2.Resize((224, 224)),  # Example resize, adjust as needed
            # v2.ToTensor(),  # This converts PIL images to PyTorch tensors
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
        ]
    )


# Create a the directory to save the checkpoints
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    print("[INFO] created checkpoints directory")

train_dataset = Cataracts_101_21_v2(
    root_dir,
    json_path,
    # dataset_name="1_Cataracts-21",
    dataset_name="2_Cataracts-101",
    split="Train",
    num_classes=num_classes,
    num_clips=4,
    clip_size=16,
    step_size=2,
    transform=transform,
)

val_dataset = Cataracts_101_21_v2(
    root_dir,
    json_path,
    # dataset_name="1_Cataracts-21",
    dataset_name="2_Cataracts-101",
    split="Validation",
    num_classes=num_classes,
    num_clips=2,
    clip_size=16,
    step_size=2,
    transform=transform,
)

# Create a dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

best_val_acc = 0 # to keep track of best validation accuracy
best_val_epoch = -1
best_metrics = {}

for epoch in range(epochs):
    # run training loop
    print("[INFO] starting training epoch {}".format(str(epoch+1)))
    loss = train(model, optimizer, criterion, train_loader, DEVICE)
    # acc = validate(model, val_loader, DEVICE)

    # TODO: Check which metric I want to use to evaluate the best model
    metrics = validate_all_metrics(model, val_loader, DEVICE, num_classes=num_classes)
    acc = metrics['accuracy']

    print(f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}")
    save_checkpoint(model, optimizer, epoch, checkpoint_path + "last_epoch.pth") # save checkpoint after each epoch
    if(acc > best_val_acc):
        best_val_acc = acc
        best_metrics = metrics
        save_checkpoint(model, optimizer, epoch, checkpoint_path + "best_model.pth", best = True)
        best_val_epoch = epoch

# print(f"[INFO] Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch+1}")
print("[INFO] Training complete")
print(f"[INFO] Best validation accuracy: {metrics['accuracy']}")
print(f"[INFO] Best validation precision: {metrics['precision']}")
print(f"[INFO] Best validation recall: {metrics['recall']}")
print(f"[INFO] Best validation f1-score: {metrics['f1_score']}")
print(f"[INFO] Best validation auc_roc: {metrics['auc_roc']}")
