from models.cnn_rnn import CNN_RNN_Model
import os
import torch
from torchvision.transforms import v2
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders import Cataracts_101_21_v2
from utils import * 
import random


# LOGGING
import wandb

import logging
import sys
from logging import StreamHandler, Formatter, FileHandler
from prettytable import PrettyTable
import argparse

parser = argparse.ArgumentParser(description="MultiTask Cataracts Training Script")

# Define command-line arguments
# TODO: Load them from a config.json file later on
# TODO: Arrange the arguments in a better way
parser.add_argument("--num_classes", type=int, help="Number of classes, default=10", default=10)
parser.add_argument("--root_dir", type=str, help="Path containing the downloaded dataset folder Cataracts_Multitask", default="/l/users/dana.mohamed")
parser.add_argument("--json_path", type=str, help="Path to the json file containing the dataset labels", default="/home/dana.mohamed/MultiTask_Video_Data_Preprocessing/2_NEW_dataset_level_labels.json")
parser.add_argument("--checkpoint_path", type=str, help="Path to save and load the model checkpoints", default="/l/users/dana.mohamed/checkpoints/")
parser.add_argument("--num_clips", type=int, help="Number of clips to sample from each video", default=2)
parser.add_argument("--clip_size", type=int, help="Number of frames in each clip", default=20)
parser.add_argument("--step_size", type=int, help="Number of frames to skip when sampling clips", default=1)
parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer", default=0.001)
parser.add_argument("--epochs", type=int, help="Number of epochs for training the model", default=1)
parser.add_argument("--batch_size", type=int, help="Batch size for training the model", default=1)
parser.add_argument("--hidden_size", type=int, help="Hidden size for the RNN", default=512)
parser.add_argument("--loss_function", type=str, help="Loss function to use for training the model", default="CrossEntropyLoss")
parser.add_argument("--optimizer", type=str, help="Optimizer to use for training the model", default="Adam")
parser.add_argument("--task", type=str, help="Task to train the model on", default="Phase_detection")
parser.add_argument("--dataset", type=str, help="Dataset to train the model on", default="2_Cataracts-101")
parser.add_argument("--model", type=str, help="Model to use for training", default="CNN_RNN")
parser.add_argument("--bidirectional", type=bool, help="Whether to use a bidirectional RNN", default=False)

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
hidden_size = args.hidden_size
bidirectional = args.bidirectional

## Directories
root_dir = args.root_dir
json_path = args.json_path
checkpoint_path = args.checkpoint_path

# Data related args
dataset = args.dataset
task = args.task
num_clips = args.num_clips
clip_size = args.clip_size
step_size = args.step_size


# TODO: Add more processing for those two parameters
model_name = args.model
model = CNN_RNN_Model(num_classes=num_classes, hidden_size=hidden_size, num_clips=num_clips, num_layers=1, bidirectional=bidirectional) if model_name == "CNN_RNN" else None
model.to(DEVICE)
criterion = nn.CrossEntropyLoss() if args.loss_function == "CrossEntropyLoss" else None
optimizer = optim.Adam(model.parameters(), lr = learning_rate) if args.optimizer == "Adam" else None
# ####################################################################################################################################

# Add Wandb logging
# start a new wandb run to track this script
random_int = random.randint(0, 1000)
wandb.init(
    # set the wandb project where this run will be logged
    project="Thesis",
    name = f"{task}_{model_name}_{dataset}_{random_int}",
    # track hyperparameters and run metadata
    config={
        "dataset_name": dataset,
        "task": task,
        "model": model_name,
        "num_classes": num_classes,
        "num_clips": num_clips,
        "clip_size": clip_size,
        "step_size": step_size,
        "root_dir": root_dir,
        "json_path": json_path,
        "checkpoint_path": checkpoint_path,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "bidirectional": bidirectional,
        "criterion": args.loss_function,
        "optimizer": args.optimizer
    }
)

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
    num_clips=num_clips,
    clip_size=clip_size,
    step_size=step_size,
    transform=transform,
)

val_dataset = Cataracts_101_21_v2(
    root_dir,
    json_path,
    # dataset_name="1_Cataracts-21",
    dataset_name="2_Cataracts-101",
    split="Validation",
    num_classes=num_classes,
    num_clips=num_clips,
    clip_size=clip_size,
    step_size=step_size,
    transform=transform,
)

test_dataset = Cataracts_101_21_v2(
    root_dir,
    json_path,
    # dataset_name="1_Cataracts-21",
    dataset_name="2_Cataracts-101",
    split="Test",
    num_classes=num_classes,
    num_clips=num_clips,
    clip_size=clip_size,
    step_size=step_size,
    transform=transform,
)

# Create a dataloader
if num_clips == -1:
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

best_val_acc = 0 # to keep track of best validation accuracy
best_val_epoch = -1
best_metrics = {}

for epoch in range(epochs):
    # run training loop
    print("[INFO] starting training epoch {}".format(str(epoch+1)))
    loss = train(model, optimizer, criterion, train_loader, DEVICE)
    wandb.log({"train_loss": loss})

    # acc = validate(model, val_loader, DEVICE)
    # TODO: Check which metric I want to use to evaluate the best model
    metrics = validate(model, val_loader, DEVICE)
    acc = metrics['accuracy']
    wandb.log({"val_accuracy": acc})
    wandb.log({"val_precision": metrics['precision']})
    wandb.log({"val_recall": metrics['recall']})
    wandb.log({"val_f1_score": metrics['f1_score']})
    
    print(f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}")
    save_checkpoint(model, optimizer, epoch, checkpoint_path + f"last_epoch_{random_int}.pth") # save checkpoint after each epoch
    if(acc > best_val_acc):
        best_val_acc = acc
        best_metrics = metrics
        save_checkpoint(model, optimizer, epoch, checkpoint_path + f"best_model_{random_int}.pth", best = True)
        best_val_epoch = epoch

# print(f"[INFO] Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch+1}")
print("[INFO] Training complete")

# Create a table with the results
print("VALIDATION SET RESULTS:")
print("#################")
results = PrettyTable()
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", best_metrics['accuracy']])
results.add_row(["Precision", best_metrics['precision']])
results.add_row(["Recall", best_metrics['recall']])
results.add_row(["F1-Score", best_metrics['f1_score']])
print(results)

table = wandb.Table(columns=["Metrics_Validation_Set", "Value"])
table.add_data("Accuracy", best_metrics['accuracy'])
table.add_data("Precision", best_metrics['precision'])
table.add_data("Recall", best_metrics['recall'])
table.add_data("F1-Score", best_metrics['f1_score'])
wandb.log({"results_table":table}, commit=False)



# Test the model
# load best_model
model = load_checkpoint(model, optimizer, checkpoint_path + f"best_model_{random_int}.pth")

test_metrics = validate(model, test_loader, DEVICE)

table = wandb.Table(columns=["Metrics_Test_Set", "Value"])
table.add_data("Accuracy", test_metrics['accuracy'])
table.add_data("Precision", test_metrics['precision'])
table.add_data("Recall", test_metrics['recall'])
table.add_data("F1-Score", test_metrics['f1_score'])
wandb.log({"test_results_table":table}, commit=False)

# Create a table with the results
print("TEST SET RESULTS:")
print("#################")
results = PrettyTable()
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", test_metrics['accuracy']])
results.add_row(["Precision", test_metrics['precision']])
results.add_row(["Recall", test_metrics['recall']])
results.add_row(["F1-Score", test_metrics['f1_score']])
print(results)

print("Run ID: ", wandb.run.id)
print("Run URL: ", wandb.run.get_url())
print(f"Random int: {random_int}")
wandb.finish()