from models.cnn_rnn import CNN_RNN_Model
from models.cnn import CNN
from models.vit import ViT
import os
import torch
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders import Cataracts_101_21_v2
from utils import *
import time

# LOGGING
import wandb

from prettytable import PrettyTable
import argparse

parser = argparse.ArgumentParser(description="MultiTask Cataracts Training Script")

# Define command-line arguments
# TODO: Load them from a config.json file later on
parser.add_argument(
    "--root_dir",
    type=str,
    help="Path containing the downloaded dataset folder Cataracts_Multitask",
    default="/l/users/dana.mohamed",
)
parser.add_argument(
    "--json_path",
    type=str,
    help="Path to the json file containing the dataset labels",
    default="/l/users/dana.mohamed/Cataracts_Multitask/labels_dataset_level.json"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to save and load the model checkpoints",
    default="/l/users/dana.mohamed/checkpoints/",
)

parser.add_argument(
    "--qualitative_results_path",
    type=str,
    help="Path to save the qualitative results, including ribbon figure and confusion matrices",
    default="qualitative_results/",
)
parser.add_argument(
    "--num_classes", type=int, help="Number of classes, default=10", default=10
)
parser.add_argument(
    "--num_clips", type=int, help="Number of clips to sample from each video", default=8
)
parser.add_argument(
    "--clip_size", type=int, help="Number of frames in each clip", default=20
)
parser.add_argument(
    "--step_size",
    type=int,
    help="Number of frames to skip when sampling clips",
    default=1,
)

parser.add_argument(
    "--learning_rate", type=float, help="Learning rate for the optimizer", default=0.001
)
parser.add_argument(
    "--epochs", type=int, help="Number of epochs for training the model", default=50
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size for training the model", default=2
)
parser.add_argument(
    "--hidden_size", type=int, help="Hidden size for the RNN", default=512
)
parser.add_argument(
    "--loss_function",
    type=str,
    help="Loss function to use for training the model",
    default="CrossEntropyLoss",
)
parser.add_argument(
    "--optimizer",
    type=str,
    help="Optimizer to use for training the model",
    default="Adam",
)

parser.add_argument(
    "--task", type=str, help="Task to train the model on", default="Phase_detection"
)
parser.add_argument(
    "--dataset",
    choices=["1_Cataracts-21", "2_Cataracts-101"],
    help="Dataset to train the model on",
    default="2_Cataracts-101",
)
parser.add_argument(
    "--architecture",
    choices=["CNN_RNN", "CNN", "ViT"],
    help="Model to use for training",
    default="CNN_RNN",
)
parser.add_argument(
    "--cnn_model",
    choices=["resnet18", "resnet50", "resnet101"],
    help="CNN model to use for training",
    default="resnet50",
)
parser.add_argument(
    "--rnn_model",
    choices=["lstm", "gru"],
    help="RNN model to use for training",
    default="lstm",
)
parser.add_argument(
    "--num_layers_rnn", type=int, help="Number of layers for the RNN", default=1
)
parser.add_argument(
    "--bidirectional",
    type=bool,
    help="Whether to use a bidirectional RNN",
    default=False,
)

parser.add_argument(
    "--resume_training",
    type=bool,
    help="Whether to resume training from a checkpoint",
    default=False,
)

parser.add_argument(
    "--wandb_run_id",
    type=str,
    help="Used when resuming training for a specific model to resume the wandb run",
    default=None,
)

# Parse the command-line arguments
args = parser.parse_args()

# Set the seed for reproducibility
torch.manual_seed(0)

# Set the device to GPU if available
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # check if NVIDIA device is visible to torch

# Parameters
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
hidden_size = args.hidden_size
bidirectional = args.bidirectional
num_layers_rnn = args.num_layers_rnn
cnn_model = args.cnn_model
rnn_model = args.rnn_model

# Directories
root_dir = args.root_dir
json_path = args.json_path
checkpoint_path = args.checkpoint_path
qualitative_results_path = args.qualitative_results_path


# Create a the directory to save the checkpoints
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    print("[INFO] created checkpoints directory")

if not os.path.exists(qualitative_results_path):
    os.makedirs(qualitative_results_path)
    print("[INFO] created qualitative results directory")

# Data related args
dataset = args.dataset
task = args.task
num_clips = args.num_clips
clip_size = args.clip_size
step_size = args.step_size


# TODO: Add more processing for those two parameters
architecture = args.architecture
architectures = {
    "CNN_RNN": CNN_RNN_Model(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers_rnn,
        cnn=cnn_model,
        rnn=rnn_model,
        bidirectional=bidirectional,
    ),
    "CNN": CNN(cnn=cnn_model, num_classes=num_classes),
    "ViT": ViT(num_classes=num_classes),
}
model = architectures[architecture]
model.to(DEVICE)

criterion = nn.CrossEntropyLoss() if args.loss_function == "CrossEntropyLoss" else None
optimizer = (
    optim.Adam(model.parameters(), lr=learning_rate)
    if args.optimizer == "Adam"
    else None
)
# ####################################################################################################################################
project_name = "Phase_detection"
run_id = None

# Add Wandb logging
if args.resume_training and args.wandb_run_id is not None:
    run_id = args.wandb_run_id
    wandb.init(project=project_name, id=run_id, resume="allow")

else: # First time training
    exp_name = (
        f"{architecture}___{cnn_model}_{rnn_model}_{num_clips}_{clip_size}"
        if architecture == "CNN_RNN"
        else f"{architecture}___{cnn_model}_{num_clips}_{clip_size}"
    )
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name=exp_name,
        # track hyperparameters and run metadata
        config=args,
    )

run_id = wandb.run.id
print(f"[INFO] Run ID: {run_id}")

# number of parameters in the model
print(
    "[INFO] number of parameters in the model: {}".format(
        sum(p.numel() for p in model.parameters())
    )
)

# Transforms
transform = v2.Compose(
    [
        v2.Resize((224, 224)),  # Example resize, adjust as needed
        # v2.ToTensor(),  # This converts PIL images to PyTorch tensors
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
    ]
)

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_val_acc = 0  # to keep track of best validation accuracy
best_val_epoch = -1
best_metrics = {}

# Capture the start time
start_time = time.time()

if args.resume_training:
    # Load the model
    ckpt = load_checkpoint(
        model, optimizer, checkpoint_path + f"last_epoch_{run_id}.pth"
    )

    start_epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"[INFO] Resuming training from epoch {start_epoch+1}")
else:
    start_epoch = 0

for epoch in range(start_epoch, epochs):
    # run training loop
    print("[INFO] starting training epoch {}".format(str(epoch + 1)))
    loss = train(model, optimizer, criterion, train_loader, DEVICE)
    wandb.log({"train_loss": loss})

    # acc = validate(model, val_loader, DEVICE)
    # TODO: Check which metric I want to use to evaluate the best model
    metrics = validate(model, val_loader, DEVICE)
    acc = metrics["accuracy"]
    wandb.log({"val_accuracy": acc})
    wandb.log({"val_precision": metrics["precision"]})
    wandb.log({"val_recall": metrics["recall"]})
    wandb.log({"val_f1_score": metrics["f1_score"]})

    print(
        f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}"
    )
    save_checkpoint(
        model, optimizer, epoch, checkpoint_path + f"last_epoch_{run_id}.pth"
    )  # save checkpoint after each epoch
    if acc > best_val_acc:
        best_val_acc = acc
        best_metrics = metrics
        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_path + f"best_model_{run_id}.pth",
            best=True,
        )
        best_val_epoch = epoch

# Capture the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time

# Log the total time taken
wandb.log({"total_time": total_time})

# Print total time taken
print(f"[INFO] Total time taken: {total_time:.2f} seconds")

# print(f"[INFO] Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch+1}")
print("[INFO] Training complete")

# Create a table with the results
print("VALIDATION SET RESULTS:")
print("#################")
results = PrettyTable()
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", best_metrics["accuracy"]])
results.add_row(["Precision", best_metrics["precision"]])
results.add_row(["Recall", best_metrics["recall"]])
results.add_row(["F1-Score", best_metrics["f1_score"]])
print(results)

table = wandb.Table(columns=["Metrics_Validation_Set", "Value"])
table.add_data("Accuracy", best_metrics["accuracy"])
table.add_data("Precision", best_metrics["precision"])
table.add_data("Recall", best_metrics["recall"])
table.add_data("F1-Score", best_metrics["f1_score"])
wandb.log({"validation_results_table": table}, commit=False)


# Test the model
# load best_model
ckpt = load_checkpoint(
    model, optimizer, checkpoint_path + f"best_model_{run_id}.pth"
)
model.load_state_dict(ckpt["model_state_dict"])

test_metrics = validate(model, test_loader, DEVICE)

# Log the test results on Wandb
table = wandb.Table(columns=["Metrics_Test_Set", "Value"])
table.add_data("Accuracy", test_metrics["accuracy"])
table.add_data("Precision", test_metrics["precision"])
table.add_data("Recall", test_metrics["recall"])
table.add_data("F1-Score", test_metrics["f1_score"])
wandb.log({"test_results_table": table}, commit=False)

columns_per_class = ["Metric"] + [f"Phase_{i}" for i in range(num_classes)]
table = wandb.Table(columns=columns_per_class)
table.add_data(*(["Jaccard"] + [str(j) for j in test_metrics["jaccard_per_class"]]))
table.add_data(*(["Precision"] + [str(p) for p in test_metrics["precision_per_class"]]))
table.add_data(*(["Recall"] + [str(r) for r in test_metrics["recall_per_class"]]))
table.add_data(*(["F1-Score"] + [str(f) for f in test_metrics["f1_per_class"]]))
wandb.log({"test_results_table_per_class": table}, commit=False)

# Print the results
# Create a table with the results
print("TEST SET RESULTS:")
print("#################")
results = PrettyTable()
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", test_metrics["accuracy"]])
results.add_row(["Precision", test_metrics["precision"]])
results.add_row(["Recall", test_metrics["recall"]])
results.add_row(["F1-Score", test_metrics["f1_score"]])
print(results)

print("Run ID: ", wandb.run.id)
print("Run URL: ", wandb.run.get_url())

wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
wandb.run.summary["test_precision"] = test_metrics["precision"]
wandb.run.summary["test_recall"] = test_metrics["recall"]
wandb.run.summary["test_f1_score"] = test_metrics["f1_score"]

# Save confusion matrix
create_confusion_matrix = True
if create_confusion_matrix:
    cf_matrix = test_metrics["confusion_matrix"]
    confusion_path = os.path.join(qualitative_results_path, f"confusion_matrix_{run_id}.png")
    
    save_confusion_matrix(cf_matrix, confusion_path)
    wandb.log({"confusion_matrix": wandb.Image(confusion_path)})

# Save qualitative results
create_qualitative_results = True
if create_qualitative_results:
    qualitative_results_path = os.path.join(qualitative_results_path, f"ribbon_diagram_{run_id}.png")
    batch = next(iter(test_loader))
    make_qualitative_results(
        {f"{architecture}": model},
        next(iter(test_loader)),
        qualitative_results_path,
        num_classes,
        DEVICE,
    )
    wandb.log({"qualitative_results": wandb.Image(qualitative_results_path)})

wandb.finish()