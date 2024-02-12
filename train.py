from models.cnn_rnn import CNN_RNN_Model
import os
import torch
from torchvision.transforms import v2
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders import Cataracts_101_21_v2
from utils import * 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # check if NVIDIA device is visible to torch

# TODO: Add command line arguments
model = CNN_RNN_Model(num_classes=10, hidden_size=512, num_layers=1, bidirectional=False)
model.to(DEVICE)

# Parameters
learning_rate = 0.001
epochs = 10
batch_size = 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
num_classes = 10

# Directories
root_dir = "/l/users/dana.mohamed"
json_path = "/home/dana.mohamed/MultiTask_Video_Data_Preprocessing/2_NEW_dataset_level_labels.json"
checkpoint_path = "/l/users/dana.mohamed/checkpoints/"
# #################################

# number of parameters in the model
print("[INFO] number of parameters in the model: {}".format(sum(p.numel() for p in model.parameters())))

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
    acc = validate(model, val_loader, DEVICE)

    # TODO: Check which metric I want to use to evaluate the best model
    # metrics = validate_all_metrics(model, val_loader, DEVICE, num_classes=num_classes)
    # acc = metrics['accuracy']

    print(f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}")
    save_checkpoint(model, optimizer, epoch, checkpoint_path + "last_epoch.pth") # save checkpoint after each epoch
    if(acc > best_val_acc):
        best_val_acc = acc
        # best_metrics = metrics
        save_checkpoint(model, optimizer, epoch, checkpoint_path + "best_model.pth", best = True)
        best_val_epoch = epoch

print(f"[INFO] Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch+1}")
print("[INFO] Training complete")
# print(f"[INFO] Best validation accuracy: {metrics['accuracy']}")
# print(f"[INFO] Best validation precision: {metrics['precision']}")
# print(f"[INFO] Best validation recall: {metrics['recall']}")
# print(f"[INFO] Best validation f1-score: {metrics['f1_score']}")
# print(f"[INFO] Best validation auc_roc: {metrics['auc_roc']}")
