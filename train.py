from models.cnn_rnn import CNN_RNN_Model
import os
import torch
from torchvision.transforms import v2
import torch.nn as nn # basic building block for neural networks
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloaders import Cataracts_101_21_v2
from utils import * 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # check if NVIDIA device is visible to torch

# TODO: Add command line arguments
model = CNN_RNN_Model(num_classes=10, hidden_size=512, num_layers=1, bidirectional=False)
model.to(DEVICE)

learning_rate = 0.001
epochs = 5 # what is an epoch?
batch_size = 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate) # what's an optimizer?
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

# TODO: Transform those to command line arguments
root_dir = "/l/users/dana.mohamed"
json_path = "/home/dana.mohamed/MultiTask_Video_Data_Preprocessing/2_NEW_dataset_level_labels.json"
checkpoint_path = "/home/dana.mohamed/Checkpoints/"

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
    num_clips=2,
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
    num_clips=2,
    clip_size=16,
    step_size=2,
    transform=transform,
)

# Create a dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

best_val_acc = 0 # to keep track of best validation accuracy

for epoch in range(epochs):
    # run training loop
    print("[INFO] starting training epoch {}".format(str(epoch+1)))
    loss = train(model, optimizer, criterion, train_loader, DEVICE)
    acc = validate(model, val_loader, DEVICE)
    print(f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}")
    save_checkpoint(model, optimizer, epoch, "checkpoints/last_epoch.pth") # save checkpoint after each epoch
    if(acc > best_val_acc):
        best_val_acc = acc
        save_checkpoint(model, optimizer, epoch, "checkpoints/best_model.pth", best = True)