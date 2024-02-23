import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)


def save_checkpoint(model, optimizer, epoch, path, best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    if best:
        print("[INFO] best checkpoint saved at epoch {}".format(epoch))
    else:
        print("[INFO] checkpoint saved at epoch {}".format(epoch))


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    # model.load_state_dict(ckpt["model_state_dict"])
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # epoch = ckpt["epoch"]
    print("[INFO] checkpoint loaded")
    return ckpt


def train(model, optimizer, criterion, train_loader, DEVICE):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    num_frames = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # move data to device
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.float()
        predictions = model(inputs)
        num_frames += predictions.size(1)

        loss = criterion(predictions, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # return running_loss / len(train_loader)
    return running_loss / num_frames


def validate(model, validation_loader, DEVICE):
    model.eval()  # Set the model to evaluation mode
    all_predicted = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during validation
        for data, labels in tqdm(validation_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            # Forward pass: compute the model output
            outputs = model(data)
            _, predicted_indices = torch.max(outputs, 2)

            all_predicted.extend(predicted_indices.cpu().numpy())

            # Convert one-hot encoded labels to class indices for comparison
            _, labels_indices = torch.max(labels, dim=-1)
            all_labels.extend(labels_indices.cpu().numpy())

    all_predicted = np.concatenate(all_predicted)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    precision = precision_score(
        all_labels, all_predicted, zero_division=0, average="macro"
    )
    recall = recall_score(all_labels, all_predicted, zero_division=0, average="macro")
    f1 = f1_score(all_labels, all_predicted, zero_division=0, average="macro")
    accuracy = accuracy_score(all_labels, all_predicted)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics
