import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

MODELS = {
    "vgg16": models.vgg16(weights="VGG16_Weights.DEFAULT"),
    "vgg19": models.vgg19(weights="VGG19_Weights.DEFAULT"),
    "resnet": models.resnet50(weights="ResNet50_Weights.DEFAULT"),
    "mobilenet": models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT"),
}


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
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print("[INFO] checkpoint loaded")
    return model


def train(model, optimizer, criterion, train_loader, DEVICE):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # move data to device
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.float()
        predictions = model(inputs)

        loss = criterion(predictions, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate(model, validation_loader, DEVICE):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_frames = 0

    with torch.no_grad():  # No need to track gradients during validation
        for data, labels in tqdm(validation_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            # Forward pass: compute the model output
            # Outputs are already probabilities, so we don't need to apply the softmax function
            outputs = model(data)

            # Convert probabilities to predicted class indices
            _, predicted_indices = torch.max(outputs, dim=-1)

            # Convert one-hot encoded labels to class indices for comparison
            _, labels_indices = torch.max(labels, dim=-1)

            # Calculate accuracy
            correct = (predicted_indices == labels_indices).sum().item()
            total_correct += correct
            total_frames += labels.size(0) * labels.size(
                1
            )  # Multiply batch size by sequence length

    accuracy = total_correct / total_frames
    print(f"[INFO] Validation Accuracy: {accuracy:.4f}")
    return accuracy


def validate_all_metrics(model, validation_loader, DEVICE, num_classes):
    model.eval()  # Set the model to evaluation mode
    all_predicted = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during validation
        for data, labels in tqdm(validation_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            # Forward pass: compute the model output
            outputs = model(data)
            # print("outputs.shape: ", outputs.shape)
            # print("labels.shape: ", labels.shape)
            # print()
            # print("labels:")
            # print(labels)
            # print()
            # Convert probabilities to predicted class indices
            _, predicted_indices = torch.max(outputs, dim=-1)
            all_predicted.extend(predicted_indices.cpu().numpy())

            # Convert one-hot encoded labels to class indices for comparison
            _, labels_indices = torch.max(labels, dim=-1)
            all_labels.extend(labels_indices.cpu().numpy())

            # print("predicted_indices.shape:", predicted_indices.shape)
            # print("labels_indices.shape:", labels_indices.shape)
    
    all_predicted = np.vstack(all_predicted)
    all_labels = np.vstack(all_labels)

    all_predicted = all_predicted.flatten()
    all_labels = all_labels.flatten()

    # print("all_predicted.shape:", all_predicted.shape)
    # print("all_labels.shape:", all_labels.shape)

    # Calculate metrics
    precision = precision_score(all_labels, all_predicted, average="macro")
    recall = recall_score(all_labels, all_predicted, average="macro")
    f1 = f1_score(all_labels, all_predicted, average="macro")
    accuracy = accuracy_score(all_labels, all_predicted)


    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1: {f1:.4f}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    return metrics

def test_phase_detection():
    pass
