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
    jaccard_score,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
import matplotlib as mpl

def save_checkpoint(model, optimizer, epoch, path, scheduler=None, best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)
    if best:
        print("[INFO] best checkpoint saved at epoch {}".format(epoch))
    else:
        print("[INFO] checkpoint saved at epoch {}".format(epoch))


def load_checkpoint(path):
    ckpt = torch.load(path)
    # model.load_state_dict(ckpt["model_state_dict"])
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # epoch = ckpt["epoch"]
    print("[INFO] checkpoint loaded")
    return ckpt


def train(model, optimizer, criterion, train_loader, DEVICE):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_frames = 0
    losses = []

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # move data to device
        optimizer.zero_grad()  # Zero the gradients
        predictions = model(inputs)
        breakpoint()
        batch_size, num_frames, num_classes = predictions.size()
        predictions = predictions.reshape(batch_size * num_frames, num_classes)
        labels = torch.argmax(labels, dim=-1)
        labels = labels.view(-1)
        total_frames += num_frames*batch_size

        # Expected input shape: (batch_size * num_frames, num_classes)
        # Expected label shape: (batch_size * num_frames) where each label is a class index
        loss = criterion(predictions, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # return running_loss / len(train_loader)
    return sum(losses) / len(losses)


def validate(model, validation_loader, DEVICE, per_class_metrics=True):
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
        all_labels, all_predicted, zero_division=0, average="micro"
    )
    recall = recall_score(all_labels, all_predicted, zero_division=0, average="micro")
    f1 = f1_score(all_labels, all_predicted, zero_division=0, average="micro")
    accuracy = accuracy_score(all_labels, all_predicted)
    jaccard = jaccard_score(all_labels, all_predicted, average="micro")
    cf_matrix = confusion_matrix(all_labels, all_predicted)
    
    # Per class metrics, where the averaging is set to None.
    precision_per_class = precision_score(all_labels, all_predicted, zero_division=1.0, average=None)
    recall_per_class = recall_score(all_labels, all_predicted, zero_division=1.0 , average=None)
    f1_per_class = f1_score(all_labels, all_predicted, zero_division=1.0, average=None)
    jaccard_per_class = jaccard_score(all_labels, all_predicted, average=None, zero_division=1.0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "jaccard": jaccard,
        "confusion_matrix": cf_matrix,
    }
    if per_class_metrics:
        metrics["precision_per_class"] = precision_per_class
        metrics["recall_per_class"] = recall_per_class
        metrics["f1_per_class"] = f1_per_class
        metrics["jaccard_per_class"] = jaccard_per_class

    return metrics

def save_confusion_matrix(cf_matrix, path):
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.heatmap(cf_matrix, annot=True)
    # Add labels and title
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    plt.title('Confusion Matrix')

    # Add tick labels (e.g., class names)
    num_classes = cf_matrix.shape[0]
    class_names = [f"P{i}" for i in range(num_classes)]

    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    plt.savefig(path, bbox_inches='tight', dpi=300)

def plot_results(predictions, title, position, total_plots, num_classes, cmap=None):
    """
    plot_results: Function to plot the results of the predictions
    :param predictions: Predictions (np.array) (num_frames,)
    :param title: Title of the plot (str)
    :param position: Position of the plot (int)
    :param total_plots: Total number of plots (int)
    """


    plt.subplot(total_plots, 1, position)
    norm = Normalize(vmin=0, vmax=num_classes-1)
    plt.imshow(predictions[np.newaxis, :], cmap=cmap, aspect="auto", norm=norm)
    plt.gca().set_yticks([])
    plt.gca().set_xticks(np.arange(0, len(predictions), 10))
    plt.ylabel(title)

def make_qualitative_results(models_dic, batch, path, num_classes, DEVICE):
    """
    Function to plot the qualitative results of the predictions
    :param models_dic: Dictionary of models (dict) {"model_name": model}
    :param batch: Batch of data (tuple) (inputs, labels)
        inputs: (batch_size, num_frames, channels, height, width)
        labels: (batch_size, num_frames, num_classes)
        batch_size should be 1
    :param path: Path to save the plot (str)
    :param num_classes: Number of classes (int)
    :param DEVICE: Device to run the models (str)
    """
    num_of_models = len(models_dic)
    if num_classes > 10:
        custom_colors = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a',
                         '#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94',
                         '#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d',
                         '#17becf','#9edae5']

    else:
        custom_colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
                    '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
    
    cmap = ListedColormap(custom_colors)

    # Load the data
    inputs, labels = batch
    if inputs.shape[0] != 1:
        print("[INFO] Batch size is greater than 1, taking the first sample only")
        inputs = inputs[0][np.newaxis, ...]
        labels = labels[0][np.newaxis, ...]

    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    # Get the ground truth
    ground_truth = torch.argmax(labels, dim=-1).squeeze().cpu().numpy() # (batch_size, num_frames) if batch_size is not 1, else (num_frames,)

    # Create a dictionary to store the predictions
    predictions = {}
    
    for model in models_dic.keys():
        models_dic[model].eval()
        models_dic[model] = models_dic[model].to(DEVICE)
        model_predictions = models_dic[model](inputs).cpu().detach().numpy()
        predictions[model] = np.argmax(model_predictions, axis=-1).squeeze() # (batch_size, num_frames) if batch_size is not 1, else (num_frames,)

    # Plotting
    plt.figure(figsize=(12, 6))

    # First, plot the ground truth
    plot_results(ground_truth, "Ground Truth", 1, num_of_models, num_classes, cmap)

    # Plot the predictions
    for i, model_name in enumerate(models_dic.keys()):
        # def plot_results(predictions, title, position, total_plots, num_classes, cmap=None):
        plot_results(predictions[model_name], model_name, i+2, num_of_models + 1, num_classes, cmap)

    # Create a legend
    handles = [mpatches.Patch(color=custom_colors[i], label=f'Phase {i}') for i in range(num_classes)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Final adjustments
    plt.xticks(np.arange(0, len(ground_truth), 10), np.arange(0, len(ground_truth), 10))
    plt.xlabel("Frame Number")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=300)