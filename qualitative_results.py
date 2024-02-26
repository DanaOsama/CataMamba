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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib as mpl

import argparse

parser = argparse.ArgumentParser(description="MultiTask Cataracts Qualitative Results Script")

# Create a custom colormap


custom_colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
                 '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']

# Lighter_colors
# custom_colors = ['#fbf8cc', '#fde4cf', '#ffcfd2', '#f1c0e8', '#cfbaf0', '#a3c4f3', '#90dbf4',
#                  '#8eecf5', '#98f5e1', '#b9fbc0']

cmap = ListedColormap(custom_colors)


def plot_results(predictions, title, position, total_plots, num_classes, cmap=None):
    """
    plot_results: Function to plot the results of the predictions
    :param predictions: Predictions (np.array) (num_frames,)
    :param title: Title of the plot (str)
    :param position: Position of the plot (int)
    :param total_plots: Total number of plots (int)
    :param num_classes: Number of classes (int)
    """
    plt.subplot(total_plots, 1, position)
    plt.imshow(predictions[np.newaxis, :], cmap=cmap, aspect="auto")
    plt.gca().set_yticks([])
    plt.gca().set_xticks(np.arange(0, len(predictions), 10))
    plt.ylabel(title)

def make_qualitative_results(models_dic, batch, path, cmap, num_classes, DEVICE):
    """
    Function to plot the qualitative results of the predictions
    :param models_dic: Dictionary of models (dict) {"model_name": model}
    :param batch: Batch of data (tuple) (inputs, labels)
        inputs: (batch_size, num_frames, channels, height, width)
        labels: (batch_size, num_frames, num_classes)
        batch_size should be 1
    :param path: Path to save the plot (str)
    :param cmap: Custom colormap (ListedColormap)
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
        if cmap is None:
            custom_colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
                        '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
    
    cmap = ListedColormap(custom_colors)

    # Load the data
    inputs, labels = batch
    if inputs.shape[0] != 1:
        raise ValueError("Batch size should be 1")

    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    # Get the ground truth
    ground_truth = torch.argmax(labels, dim=-1).cpu().numpy()

    # Create a dictionary to store the predictions
    predictions = {}
    
    for model in models_dic.keys():
        predictions = models_dic[model](inputs).cpu().detach().numpy()
        predictions[model] = np.argmax(predictions, axis=-1).squeeze() # (batch_size, num_frames) if batch_size is not 1, else (num_frames,)

    # Plotting
    plt.figure(figsize=(12, 6))

    # First, plot the ground truth
    plot_results(ground_truth, "Ground Truth", 1, num_of_models, cmap)

    # Plot the predictions
    for i, model_name in enumerate(models_dic.keys()):
        plot_results(predictions[model_name], model_name, i+2, plot_results(ground_truth, "Ground Truth", 1, num_of_models)
)

    # Create a legend
    handles = [mpatches.Patch(color=custom_colors[i], label=f'Phase {i+1}') for i in range(len(custom_colors))]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Final adjustments
    plt.xticks(np.arange(0, len(ground_truth), 10), np.arange(0, len(ground_truth), 10))
    plt.xlabel("Frame Number")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=300)