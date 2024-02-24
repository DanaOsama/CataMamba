from dataloaders import Cataracts_101_21, Cataracts_101_21_v2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import json
from torchvision.transforms import v2

root_dir = "/l/users/dana.mohamed"
json_path = "/home/dana.mohamed/MultiTask_Video_Data_Preprocessing/2_NEW_dataset_level_labels.json"

#
import torch


def sample_random_images(data_loader, num_samples=16):
    images = []
    sampled = 0

    for batch in data_loader:
        # With batch size of 1, batch[0] is a list containing a single list of images for the video
        # We access this list with [0] and then another [0] to get the list of images
        batch_images = batch[0][
            0
        ]  # Adjust this if your DataLoader structure is different

        # Flatten the list if your DataLoader returns a nested list even with batch_size=1
        # If batch_images is already a flat list of tensors, you can skip this step

        # Randomly sample images from this batch
        # Ensure we don't sample more images than available or more than we need
        num_to_sample = min(num_samples - sampled, len(batch_images))
        if num_to_sample <= 0:
            break  # Break if we've already sampled enough images

        indices = torch.randperm(len(batch_images))[:num_to_sample]
        for idx in indices:
            images.append(batch_images[idx])

        sampled += num_to_sample
        if sampled >= num_samples:
            break

    # Since images can be of varying sizes (due to different datasets), we don't stack them here
    # If you need to stack or process them further, ensure they are all the same size
    return images


def save_image_grid(images, filename="sample_grid.png", nrow=4, normalize=True):
    """
    Save a grid of images to a file.

    Args:
    - images (Tensor): A tensor of images of shape (B, C, H, W).
    - filename (str): Filename to save the image grid.
    - nrow (int): Number of images in each row of the grid.
    - normalize (bool): Whether to normalize the images to [0, 1] before saving.
    """
    grid = make_grid(images, nrow=nrow, normalize=normalize)
    save_image(grid, filename)


dataset_class_version = 2

if dataset_class_version == 1:

    # Opening JSON file
    f = open(json_path)

    # returns JSON object as a dictionary
    data = json.load(f)
    # data = data['Train']['2_Cataracts-101']
    data = data["Train"]["1_Cataracts-21"]

    # Transforms list
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Example resize, adjust as needed
    #     # transforms.ToTensor(),  # This converts PIL images to PyTorch tensors
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
    # ])

    transform = v2.Compose(
        [
            v2.Resize((224, 224)),  # Example resize, adjust as needed
            # v2.ToTensor(),  # This converts PIL images to PyTorch tensors
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
        ]
    )

    # Create a dataset
    train_dataset = Cataracts_101_21(
        data_list=data, root_dir=root_dir, transform=transform
    )

    # Create a dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("Length of train_loader: ", len(train_loader))

    # Assuming you have a DataLoader named `data_loader`
    random_images = sample_random_images(train_loader, num_samples=16)
    save_image_grid(random_images, filename="random_samples_21.png", nrow=4)

    # Close the JSON file
    f.close()

elif dataset_class_version == 2:
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
        num_clips=2,
        clip_size=16,
        step_size=2,
        transform=transform,
    )

    # Create a dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("Length of train_loader: ", len(train_loader))

    # Assuming you have a DataLoader named `data_loader`
    random_images = sample_random_images(train_loader, num_samples=16)
    save_image_grid(random_images, filename="random_samples_v2_101.png", nrow=4)