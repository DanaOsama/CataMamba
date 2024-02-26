import torch
import matplotlib.pyplot as plt

# import ignite
import tempfile
import sys
import shutil
import os
import logging
import json
from monai.data import CacheDataset, DataLoader
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# from torchvision import transforms
from torchvision.transforms import v2
import re

# Set numpy random seed for reproducibility
np.random.seed(0)

def extract_frame_number(filename):
    # The regex pattern \d+ matches one or more digits
    match = re.search(r"\d+", filename)
    if match:
        return (
            match.group()
        )  # This will return the first occurrence of one or more digits in the string
    else:
        return None  # Or raise an error/exception if preferred



class Cataracts_101_21_v2(Dataset):
    def __init__(
        self,
        root_dir,
        json_path,
        dataset_name,
        split,
        num_classes=10,
        num_clips=2,
        clip_size=16,
        step_size=2,
        transform=None,
    ):
        """
        Parameters:
        - root_dir (string): Directory containing the downloaded and extracted dataset zip file.
        - json_path (string): Path to the JSON file with video paths and labels.
        - dataset_name (string): Name of the dataset to load. Either: "1_Cataracts-21" or "2_Cataracts-101".
        - split (string): Split of the dataset to load. Either: "Train", "Validation", or "Test".
        - num_classes (int): Number of classes in the dataset.
        - num_clips (int): Number of clips to sample from each video.
        - clip_size (int): Number of frames in each clip.
        - step_size (int): Number of frames to skip when sampling clips.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        if dataset_name == "9_CATARACTS":
            if split in ['Train', 'Validation']:
                dataset_name = "9_CATARACTS-A"
            else:
                dataset_name = "9_CATARACTS-B"

        self.root_dir = root_dir
        self.json_path = json_path
        self.dataset_name = dataset_name
        self.split = split
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.clip_size = clip_size
        self.step_size = step_size
        self.transform = transform

        self.data = self._load_data()

        video_ids = []
        for d in self.data:
            video_ids.append(int(d['Video_ID']))

        # Loading some extra information for the Cataracts-101 dataset
        if self.dataset_name == "2_Cataracts-101":
            # Loading the csv file for label-to-class mapping
            path = os.path.join(root_dir, "Cataracts_Multitask/2_Cataracts-101/phases.csv")
            label_to_class = pd.read_csv(path)
            self.label_to_class = dict(zip(label_to_class['Phase'], label_to_class['Meaning']))

            # Load some extra information about the videos for Cataracts-101
            extra_info_csv = "Cataracts_Multitask/2_Cataracts-101/videos.csv"
            path = os.path.join(root_dir, extra_info_csv)
            video_extra_info = pd.read_csv(path)
            video_extra_info = video_extra_info[video_extra_info['VideoID'].isin(video_ids)]

            # Sort the videos by VideoID
            video_extra_info.sort_values(by="VideoID", inplace=True)

            self.surgeon_ids = video_extra_info["Surgeon"].values
            self.surgeon_experience = video_extra_info["Experience"].values
        
        elif self.dataset_name == "1_Cataracts-21":
            # Loading the csv file for label-to-class mapping
            path = os.path.join(root_dir, "Cataracts_Multitask/1_Cataracts-21/phases.csv")
            label_to_class = pd.read_csv(path)
            self.label_to_class = dict(zip(label_to_class['Phase'], label_to_class['Meaning']))

        elif "9_CATARACTS" in self.dataset_name:
            # Loading the csv file for label-to-class mapping
            path = os.path.join(root_dir, "Cataracts_Multitask/", self.dataset_name, "phases.csv")
            label_to_class = pd.read_csv(path)
            self.label_to_class = dict(zip(label_to_class['Steps'], label_to_class['Meaning']))
        
        else:
            raise ValueError("Invalid dataset name")


    def _load_data(self):
        try:
            # Opening JSON file
            f = open(self.json_path)

            # returns JSON object as a dictionary
            data = json.load(f)

            # data is of type list
            # Each entry in the list is a data sample
            data = data[self.split][self.dataset_name]

            f.close()

            return data
        except FileNotFoundError:
            print("File not found")
            return None


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.root_dir is None:
            folder_path = self.data[index]["File_Path"]
            annotation_path = self.data[index]["Ground_Truth_Path"]
        else:
            folder_path = os.path.join(
                self.root_dir, self.data[index]["File_Path"]
            )
            annotation_path = os.path.join(
                self.root_dir, self.data[index]["Ground_Truth_Path"]
            )
        
        # Load the annotation CSV file
        annotations = pd.read_csv(annotation_path)
        if "9_CATARACTS" in self.dataset_name:
            annotations = annotations[['Frame', 'Steps']]
            annotations.columns = ['FrameNo', 'Phase']
            # Dropping any "idle" frames
            indices_to_drop = annotations[annotations['Phase'] == 0].index
            # Dropping rows by indices
            annotations = annotations.drop(indices_to_drop)


        # count number of frames
        num_frames = len(annotations)
        assert num_frames > 0

        # TODO: Add better support for when the whole video is used, i.e. num_clips = -1
        if self.num_clips != -1:
            required_num_frames = self.clip_size * self.num_clips * self.step_size
        else:
            # If the number of clips is -1, then use all frames
            required_num_frames = num_frames
        
        # The offset variable determines the starting frame of the clip
        offset = 0

        # If there are more frames than required, then sample starting offset
        if required_num_frames < num_frames:
            offset_start_range = num_frames - required_num_frames
            offset = np.random.randint(0, offset_start_range)

        slice_object = slice(offset, required_num_frames + offset, self.step_size)
        
        selected_frames = annotations.iloc[slice_object]

        images = []

        for frame_number in selected_frames["FrameNo"]:
            frame_path = folder_path + f"frame_{frame_number}.jpg"
            image = Image.open(frame_path)
            # transform the image to tensor
            # TODO: Check why am I doing ToImage again
            # transforms = v2.Compose(
            #     [v2.ToDtype(torch.float32, scale=True)]
            # )
            transforms = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            )
            image = transforms(image)

            if self.transform:
                image = self.transform(image)
            
            image = image.unsqueeze(0)
            images.append(image)
        
        # print("annotations.columns:", annotations.columns)
        # print("selected_frames.columns:", selected_frames.columns)
        # labels = selected_frames.drop(columns=["FrameNo"])
        # labels = pd.get_dummies(labels, columns=['Phase'], drop_first=False)
        # print("labels.columns:", labels.columns)
        # labels = torch.tensor(labels.values)
        # print("labels.shape:", labels.shape)

        labels = torch.tensor(selected_frames["Phase"].values)
        if (min([int(element) for element in self.label_to_class.keys()]) == 1):
            labels = labels - 1

        # Convert labels to one-hot encoded format
        labels = F.one_hot(labels, num_classes=self.num_classes)



        # Pad the images if the number of frames is less than the required number of frames
        # The padding is done by repeating the last frame
            # This line (self.num_clips * self.clip_size) - len(images) 
            # calculates the number of frames to pad
        # TODO: Double check logic when using large numbers for clip_size and num_clips
        # TODO: Padding: repeat last frame or insert blank frames (frame of zeros)
        if len(images) < (self.num_clips * self.clip_size):
            images.extend([images[-1]] *
                          ((self.num_clips * self.clip_size) - len(images)))
            labels = torch.cat([labels, labels[-1].repeat(
                ((self.num_clips * self.clip_size) - len(labels), 1))])

        images = torch.cat(images, dim=0)
        return images, labels


class Cataracts_101_21(Dataset):
    def __init__(self, data_list, root_dir=None, transform=None):
        self.data_list = data_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.root_dir is None:
            folder_path = self.data_list[index]["File_Path"]
            frames = [
                os.path.join(folder_path, frame)
                for frame in os.listdir(folder_path)
                if frame.endswith(".jpg") or frame.endswith(".png")
            ]
            annotation_path = self.data_list[index]["Ground_Truth_Path"]
        else:
            folder_path = os.path.join(
                self.root_dir, self.data_list[index]["File_Path"]
            )
            annotation_path = os.path.join(
                self.root_dir, self.data_list[index]["Ground_Truth_Path"]
            )

        # Load the annotation CSV file
        annotations = pd.read_csv(annotation_path)

        # Load all frames and their corresponding phase labels
        frames = [
            os.path.join(folder_path, frame)
            for frame in os.listdir(folder_path)
            if frame.endswith(".jpg") or frame.endswith(".png")
        ]
        frames.sort()  # Ensure frames are in order

        images = []
        labels = []
        for frame in frames:
            image = Image.open(frame)
            # transform the image to tensor
            # image = v2.ToTensor()(image)
            transforms = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            )
            image = transforms(image)

            if self.transform:
                image = self.transform(image)
            images.append(image)

            # Extract the frame name and find the corresponding phase label
            frame_name = extract_frame_number(os.path.basename(frame))
            # print the types of the columns in the pandas dataframe
            # print(annotations.dtypes)
            # print(f"Phase label for frame {frame_name}: ", annotations[annotations['FrameNo'] == int(frame_name)]['Phase'].values)
            # phase_label = annotations[annotations['FrameNo'] == int(frame_name)]['Phase'].values
            # print("Type of phase_label: ", type(phase_label))
            # print("Shape of phase_label: ", phase_label.shape)
            # labels.append(phase_label)
            labels_2 = torch.tensor(annotations["Phase"].values)

        # # Convert labels to a tensor
        #     # print("Labels: ", labels)
        # print("Length of images: ", len(images))
        # print("Length of labels: ", len(labels))
        # labels = torch.tensor(labels, dtype=torch.long)

        # # Stack images to create a batch-like tensor for all frames in the video
        # if len(images) > 0:
        #     images = torch.stack(images)
        # else:
        #     raise RuntimeError(f"No images found in {folder_path}")

        return images, labels_2