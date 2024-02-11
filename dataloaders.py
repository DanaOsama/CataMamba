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


"""
All Dataset classes receive the split from the JSON file

For Cataracts101, the usage looks like this:
json_path = "2_NEW_dataset_level_labels.json"
# Opening JSON file
f = open(json_path)

# returns JSON object as a dictionary
data = json.load(f)
data = data['Train']['2_Cataracts-101']

How would this framework fit within a healthcare workflow?
"""


def extract_frame_number(filename):
    # The regex pattern \d+ matches one or more digits
    match = re.search(r"\d+", filename)
    if match:
        return (
            match.group()
        )  # This will return the first occurrence of one or more digits in the string
    else:
        return None  # Or raise an error/exception if preferred


class IrisPupilSegmentation(Dataset):
    pass


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
        self.root_dir = root_dir
        self.json_path = json_path
        self.dataset_name = dataset_name
        self.split = split
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.clip_size = clip_size
        self.step_size = step_size
        self.transform = transform

        # self.num_frames = []
        # self.csv_files = []
        self.data = self._load_data()

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

            # Sort the videos by VideoID
            video_extra_info.sort_values(by="VideoID", inplace=True)

            self.surgeon_ids = video_extra_info["Surgeon"].values
            self.surgeon_experience = video_extra_info["Experience"].values
        
        elif self.dataset_name == "1_Cataracts-21":
            # Loading the csv file for label-to-class mapping
            path = os.path.join(root_dir, "Cataracts_Multitask/1_Cataracts-21/phases.csv")
            label_to_class = pd.read_csv(path)
            self.label_to_class = dict(zip(label_to_class['Phase'], label_to_class['Meaning']))
        
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

        # count number of frames
        num_frames = len(annotations)
        assert num_frames > 0

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
            # image = v2.ToTensor()(image)
            transforms = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            )
            image = transforms(image)

            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Pad the images if the number of frames is less than the required number of frames
        # The padding is done by repeating the last frame
            # This line (self.num_clips * self.clip_size) - len(images) 
            # calculates the number of frames to pad
        if len(images) < (self.num_clips * self.clip_size):
            images.extend([images[-1]] *
                          ((self.num_clips * self.clip_size) - len(images)))
        
        labels = torch.tensor(selected_frames["Phase"].values)
        if (min([int(element) for element in self.label_to_class.keys()]) == 1):
            labels = labels - 1

        # Convert labels to one-hot encoded format
        labels = F.one_hot(labels, num_classes=self.num_classes)

        # Stack images to create a batch-like tensor for all frames in the video
        images = torch.stack(images, dim=0)

        # print("Labels: ", labels)
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


class TaskLoader(Dataset):

    def __init__(self, json_path, task_list, transform=None):
        """
        Args:
            json_path (string): Path to the JSON file with video paths and labels.
            task_list (list): List of tasks that will be loaded. All possible tasks are:
                ['All_Segmentation', 'Pupil_Segmentation', 'Lens_Segmentation', 'Iris/Pupil Segmentation',
                    'Phase_Detection', 'Tool_Presence', 'Step_Detection']
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        tasks = {task: [] for task in task_list}
        self.task_list = task_list
        self.splits = {split: tasks.copy() for split in data.keys()}
        self.transform = transform

        json_path = "3_NEW_Task_level_labels.json"
        # Opening JSON file
        f = open(json_path)

        # returns JSON object as a dictionary
        data = json.load(f)

        for split in data.keys():
            for task in tasks.keys():
                # print(f"Split: {split}  Task: {task}")
                if "Segmentation" in task:
                    for item in data[split]["Segmentation"][task]:
                        temp = {item["File_Path"]: item["Ground_Truth_Path"]}
                        self.splits[split][task].append(temp)
                else:
                    for item in data[split][task]:
                        temp = {item["File_Path"]: item["Ground_Truth_Path"]}
                        self.splits[split][task].append(temp)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        frames = [
            os.path.join(video_path, frame)
            for frame in os.listdir(video_path)
            if frame.endswith(".jpg") or frame.endswith(".png")
        ]
        frames.sort()  # Ensure frames are in order

        # Load frames
        images = [Image.open(frame) for frame in frames]
        if self.transform:
            images = [self.transform(image) for image in images]

        # Load labels
        labels = self.labels[video_path]

        return torch.stack(images), labels


# class IXIDataset(Randomizable, CacheDataset):
#     resource = "http://biomedic.doc.ic.ac.uk/" + "brain-development/downloads/IXI/IXI-T1.tar"
#     md5 = "34901a0593b41dd19c1a1f746eac2d58"

#     def __init__(
#         self,
#         root_dir,
#         section,
#         transform,
#         download=False,
#         seed=0,
#         val_frac=0.2,
#         test_frac=0.2,
#         cache_num=sys.maxsize,
#         cache_rate=1.0,
#         num_workers=0,
#     ):
#         if not os.path.isdir(root_dir):
#             raise ValueError("Root directory root_dir must be a directory.")
#         self.section = section
#         self.val_frac = val_frac
#         self.test_frac = test_frac
#         self.set_random_state(seed=seed)
#         dataset_dir = os.path.join(root_dir, "ixi")
#         tarfile_name = f"{dataset_dir}.tar"
#         if download:
#             download_and_extract(self.resource, tarfile_name, dataset_dir, self.md5)
#         # as a quick demo, we just use 10 images to show

#         self.datalist = [
#             {"image": os.path.join(dataset_dir, "IXI314-IOP-0889-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI249-Guys-1072-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI609-HH-2600-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI173-HH-1590-T1.nii.gz"), "label": 1},
#             {"image": os.path.join(dataset_dir, "IXI020-Guys-0700-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI342-Guys-0909-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI134-Guys-0780-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI577-HH-2661-T1.nii.gz"), "label": 1},
#             {"image": os.path.join(dataset_dir, "IXI066-Guys-0731-T1.nii.gz"), "label": 1},
#             {"image": os.path.join(dataset_dir, "IXI130-HH-1528-T1.nii.gz"), "label": 0},
#         ]
#         data = self._generate_data_list()
#         super().__init__(
#             data,
#             transform,
#             cache_num=cache_num,
#             cache_rate=cache_rate,
#             num_workers=num_workers,
#         )

#     def randomize(self, data=None):
#         self.rann = self.R.random()

#     def _generate_data_list(self):
#         data = []
#         for d in self.datalist:
#             self.randomize()
#             if self.section == "training":
#                 if self.rann < self.val_frac + self.test_frac:
#                     continue
#             elif self.section == "validation":
#                 if self.rann >= self.val_frac:
#                     continue
#             elif self.section == "test":
#                 if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
#                     continue
#             else:
#                 raise ValueError(
#                     f"Unsupported section: {self.section}, " "available options are ['training', 'validation', 'test']."
#                 )
#             data.append(d)
#         return data
