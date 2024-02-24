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


import argparse

parser = argparse.ArgumentParser(description="MultiTask Cataracts Qualitative Results Script")
