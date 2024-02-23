import torch
import torch.nn as nn
import torchvision.models as models

feature_sizes = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048}


class CNN(nn.Module):
    def __init__(self, cnn="resnet50", num_classes=10):
        super(CNN, self).__init__()
        if cnn == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif cnn == "resnet50":
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif cnn == "resnet101":
            self.cnn = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Invalid CNN model name")
        self.cnn.fc = nn.Identity()  # Use the CNN as a fixed feature extractor
        self.fc = nn.Linear(feature_sizes[cnn], num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()

        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.cnn(x)
        x = self.fc(x)
        x = x.view(batch_size, sequence_length, -1)
        return x
