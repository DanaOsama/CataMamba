import torch
from mamba_ssm import Mamba
import torch.nn as nn
import torchvision.models as models


feature_sizes = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048}


class mamba_cat(nn.Module):
    def __init__(
        self,
        feature_extractor="resnet18",
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=10,
    ):
        super(mamba_cat, self).__init__()
        self.feature_extractor = Feature_Extractor(feature_extractor)

        self.mamba = Mamba(
            d_model=feature_sizes[feature_extractor],  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bias=False,
        )
        out_proj_in_features = self.mamba.out_proj.in_features
        self.mamba.out_proj = torch.nn.Linear(out_proj_in_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.mamba(x)
        return x


class Feature_Extractor(nn.Module):
    def __init__(self, cnn="resnet50"):
        super(Feature_Extractor, self).__init__()
        if cnn == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif cnn == "resnet50":
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif cnn == "resnet101":
            self.cnn = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Invalid CNN model name")
        self.cnn.fc = nn.Identity()  # Use the CNN as a fixed feature extractor
        # self.fc = nn.Linear(feature_sizes[cnn], num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()

        x = x.view(batch_size * sequence_length, C, H, W)

        x = self.cnn(x)  # (batch_size * sequence_length, feature_size)
        x = x.view(batch_size, sequence_length, -1)
        return x
