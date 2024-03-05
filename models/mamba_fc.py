import torch
from mamba_ssm import Mamba
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

feature_sizes = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048}


class cata_mamba_fc(nn.Module):
    def __init__(
        self,
        feature_extractor="resnet18", 
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=10,
        N=2, 
    ):
        super(cata_mamba_fc, self).__init__()
        self.feature_extractor = Feature_Extractor(feature_extractor)

        # Create N instances of the Mamba block
        self.mambas = nn.ModuleList(
            [
                Mamba(
                    d_model=feature_sizes[feature_extractor],  # Model dimension
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,  # Local convolution width
                    expand=expand,  # Block expansion factor
                    bias=False,
                )
                for _ in range(N)
            ]
        )

        self.fc = nn.Linear(feature_sizes[feature_extractor], num_classes)
        # Learnable parameter for scaling the residual connection
        self.res_scale = nn.Parameter(torch.ones(1))

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        spatial_features = self.feature_extractor(x)
        x = spatial_features

        for mamba in self.mambas:
            residual = x  # Store current state to act as a residual for the connection
            x = mamba(x)
            x += residual  # Add the residual (input) to the output of the Mamba block

        # Add the scaled residual connection from the feature extraction to the conv1d input
        x += self.res_scale * spatial_features  # Scale and add the feature extraction output to the mamba outputs

        x = self.fc(x)
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

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.cnn(x)  # (batch_size * sequence_length, feature_size)
        x = x.view(batch_size, sequence_length, -1)
        return x
