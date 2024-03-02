import torch
from mamba_ssm import Mamba
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

feature_sizes = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048}


class cata_mamba(nn.Module):
    def __init__(
        self,
        feature_extractor="resnet18", 
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=10,
        N=2, 
        dilation_levels=3,
    ):
        super(cata_mamba, self).__init__()
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

        self.conv1d = DynamicConv1D(
            num_convs=dilation_levels,
            in_channels=feature_sizes[feature_extractor],
            out_channels=num_classes,
            kernel_size=1,
            num_classes=10,
        )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        x = self.feature_extractor(x)

        for mamba in self.mambas:
            residual = x  # Store current state to act as a residual for the connection
            x = mamba(x)
            x += residual  # Add the residual (input) to the output of the Mamba block

        x = x.transpose(
            1, 2
        )
        x = self.conv1d(x)
        x = x.transpose(1, 2)
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


class DynamicConv1D(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, kernel_size, num_classes):
        super(DynamicConv1D, self).__init__()
        # Ensure dilation_rates is a list with length equal to num_convs
        # assert len(dilation_rates) == num_convs, "Dilation rates list must match number of convolutions."
        dilation_rates = [2**i for i in range(num_convs)]

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, dilation=dilation_rate
                )
                for dilation_rate in dilation_rates
            ]
        )

        # Mixing weights for combining the outputs of the convolutions
        self.mixing_weights = nn.Parameter(torch.randn(num_convs))

        # # Final output layer
        # self.final_conv = nn.Conv1d(out_channels, num_classes, 1)  # Kernel size of 1 for class scores

    def forward(self, x):
        # Collect outputs from each convolution
        conv_outputs = [conv(x) for conv in self.convs]

        # Combine outputs using learned mixing weights
        weights = F.softmax(
            self.mixing_weights, dim=0
        )  # Ensure the weights sum up to 1
        combined = torch.stack(
            [weights[i] * conv_outputs[i] for i in range(len(self.convs))], dim=0
        )
        out = torch.sum(combined, dim=0)  # Sum across the first dimension
        # out = self.final_conv(out)  # Final output layer
        return out
