import torch
import torch.nn as nn
import torchvision.models as models

class ViT(nn.Module):
    def __init__(self, num_classes, dim=None, heads=None, depth=None, mlp_dim=None, pool="cls"):
        super(ViT, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # print(vars(self.vit))

        self.vit.num_classes = num_classes
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)

        # self.dim = dim
        # self.heads = heads
        # self.depth = depth

        # self.patch_size = 16
        # self.patch_dim = 3 * self.patch_size ** 2
        # self.num_patches = 32
        # self.patch_dim = dim
        # self.patch_size = 16

        # self.patch_to_embedding = nn.Linear(self.patch_dim, self.dim)
        # self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # self.transformer = nn.TransformerEncoder
    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()

        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.vit(x)
        print("x.size after vit", x.size())
        x = x.view(batch_size, sequence_length, -1)
        print("x.size after view", x.size())

        return x