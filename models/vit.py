import torch
import torch.nn as nn
import torchvision.models as models


class ViT(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        hidden_dim=768,
        mlp_dim=3072,
        attention_dropout=0.0,
        dropout=0.0,
        representation_size=None,
        norm_layer=nn.LayerNorm,
        seq_length=197,
    ):
        super(ViT, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # print(vars(self.vit))

        self.vit.num_classes = num_classes
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)
        # TODO: Check this. Where is it coming from? 
        self.vit.image_size = image_size
        self.vit.patch_size = patch_size
        self.vit.hidden_dim = hidden_dim
        self.vit.mlp_dim = mlp_dim
        self.vit.attention_dropout = attention_dropout
        self.vit.dropout = dropout
        self.vit.representation_size = representation_size
        self.vit.norm_layer = norm_layer
        self.vit.seq_length = seq_length

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.vit(x)
        x = x.view(batch_size, sequence_length, -1)
        return x
