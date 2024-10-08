import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .utils import *

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches, num_patches]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # Shape: [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch_size, num_patches + 1, embed_dim]
        x = x + self.pos_embedding
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_out))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, 
                 num_classes=10, norm = False, mean = None, std = None):
        super(VisionTransformer, self).__init__()
        self._num_classes = num_classes
        self.norm = norm
        self.mean = mean
        self.std = std
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, self.embed_dim)
        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(self.embed_dim, num_heads, self.embed_dim * 4) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dim, self._num_classes)
    
    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        self.head = nn.Linear(self.embed_dim, self._num_classes)
        
    def forward(self, x):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = x[:, 0]  # Select the CLS token
        # x = self.pool(x.unsqueeze(1)).squeeze(1)
        x = self.head(x)
        return x

def ViT_B16(num_classes=10, Norm=True, norm_mean=None, norm_std=None):
    return VisionTransformer(img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=12, 
                             num_classes=num_classes, norm=Norm, mean=norm_mean, std=norm_std)

def ViT_B32(num_classes=10, Norm=True, norm_mean=None, norm_std=None):
    return VisionTransformer(img_size=224, patch_size=32, in_channels=3, embed_dim=768, num_heads=12, num_layers=12, 
                             num_classes=num_classes, norm=Norm, mean=norm_mean, std=norm_std)