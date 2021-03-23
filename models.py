import torch
from torch.utils.data import Dataset
from torchvision.models import vgg19
import numpy as np
import torch.nn as nn
import math


class Residual_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels, 0.8),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels, 0.8)
        )
    def forward(self, x):
        return self.net(x) + x


class Upsampling_Block(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(Upsampling_Block, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * up_scale** 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * up_scale**2),
            nn.PixelShuffle(upscale_factor=up_scale),
            nn.PReLU(),
        )
    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, in_channels = 3, n_residual_blocks = 16, up_scale = 4):
        super(Generator, self).__init__()
        
        self.num_upsample_block = int(math.log(up_scale, 2))
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        
        
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(Residual_Block(in_channels=64))
        self.residual_blocks = nn.Sequential(*res_blocks) 
        
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, 0.8),
        )
        
        
        upsampling = []
        for i in range(self.num_upsample_block):
            upsampling.append(Upsampling_Block(in_channels=64, up_scale=2))          
        self.upsampling = nn.Sequential(*upsampling)
        
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )
    
    def forward(self, x):
        out1 = self.block1(x)
    
        out = self.residual_blocks(out1)
        
        out2 = self.block2(out)
        
        out = torch.add(out1, out2)
        
        
        out = self.upsampling(out)
        
        out = self.block3(out)
    
        return out
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.net(x)
        return torch.sigmoid(out.view(batch_size))
    

class VGGExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)