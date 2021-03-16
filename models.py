import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import glob
import torch.nn as nn
import math
from dataset import ImageDataset

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
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        # )
        
        def Discriminator_Block(in_c, out_c, first=False):
            layers = []
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
            
            if first == False:
                layers.append(nn.BatchNorm2d(out_c))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            return layers
        
        layers = []
        in_c = in_channels
        for i, out_c in enumerate([64, 128, 256, 512]):
            block_layer = Discriminator_Block(in_c, out_c, first=(i==0))
            layers.extend(block_layer)
            in_c = out_c
        
        layers.append(nn.Conv2d(out_c, 1, kernel_size=3, stride=1, padding=1))
         
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.net(x)
        print(out.shape)
        return out
    
        

if __name__=="__main__":
    voc2012 = ImageDataset('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/', (88,88))
    dataloader = torch.utils.data.DataLoader(voc2012, 
                                         batch_size=32, 
                                         shuffle=True, 
                                         num_workers=2)

    gen = Generator(in_channels=3, n_residual_blocks=1)
    disc = Discriminator(in_channels=3)
    for b, data in enumerate(dataloader):
        print(b)
        lr, hr = data['lr'], data['hr']
        out = gen(lr)
        print(out.shape)
        prob = disc(out)
        break

 