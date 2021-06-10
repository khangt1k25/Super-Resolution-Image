from skimage.metrics import structural_similarity
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import math
import torch
import torch.nn as nn

# just take y channel
def convert_to_y_channel(img: Image.Image):
    
    img = np.array(img.convert('RGB'))
    return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256 

# makesure y_img1 and y_img2 in range(0, 1)
def calculate_psnr_y_channel(img1: Image.Image, img2: Image.Image) -> float:

    y_img1 = convert_to_y_channel(img1)
    y_img2 = convert_to_y_channel(img2)
    return 10 * math.log10(1.0/((y_img1/255.0 - y_img2/255.0)**2).mean() + 1e-8)  


# makesure y_img1 and y_img2 in range(0, 1)

def calculate_ssim_y_channel(img1: Image.Image, img2: Image.Image):
    y_img1 = convert_to_y_channel(img1)
    y_img2 = convert_to_y_channel(img2)
    return structural_similarity(y_img1 / 255.0, y_img2 / 255.0)  

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]