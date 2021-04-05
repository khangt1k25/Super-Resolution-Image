from skimage.metrics import structural_similarity
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import math


def convert_to_y_channel(img: Image.Image):
    
    img = np.array(img.convert('RGB'))
    return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256 # just take y channel


def calculate_psnr_y_channel(img1: Image.Image, img2: Image.Image) -> float:

    y_img1 = convert_to_y_channel(img1)
    y_img2 = convert_to_y_channel(img2)
    return 10 * math.log10(1.0/((y_img1/255.0 - y_img2/255.0)**2).mean() + 1e-8)  # makesure y_img1 and y_img2 in range(0, 1)


def calculate_ssim_y_channel(img1: Image.Image, img2: Image.Image):
    
    y_img1 = convert_to_y_channel(img1)
    y_img2 = convert_to_y_channel(img2)
    return structural_similarity(y_img1 / 255.0, y_img2 / 255.0)  # makesure y_img1 and y_img2 in range(0, 1)