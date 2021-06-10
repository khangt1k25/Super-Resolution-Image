import torch
import torch.nn as nn
from models import Generator, SRResNet
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import argparse
import os

parse = argparse.ArgumentParser()

parse.add_argument("--image_in", default="1.jpg", type=str, help="Name of input image")
parse.add_argument("--image_out", default="1_result.jpg", type=str, help="Name of output image")
parse.add_argument("--model", default="SRGAN", type=str, help="SRGAN or SRResnet")
opt = parse.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






if __name__ == '__main__':
    if opt.model == "SRGAN":
        model = Generator(in_channels=3, n_residual_blocks=7, up_scale=4)
        model.load_state_dict(torch.load('./experiments/srgan.pt', map_location=device)['G_state_dict'])
    else:
        model = SRResNet(scale_factor=4, kernel_size=9, n_channels=64)
        model.load_state_dict(torch.load('./experiments/srresnet.pt', map_location=device)['model_state_dict'])

    root_in = './pics/SRF_4/LR'
    root_out = './pics/SRF_4/result'
    input_path = os.path.join(root_in, opt.image_in)
    output_path = os.path.join(root_out, opt.image_out)
    image = Image.open(input_path)

    lr = ToTensor()(image)
    lr = lr.unsqueeze(0)
    #lr = lr[:,:3,:,:]
    # print(lr.shape)
    print('Forwarding ...')
    with torch.no_grad():
        sr = model(lr)
        sr = torch.clamp(sr, 0.0, 1.0)
    sr_img = ToPILImage()(sr.cpu().squeeze())
    
    sr_img.save(output_path)
    print('Completed\n')
    
