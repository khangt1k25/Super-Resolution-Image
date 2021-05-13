import torch
import torch.nn as nn
from models import Generator
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator(in_channels=3, n_residual_blocks=7, up_scale=4)

model.load_state_dict(torch.load('./l2.pt', map_location=device)['G_state_dict'])

image = Image.open('./pics/camtu.jpg')


lr = ToTensor()(image)
lr = lr.unsqueeze(0)
print('Forwarding ...')
with torch.no_grad():
    sr = model(lr)
    #sr = torch.clamp(sr, 0.0, 1.0)

sr_img = ToPILImage()(sr.cpu().squeeze())
sr_img.save('./pics/camtu_after.png')
print('Completed\n')

