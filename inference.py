import torch
import torch.nn as nn
from models import Generator, SRResNet
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = Generator(in_channels=3, n_residual_blocks=7, up_scale=4)
model = SRResNet(scale_factor=4, kernel_size=9, n_channels=64)

model.load_state_dict(torch.load('./experiments/srresnet.pt', map_location=device)['model_state_dict'])

image = Image.open('./pics/hellu_bg.jpg')

print(image.size)
lr = ToTensor()(image)
lr = lr.unsqueeze(0)
#lr = lr[:,:3,:,:]
print(lr.shape)
print('Forwarding ...')
with torch.no_grad():
    sr = model(lr)
    #sr = torch.clamp(sr, 0.0, 1.0)

sr_img = ToPILImage()(sr.cpu().squeeze())
sr_img.save('./pics/hellu_bg_resnet.jpg')
#sr_img.save('./pics/camtu_sf_after2.jpg')

print('Completed\n')

