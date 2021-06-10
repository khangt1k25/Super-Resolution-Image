import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.utils as utils
from tqdm import tqdm
from models import Generator, Discriminator, VGGExtractor
from datasets import TestDatasetFromFolder, display_transform
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel, TVLoss
from models import SRResNet, Generator
import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="SRGAN", type=str, help="Model type")
opt = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opt.model == "SRGAN":
    model = Generator(in_channels=3, n_residual_blocks=7, up_scale=4)
    model.load_state_dict(torch.load('./experiments/srgan.pt', map_location=device)['G_state_dict'])
else:
    model = SRResNet(scale_factor=4, kernel_size=9, n_channels=64)
    model.load_state_dict(torch.load('./experiments/srresnet.pt', map_location=device)['model_state_dict'])


def test_for_dataset(datasetdir):
    outputdir = os.path.join(datasetdir, "tinhtoanSRResnet")
    os.makedirs(outputdir, exist_ok=True)
    testdataset = TestDatasetFromFolder(datasetdir)
    size = len(testdataset)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    avg_psnr = 0.
    avg_ssim = 0.
    
    for batch, data in tqdm(enumerate(testloader), leave=False):
        imagename = str(data[0][0])
        lr, hr_restore, hr = data[1].to(device), data[2].to(device), data[3].to(device)

        assert lr.shape[0] == 1
        with torch.no_grad():
            sr = model(lr)
            sr = torch.clamp(sr, 0.0, 1.0)
        
        sr_img = ToPILImage()(sr.cpu().squeeze())
        hr_img = ToPILImage()(hr.cpu().squeeze())

        psnr = calculate_psnr_y_channel(sr_img, hr_img)
        ssim = calculate_ssim_y_channel(sr_img, hr_img)
        avg_psnr += psnr
        avg_ssim += ssim

        test_images = torch.stack([
            #display_transform()(hr_restore.squeeze(0)),
            #display_transform()(hr.cpu().squeeze(0)),
            display_transform()(sr.cpu().squeeze(0))
        ])
        print(imagename)
        image = utils.make_grid(test_images, nrow=1, padding=0)
        path = os.path.join(outputdir, imagename.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + imagename.split('.')[-1])
        utils.save_image(image, path, padding=0)



    avg_psnr /= (size)
    avg_ssim /= (size)
    print(f"Dataset {datasetdir} : PNSR {avg_psnr} --- SSIM {avg_ssim}")


if __name__ == '__main__':
    DATASETS = ['BSD100', 'Set5', 'Set14']
    UPSCALE_FACTOR = 4
    ROOT = './test/'
    for dataset in DATASETS:
        dataset_folder = os.path.join(ROOT, dataset)
        test_folder = os.path.join(dataset_folder, f'SRF_{UPSCALE_FACTOR}')
        test_for_dataset(test_folder)

