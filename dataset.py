import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_h, hr_w = hr_shape
        
        self.lr_transformer = transforms.Compose(
                [
                    transforms.Resize((hr_h//4, hr_h//4), Image.BICUBIC),
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                ]
        )
        
        self.hr_transformer = transforms.Compose(
                [
                    transforms.Resize((hr_h, hr_h), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                ]
        )
        
        self.files = sorted(glob.glob(root+'/*.*'))
        
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        
        lr = self.lr_transformer(img)
        hr = self.hr_transformer(img)
        
        return  {'lr':lr, 'hr':hr}
    def __len__(self):
        return len(self.files)