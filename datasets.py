from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import pickle
import glob

class TrainDataset(Dataset):
    def __init__(self, path, crop_size, upscale_factor):
        
        super(TrainDataset, self).__init__()
        with open(path, 'rb') as f:
            self.images = pickle.load(f)
        self.hr_transformer = transforms.Compose(
            [
                transforms.RandomCrop(crop_size),
                ToTensor()
            ]
        )
        self.lr_transformer = transforms.Compose(
            [
                ToPILImage(),
                transforms.Resize((crop_size//upscale_factor, crop_size//upscale_factor), Image.BICUBIC),
                ToTensor(),
            
            ]
        )

    def __getitem__(self, index):
        
        image = Image.fromarray(self.images[index])
        hr = self.hr_transformer(image)
        lr = self.lr_transformer(hr)
        #to_tensor = transforms.ToTensor()
        return  {'lr' :lr, 'hr': hr}

    def __len__(self):
        return len(self.images)


class ValidDataset(Dataset):
    def __init__(self, path, crop_size, upscale_factor):
        super(ValidDataset, self).__init__()
        with open(path, 'rb') as f:
            self.images = pickle.load(f)
        
        self.hr_transformer = transforms.Compose(
            [
                transforms.RandomCrop(crop_size),
                ToTensor(),
            ]
        )
        self.lr_transformer = transforms.Compose(
            [
                ToPILImage(),
                transforms.Resize((crop_size//upscale_factor, crop_size//upscale_factor), Image.BICUBIC),
                ToTensor(),
            ]
        )
        self.hr_restore_transformer = transforms.Compose(
            [
                ToPILImage(),
                transforms.Resize(crop_size, Image.BICUBIC),
                ToTensor()
            ]
        )
        
    def __getitem__(self, index):
        
        image = Image.fromarray(self.images[index])
        hr = self.hr_transformer(image)
        lr = self.lr_transformer(hr)
        hr_restore = self.hr_restore_transformer(lr)
        return  {'lr': lr, 'hr': hr, 'hr_restore': hr_restore}
    
    def __len__(self):
        return len(self.images)
        