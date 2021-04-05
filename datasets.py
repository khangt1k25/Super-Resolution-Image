from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
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
            ]
        )
        self.lr_transformer = transforms.Compose(
            [
                transforms.Resize((crop_size//upscale_factor, crop_size//upscale_factor), Image.BICUBIC),
            ]
        )

    def __getitem__(self, index):
        
        image = Image.fromarray(self.images[index])
        hr = self.hr_transformer(image)
        lr = self.lr_transformer(hr)
        to_tensor = transforms.ToTensor()
        return  {'lr' :to_tensor(lr), 'hr': to_tensor(hr)}

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
            ]
        )
        self.lr_transformer = transforms.Compose(
            [
                transforms.Resize((crop_size//upscale_factor, crop_size//upscale_factor), Image.BICUBIC),
            ]
        )
        self.hr_restore_transformer = transforms.Compose(
            [
                transforms.Resize(crop_size, Image.BICUBIC)
            ]
        )
        
    def __getitem__(self, index):
        
        image = Image.fromarray(self.images[index])
        hr = self.hr_transformer(image)
        lr = self.lr_transformer(hr)
        hr_restore = self.hr_restore_transformer(lr)
        to_tensor = transforms.ToTensor()    
        return  {'lr': to_tensor(lr), 'hr': to_tensor(hr), 'hr_restore': to_tensor(hr_restore)}
    
    def __len__(self):
        return len(self.images)
        