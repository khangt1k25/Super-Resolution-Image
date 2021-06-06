from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import pickle
import glob
import os

# Train and valid dataset used from npy to train by Colab"
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
        


# Test dataset used from Folder of images"
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor=4):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = os.path.join(dataset_dir, 'LR')
        self.hr_path = os.path.join(dataset_dir, 'HR')
        self.upscale_factor = upscale_factor
        self.lr_filenames = [
            os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [
            os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])