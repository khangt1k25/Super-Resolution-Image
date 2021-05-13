import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from models import SRResNet
from datasets import TrainDataset, ValidDataset
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel
from tqdm import tqdm



criterion = nn.MSELoss()

class SRResnet_trainer():
    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer.to(self.device)
        
        self.l = 0 #version
        self.log_path_ssim = f'Logs/SSIM_{self.l}.txt'
        self.log_path_psnr = f'Logs/PSNR_{self.l}.txt'
        self.log_path_loss = f'Logs/GLoss_{self.l}.txt'


    def train(self, trainloader, trainloader_v2, validloader, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch+1):
            self.model.train()

            loss_epoch = 0.
            b=0
            for batch, data in tqdm(enumerate(trainloader), leave=False):
                lr = Variable(data['lr']).to(self.device)
                hr = Variable(data['hr']).to(self.device)
              

                sr = self.model(lr)
                
                # optimize G
                self.optimizer.zero_grad()
                
                loss = criterion(sr, hr)
                loss.backward()

                self.optimizer.step()


                loss_epoch += loss.item()
                b+=1
            
            
            loss_epoch /= b

            psnr_valid, ssim_valid = self.valid(validloader)
            psnr_train, ssim_train = self.valid(trainloader_v2)

            self.save_logs(loss_epoch, ssim_valid,psnr_valid)

            print(f'\nEpoch {epoch} -- PSNR train {psnr_train} -- PSNR valid {psnr_valid} -- Loss {loss_epoch}\n')
            print(f'\nEpoch {epoch} -- SSIM train {ssim_train} -- SSIM valid {ssim_valid}\n')

        self.saving()

    
    def save_logs(self, loss_epoch, ssim_valid, psnr_valid):
        with open(self.log_path_loss, 'a') as f:
          f.write(str(loss_epoch))
          f.write('\n')
        with open(self.log_path_ssim, 'a') as f:
          f.write(str(ssim_valid))
          f.write('\n')
        with open(self.log_path_psnr, 'a') as f:
          f.write(str(psnr_valid))
          f.write('\n')  

    def valid(self, loader):
        self.model.eval()

        avg_psnr = 0.
        avg_ssim = 0.
        for batch, data in tqdm(enumerate(loader), leave=False):
            lr, hr= data['lr'].to(self.device), data['hr'].to(self.device)
            assert lr.shape[0] == 1
            with torch.no_grad():
                sr = self.model(lr)
            
            sr_img = ToPILImage()(sr.cpu().squeeze())
            hr_img = ToPILImage()(hr.cpu().squeeze())

            psnr = calculate_psnr_y_channel(sr_img, hr_img)
            ssim = calculate_ssim_y_channel(sr_img, hr_img)
            avg_psnr += psnr
            avg_ssim += ssim
        avg_psnr /= (batch+1)
        avg_ssim /= (batch+1)

        return avg_psnr, avg_ssim
    

    def saving(self):
        filename = f'./checkpoint/srresnet.pt'
                
        torch.save({
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
            }, filename
        )


    def load(self):
        filename = f'./checkpoint/srresnet.pt'
        try:
            checkpoint = torch.load(filename)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Load model successfully')
        except:
            print(f'Load model fail')

        

if __name__ == "__main__":
    

    ## Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size=88
    upscale_factor=4
    batch_size=64
    nblocks = 3
    lr = 0.001
    betas = (0.99, 0.999)
    TRAIN_PATH = './compress_data/voc_train.pkl'
    VALID_PATH = './compress_data/voc_valid.pkl'
    
    
    ## Set up
    train_dataset = TrainDataset(TRAIN_PATH, crop_size=crop_size, upscale_factor=upscale_factor)
    valid_dataset = ValidDataset(VALID_PATH, crop_size=crop_size, upscale_factor=upscale_factor)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    trainloader_v2 = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2) # need to calculate score metrics
    validloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)



    model = SRResNet(scale_factor=upscale_factor, kernel_size=9, n_channels=64)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=betas)
    

    ## Training
    srresnet = SRResnet_trainer(model, optimizer, device)
    srresnet.train(trainloader, trainloader_v2,  validloader, 1, 100)