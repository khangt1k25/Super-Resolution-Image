import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from models import Generator, Discriminator, VGGExtractor
from datasets import TrainDataset, ValidDataset
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel, TVLoss
from tqdm import tqdm


adv_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
tv_loss = TVLoss()


class SRGAN_Trainer():
    def __init__(self, generator, discriminator, vggExtractor, optimizer_G, optimizer_D, device):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.vggExtractor = vggExtractor.to(device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D


    def train(self, trainloader, trainloader_v2, validloader, start_epoch, end_epoch):
        
        for epoch in range(start_epoch, end_epoch+1):
            self.generator.train()
            self.discriminator.train()
            Gloss_epoch = 0.
            Dloss_epoch = 0.
            for batch, data in tqdm(enumerate(trainloader), leave=False):
                lr = Variable(data['lr']).to(self.device)
                hr = Variable(data['hr']).to(self.device)
                batch_size = lr.shape[0]
                
                valid = Variable(torch.Tensor(batch_size).fill_(1.0), requires_grad=False).to(self.device)
                fake = Variable(torch.Tensor(batch_size).fill_(0.0), requires_grad=False).to(self.device)
                
            
                sr = self.generator(lr)
                
                # optimize G
                optimizer_G.zero_grad()
                
                loss_G = .001*adv_loss(self.discriminator(sr), valid) + \
                    0.006*l1_loss(self.vggExtractor(hr).detach(), self.vggExtractor(sr))+\
                        l1_loss(hr.detach(), sr) + 2e-8* tv_loss(sr)
                loss_G.backward()
                optimizer_G.step()

                # optimize D
                optimizer_D.zero_grad()
                loss_D = adv_loss(self.discriminator(sr.detach()), fake)+\
                        adv_loss(self.discriminator(hr), valid)
                    
                loss_D.backward()
                optimizer_D.step()

                Gloss_epoch += loss_G.item()
                Dloss_epoch += loss_D.item()

            
            
            Gloss_epoch /= (batch+1)
            Dloss_epoch /= (batch+1)

            print(f'\nEpoch {epoch} -- D_loss {Dloss_epoch} -- G_loss {Gloss_epoch}\n')
          

            if epoch % 10 == 0:
                psnr_valid, ssim_valid = self.valid(validloader)
                psnr_train, ssim_train = self.valid(trainloader_v2)
                
                print(f'\nEpoch {epoch} -- PSNR train {psnr_train} -- PSNR valid {psnr_valid}\n')
                print(f'\nEpoch {epoch} -- SSIM train {ssim_train} -- SSIM valid {ssim_valid}\n')
                self.saving(epoch)
               

    def valid(self, loader):
        self.generator.eval()
        self.discriminator.eval()
        avg_psnr = 0.
        avg_ssim = 0.
        for batch, data in tqdm(enumerate(loader), leave=False):
            lr, hr= data['lr'].to(self.device), data['hr'].to(self.device)
            assert lr.shape[0] == 1
            with torch.no_grad():
                sr = self.generator(lr)
            
            sr_img = ToPILImage()(sr.cpu().squeeze())
            hr_img = ToPILImage()(hr.cpu().squeeze())

            psnr = calculate_psnr_y_channel(sr_img, hr_img)
            ssim = calculate_ssim_y_channel(sr_img, hr_img)
            avg_psnr += psnr
            avg_ssim += ssim
        avg_psnr /= (batch+1)
        avg_ssim /= (batch+1)

        return avg_psnr, avg_ssim
    

    def saving(self, epoch):
        filename = f'./experiments/srgan_{epoch}.pt'
                
        torch.save({
            'G_state_dict':generator.state_dict(),
            'D_state_dict':discriminator.state_dict(),
            'optimizer_D_state_dict':optimizer_D.state_dict(),
            'optimizer_G_state_dict':optimizer_G.state_dict()
            }, filename
        ) 


    def load(self, epoch):
        filename = f'./experiments/srgan_{epoch}.pt'
        try:
            checkpoint = torch.load(filename)

            self.generator.load_state_dict(checkpoint['G_state_dict'])
            self.discriminator.load_state_dict(checkpoint['D_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            print(f'Load model at epoch {epoch} successfully')
        except:
            print(f'Load model at epoch {epoch} fail')

        

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



    generator = Generator(in_channels=3, n_residual_blocks=nblocks, up_scale=upscale_factor)
    discriminator = Discriminator()
    vggExtractor = VGGExtractor()
    optimizer_G = torch.optim.Adam(params=generator.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=lr, betas=betas)
    

    ## Training
    trainer = SRGAN_Trainer(generator, discriminator, vggExtractor, optimizer_G, optimizer_D, device)
    trainer.train(trainloader, trainloader_v2,  validloader, 1, 50)