import torch 
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader
from models import Generator, Discriminator, VGGExtractor
from datasets import TrainDataset, ValidDataset
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel


class SRGAN():
    def __init__(self, generator, discriminator, vggExtractor, optimizer_G, optimizer_D, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.vggExtractor = vggExtractor.to(device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
    
    def train(self, trainloader, validloader, epochs):
        self.generator.train()
        self.discriminator.train()
        
        plot_Gloss= []
        plot_Dloss= []
        for epoch in range(1, epochs+1):
            Gloss_epoch = 0.
            Dloss_epoch = 0.
            for batch, data in tqdm(enumerate(trainloader), leave=False):
                lr = Variable(data['lr']).to(device)
                hr = Variable(data['hr']).to(device)
                batch_size = lr.shape[0]
                
                valid = Variable(torch.Tensor(batch_size).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(torch.Tensor(batch_size).fill_(0.0), requires_grad=False).to(device)
                
            
                sr = self.generator(lr)
                
                # optimize G
                optimizer_G.zero_grad()
                
                loss_G = .001*adv_loss(self.discriminator(sr), valid) + \
                    0.006*l1_loss(self.vggExtractor(hr).detach(), self.vggExtractor(sr))+\
                        l1_loss(hr.detach(), sr)
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
            plot_Gloss.append(Gloss_epoch)
            plot_Dloss.append(Dloss_epoch)

            if epoch % 10 == 0:
                self.valid(validloader)
                filename = f'/content/drive/MyDrive/Colab Notebooks/Super-Resolution/checkpoint/epoch{epoch}.pt'
                torch.save({
                    'G_state_dict':generator.state_dict(),
                    'D_state_dict':discriminator.state_dict()
                    }, filename
                )
                
                
                
            print(f'\nEpoch {epoch} -- D_loss {Dloss_epoch} -- G_loss {Gloss_epoch}\n')
        
        return plot_Gloss, plot_Dloss

    def valid(self, loader):
        self.generator.eval()
        self.discriminator.eval()
        avg_psnr = 0.
        avg_ssim = 0.
        for batch, data in tqdm(enumerate(loader), leave=False):
            lr, hr, hr_restore = data['lr'].to(device), data['hr'].to(device), data['hr_restore']
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Config
    crop_size=88
    upscale_factor=4
    batch_size=64
    nblocks = 5
    lr = 0.001
    betas = (0.99, 0.999)
    
    
    ## Set up
    train_dataset = TrainDataset(TRAIN_PATH, crop_size=crop_size, upscale_factor=upscale_factor)
    validdataset = ValidDataset(VALID_PATH, crop_size=crop_size, upscale_factor=upscale_factor)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    generator = Generator(in_channels=3, n_residual_blocks=nblocks, up_scale=upscale_factor)
    discriminator = Discriminator()
    vggExtractor = VGGExtractor()
    optimizer_G = torch.optim.Adam(params=generator.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=lr, betas=betas)
    adv_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    ## Training
    srgan = SRGAN(generator, discriminator, vggExtractor, optimizer_G, optimizer_D, device)
    plot_G, plot_D = srgan.train(trainloader, validloader, 100)