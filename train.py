from models import Generator, Discriminator, VGGExtractor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import pickle


# load data
train_path = "../compress_data/train_data.pkl"
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
train_loader = DataLoader(train_data, batch_size=64)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# init model & optimizer & loss tools
generator = Generator(in_channels=3, n_residual_blocks=5, up_scale=4).to(device)
discriminator = Discriminator(in_channels=3).to(device)
vggExtractor = VGGExtractor().to(device)
optimizer_G = torch.optim.Adam(params=generator.parameters())
optimizer_D = torch.optim.Adam(params=discriminator.parameters())
adv_loss = nn.BCELoss()
l1_loss = nn.L1Loss()


# start training
for epoch in range(epochs):
    Gloss_epoch = 0.
    Dloss_epoch = 0.
    for batch, data in enumerate(tqdm(train_loader)):
        lr = Variable(data['lr']).to(device)
        hr = Variable(data['hr']).to(device)
        batch_size = lr.shape[0]
        
        valid = Variable(torch.Tensor(batch_size).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(batch_size).fill_(0.0), requires_grad=False).to(device)
        
    
        sr = generator(lr)

        
        # optimize G
        optimizer_G.zero_grad()
        
        loss_G = .001*adv_loss(discriminator(sr), valid) + \
            0.006*l1_loss(vggExtractor(hr).detach(), vggExtractor(sr))+\
                l1_loss(hr.detach(), sr)
        loss_G.backward()
        optimizer_G.step()


        # optimize D
        optimizer_D.zero_grad()
        loss_D = adv_loss(discriminator(sr.detach()), fake)+\
                adv_loss(discriminator(hr), valid)
            
        loss_D.backward()
        optimizer_D.step()

        Gloss_epoch += loss_G.item()
        Dloss_epoch += loss_D.item()
    
    if epoch % 10 == 0 and epoch != 0:
        filename = f'/content/drive/MyDrive/Colab Notebooks/github_repos/checkpoint/epoch{epoch}.pt'
        torch.save({'G_state_dict':generator.state_dict(), 'D_state_dict':discriminator.state_dict()}, filename)
        
        
        
    print(f'D_loss {Dloss_epoch} -- G_loss {Gloss_epoch}')