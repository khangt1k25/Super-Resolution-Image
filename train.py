from models import Generator, Discriminator
from dataset import ImageDataset
import torch
import torch.nn as nn
from torch.autograd import Variable

voc2012_train = ImageDataset('./train/VOC-2012-train')
dataloader = torch.utils.data.DataLoader(voc2012_train, 
                                        batch_size=32, 
                                        shuffle=True, 
                                        num_workers=2)


generator = Generator(in_channels=3, n_residual_blocks=1, up_scale=4)
discriminator = Discriminator(in_channels=3)
optimizer_G = torch.optim.Adam(params=generator.parameters())
optimizer_D = torch.optim.Adam(params=discriminator.parameters())
adv_loss = nn.BCELoss()


epochs = 10

for epoch in range(epochs):
    Gloss_epoch = 0.0
    Dloss_epoch = 0.0
    for batch, data in enumerate(dataloader):
        lr = Variable(data['lr'])
        hr = Variable(data['hr'])
        batch_size = lr.shape[0]
        valid = Variable(torch.Tensor(batch_size, 1).fill_(1.0),
                        requires_grad=False)
        fake = Variable(torch.Tensor(batch_size, 1).fill_(0.0), 
                        requires_grad=False)
        
        sr = generator(lr)
        # optimize D
        optimizer_D.zero_grad()
        loss_D = adv_loss(discriminator(sr.detach()), fake)+\
                adv_loss(discriminator(hr), valid)
            
        loss_D.backward()
        optimizer_D.step()
        
        # optimize G
        optimizer_G.zero_grad()
        loss_G = adv_loss(discriminator(sr), valid)
        loss_G.backward()
        optimizer_G.step()
        
        
        Gloss_epoch += loss_G.item()
        Dloss_epoch += loss_D.item()
        


        
    print(Gloss_epoch)
    print(Dloss_epoch)
    



    