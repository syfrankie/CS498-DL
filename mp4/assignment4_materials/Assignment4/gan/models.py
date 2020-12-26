import torch
import torch.nn as nn
from gan.spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        dis_channel = 128
        dis_kernel = 4

        self.dis1 = nn.Conv2d(input_channels, dis_channel, dis_kernel, stride=2, padding=1)
        self.dis2 = SpectralNorm(nn.Conv2d(dis_channel, dis_channel*2, dis_kernel, stride=2, padding=1))
        self.dis3 = SpectralNorm(nn.Conv2d(dis_channel*2, dis_channel*4, dis_kernel, stride=2, padding=1))
        self.dis4 = SpectralNorm(nn.Conv2d(dis_channel*4, dis_channel*8, dis_kernel, stride=2, padding=1))
        self.dis5 = nn.Sequential(
            nn.Conv2d(dis_channel*8, 1, dis_kernel, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
            )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
            
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.relu(self.dis1(x))
        x = self.relu(self.dis2(x))
        x = self.relu(self.dis3(x))
        x = self.relu(self.dis4(x))
        x = self.dis5(x)
        x = x.view(x.size()[0], -1)
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        gen_channel = 64
        gen_kernel = 4

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, gen_channel*16, gen_kernel, padding=1),
            nn.BatchNorm2d(gen_channel*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(gen_channel*16, gen_channel*8, gen_kernel, stride=2, padding=1),
            nn.BatchNorm2d(gen_channel*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(gen_channel*8, gen_channel*4, gen_kernel, stride=2, padding=1),
            nn.BatchNorm2d(gen_channel*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(gen_channel*4, gen_channel*2, gen_kernel, stride=2, padding=1),
            nn.BatchNorm2d(gen_channel*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(gen_channel*2, gen_channel, gen_kernel, stride=2, padding=1),
            nn.BatchNorm2d(gen_channel),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(gen_channel, output_channels, gen_kernel, stride=2, padding=1),
            nn.Tanh()
            )
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.gen(x)
        ##########       END      ##########

        return x
    

