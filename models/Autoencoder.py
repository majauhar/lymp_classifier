# -- Encoder - Decoder
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(3, 64, 64)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.channel_mult*1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*16, self.channel_mult*16, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)

        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, int(self.flat_fts/2)),
            nn.Linear(int(self.flat_fts/2), output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, 3, 64, 64)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Encoder_Resnet(nn.Module):
    def __init__(self, output_size, input_size=(3, 64, 64)):
        super(CNN_Encoder_Resnet, self).__init__()
        self.resnet = models.resnet18()
        self.flat_fts = 512
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.flat_fts, int(self.flat_fts/2)),
            nn.Linear(int(self.flat_fts/2), output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, 3, 64, 64)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.resnet(x)
        return x

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size=256, input_size=(3, 64, 64)):
        super(CNN_Decoder, self).__init__()
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = input_size[0]
        self.fc_output_dim = 4096

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*8,
                                4, 1, 0, bias=False),  # Adjusted kernel, stride, padding
            nn.BatchNorm2d(self.channel_mult*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*4,
                                4, 2, 1, bias=False),  # Adjusted kernel, stride, padding
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                4, 2, 1, bias=False),  # Adjusted kernel, stride, padding
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult,
                                4, 2, 1, bias=False),  # Adjusted kernel, stride, padding
            nn.BatchNorm2d(self.channel_mult),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult, self.output_channels,
                                4, 2, 1, bias=False),  # Adjusted kernel, stride, padding
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x
    
class Network(nn.Module):
    def __init__(self, embedding_size=256):
        super(Network, self).__init__()
        self.encoder = CNN_Encoder_Resnet(embedding_size) # , input_size=(3, 224, 224))
        self.decoder = CNN_Decoder(embedding_size) #, input_size=(3, 224, 224))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)