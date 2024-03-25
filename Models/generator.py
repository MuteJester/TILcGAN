import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module): # first version of generator that achvied worse results
    def __init__(self, nz, ngf, nc, n_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.nz = nz
        self.n_classes = n_classes

        self.main = nn.Sequential(
            # input is concatenated Z and label embedding, going into a convolution
            nn.ConvTranspose2d(nz + n_classes, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size: (ngf * 16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf * 8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf * 4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf * 2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size: (nc) x 128 x 128
            nn.ConvTranspose2d(nc, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 256 x 256
        )

    def forward(self, noise, labels):
        # firist embed labels
        c = self.label_emb(labels)
        # make sure label embeddings match the dimensions of noise
        c = c.unsqueeze(-1).unsqueeze(-1).expand_as(noise[:, :self.n_classes, :, :])
        # concatenate noise and label embeddings along the channel dimension
        x = torch.cat([noise, c], dim=1)
        # pass though the main  generator block
        x = self.main(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class GeneratorWithResiduals(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes):
        super(GeneratorWithResiduals, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.nz = nz
        self.n_classes = n_classes

        self.init_size = ngf * 16
        self.l1 = nn.Sequential(nn.Linear(nz + n_classes, self.init_size * 4 * 4))

        # first layer after concatenation needs to handle nz + n_classes channels
        self.first_layer = nn.Sequential(
            nn.BatchNorm2d(nz + n_classes),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz + n_classes, self.init_size, 4, 1, 0, bias=False)
        )

        self.conv_blocks = nn.Sequential(
            ResidualBlock(self.init_size),
            nn.ConvTranspose2d(self.init_size, ngf * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ResidualBlock(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            ResidualBlock(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            ResidualBlock(ngf * 2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),  # upscale to 128x128
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            ResidualBlock(ngf),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, stride=2, padding=1),  # upscale to 256x256
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            ResidualBlock(ngf // 2),
            nn.ConvTranspose2d(ngf // 2, nc, 4, stride=2, padding=1),  # final layer to ensure output size is 256x256
            nn.Tanh()  # ensure output is in the range [-1, 1]
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        c = c.unsqueeze(-1).unsqueeze(-1).expand_as(noise[:, :self.n_classes, :, :])
        x = torch.cat([noise, c], dim=1)
        x = self.first_layer(x)
        x = self.conv_blocks(x)
        return x
