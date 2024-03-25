import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, n_classes):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.n_classes = n_classes

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        # simple conv based reduction of image dimension up to a flast MLP layer
        self.conv1 = nn.Conv2d(nc + n_classes, ndf, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.conv6 = nn.Conv2d(ndf * 16, 1, 8, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        # embed labels and concatenate to the image
        labels = self.label_embedding(labels)
        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.expand(labels.size(0), labels.size(1), x.size(2), x.size(3))
        x = torch.cat([x, labels], 1)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = self.conv6(x)
        x = self.sigmoid(x)

        return x.view(-1, 1).squeeze(1)