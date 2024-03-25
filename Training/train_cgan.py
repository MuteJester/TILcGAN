import torch
from torch import optim
from torch.utils.data import DataLoader
from Utils.dataloader import TILDataset
from torchvision import transforms
from Models.generator import Generator,GeneratorWithResiduals
from Models.discriminator import Discriminator
from Utils.dataloader import TILDataset
import torchvision.utils as vutils


def train_cgan(epochs, batch_size, learning_rate, nz, ngf, ndf, nc, n_classes, root_dir):
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the Generator and Discriminator
    generator = GeneratorWithResiduals(nz, ngf, nc, n_classes).to(device)
    discriminator = Discriminator(ndf, nc, n_classes).to(device)

    # loss function
    criterion = torch.nn.BCELoss()

    # optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # data loader
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # convert images from 224x224 for easier work with power 2 based logic
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = TILDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # Fixed noise for generating image grids
    loss_log = []

    # training loop
    for epoch in range(epochs):

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            b_size = images.size(0)

            # Create labels for real and fake images
            real_label = torch.empty(b_size, device=device).uniform_(0.7, 1.2)
            fake_label = torch.empty(b_size, device=device).uniform_(0.0, 0.3)

            # Train Discriminator
            discriminator.zero_grad()
            output = discriminator(images, labels).view(-1)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            # Generate fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = generator(noise, labels)
            output = discriminator(fake_images.detach(), labels).view(-1)
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images, labels).view(-1)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # log the same message we print to the screen, we will parse it in the analysis stage
            loss_log.append(f'Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(dataloader)}], Loss D: {errD.item()}, Loss G: {errG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1}/{D_G_z2}')

            if i % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(dataloader)}], Loss D: {errD.item()}, Loss G: {errG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1}/{D_G_z2}')

                with torch.no_grad():
                    fake_images = generator(fixed_noise, labels[:fixed_noise.size(0)]).detach().cpu()
                    # create a grid of images
                img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)

                # save the grid of images to a file
                vutils.save_image(img_grid, f'E:/TILcGAN/output/epoch_{epoch + 1}_step_{i}.png')
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                }, f'E:/TILcGAN/checkpoints/ckpt_epoch_{epoch}.pth')
        torch.save(loss_log, 'E:/TILcGAN/loss_log.pth')


    print('Training finished.')


if __name__ == '__main__':
    # our hyperparameters
    epochs = 500
    batch_size = 64
    learning_rate = 0.0002
    nz = 100  # size of z latent vector (generator input)
    ngf = 64  # size of feature maps in the generator
    ndf = 64  # size of feature maps in the discriminator
    nc = 3  # Number of channels in the training images.
    n_classes = 9
    root_dir = 'E:/TILcGAN/Data/train/'

    train_cgan(epochs, batch_size, learning_rate, nz, ngf, ndf, nc, n_classes, root_dir)
