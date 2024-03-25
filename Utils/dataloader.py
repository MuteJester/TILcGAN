from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class TILDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.1, 0.1, 0.1])
            ])

        super(TILDataset, self).__init__(root=root_dir, transform=transform)

