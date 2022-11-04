import torch
import torchvision
from torchvision import transforms


class CIFAR10(object):
    def __init__(self, input_size=32):
        self.n_classes = 10
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5), (0.5)
                # (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
            ),
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=transform
        )

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

    def test_dataloader(self, *args, **kwrk):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=16,
            num_workers=2,
            drop_last=False
        )
