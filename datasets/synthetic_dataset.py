import os

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, root, attack=None, transform=None):

        """
        :param root:  root directory
        :param batch_size: number of images in batch file
        """

        self.root = root

        self.images = torch.load(os.path.join(root, "images.pt")).detach()
        self.labels = torch.load(os.path.join(root, "labels.pt")).detach()
        self.labels = self.labels.type(torch.LongTensor)

        assert torch.min(self.images) >= 0 and torch.max(self.images) <= 1
        assert torch.min(self.labels) >= 0 and torch.max(self.labels) <= 9

        self.size = self.labels.size(0)
        self.adv = None
        if os.path.exists(os.path.join(root, "adv_images.pt")):
            self.adv = torch.load(os.path.join(root, "adv_images.pt")).detach()
        else:
            if attack is not None:
                i = 0
                step = 128
                while i < self.size:
                    e = min(i + step, self.size)
                    adv = (attack(self.images[i:e], self.labels[i:e])).cpu()
                    i += step
                    if self.adv is None:
                        self.adv = adv
                    else:
                        self.adv = torch.cat((self.adv, adv), 0)

                torch.save(self.adv, os.path.join(root, "adv_images.pt"))

        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        if self.adv is not None and len(self.adv) > 0:
            adv = self.adv[idx]
            if self.transform is not None:
                adv = self.transform(adv)
        else:
            adv = None

        image = self.images[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.adv is not None and len(self.adv) > 0:
            return image, adv, self.labels[idx]
        else:
            return image[0, :, :].unsqueeze(0), self.labels[idx]


if __name__ == "__main__":
    path = "../data/teacher_alexnet_student_resnet18_cifar10_synthetic/"
    dataset = SyntheticDataset(path)
    print(dataset[0])
