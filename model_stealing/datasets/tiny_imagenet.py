import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class TinyImagenet(object):
    def __init__(self, input_size=224):
        self.n_classes = 200
        # data_dir = '/media2/inder/dbma_shubham/PYTORCH-CIFAR/data/tiny-imagenet-200'
        data_dir = '/media2/inder/GenericRobustness/pytorch-cifar/data/tiny-imagenet-200'
        num_workers = {'train': 10, 'val': 0, 'test': 0}
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        self.dataloaders = {
            x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
            for x in ['train', 'val', 'test']}

        small_labels = {}
        with open(os.path.join(data_dir, "words.txt"), "r") as dictionary_file:
            line = dictionary_file.readline()
            while line:
                label_id, label = line.strip().split("\t")
                small_labels[label_id] = label
                line = dictionary_file.readline()

        labels = {}
        label_ids = {}
        for label_index, label_id in enumerate(self.dataloaders['train'].dataset.classes):
            label = small_labels[label_id]
            labels[label_index] = label
            label_ids[label_id] = label_index

        val_label_map = {}
        with open(os.path.join(data_dir, "val/val_annotations.txt"), "r") as val_label_file:
            line = val_label_file.readline()
            while line:
                file_name, label_id, _, _, _, _ = line.strip().split("\t")
                val_label_map[file_name] = label_id
                line = val_label_file.readline()

        for i in range(len(self.dataloaders['val'].dataset.imgs)):
            file_path = self.dataloaders['val'].dataset.imgs[i][0]

            file_name = os.path.basename(file_path)
            label_id = val_label_map[file_name]

            self.dataloaders['val'].dataset.imgs[i] = (file_path, label_ids[label_id])

    def train_dataloader(self, *args, **kwargs):
        return self.dataloaders["train"]

    def test_dataloader(self, *args, **kwrk):
        return self.dataloaders["val"]
