from datasets.cifar10 import CIFAR10
from datasets.svhn import SVHN


def load_dataset(dataset, root="./data", ):
    if dataset == "cifar10":
        return CIFAR10()
    if dataset == "svhn":
        return SVHN()
