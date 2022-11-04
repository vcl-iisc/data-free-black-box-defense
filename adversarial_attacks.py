import torchattacks

from constants import CIFAR10


def get_train_attack(dataset, attack, model):
    if dataset == CIFAR10:
        print('Attack using PGD on cifar10 dataset')
        eps, alpha, steps = 8 / 255, 2 / 255, 7
    elif dataset == "mnist":
        print('Attack using PGD on mnist dataset')
        eps, alpha, steps = 0.3, 0.01, 40
    elif dataset == "fmnist":
        print('Attack using PGD on fmnist dataset')
        # eps, alpha, steps = 0.3, 0.01, 100
        eps, alpha, steps = 0.2, 0.02, 40
    elif dataset == "svhn":
        print('Attack using PGD on svhn dataset')
        # eps, alpha, steps = 0.3, 0.01, 100
        eps, alpha, steps = 0.02, 0.02 / 10, 10
    else:
        print("dataset not defined ", dataset)
        raise Exception("dataset not defined")
    if attack == 'PGD':
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    else:
        print("attack not defined ", attack)
        raise Exception("attack not defined")

    return attack


def get_test_attack(dataset, attack, model):
    if attack == 'PGD':
        if dataset == CIFAR10:
            print('Attack using PGD on cifar10 dataset')
            eps, alpha, steps = 8 / 255, 2 / 255, 20
        elif dataset == "mnist":
            print('Attack using PGD on mnist dataset')
            eps, alpha, steps = 0.3, 0.01, 100
        elif dataset == "fmnist":
            print('Attack using PGD on fmnist dataset')
            # eps, alpha, steps = 0.3, 0.01, 100
            eps, alpha, steps = 0.2, 0.02, 100
        elif dataset == "svhn":
            print("Attack using PGD on svhn dataset")
            eps, alpha, steps = 4 / 255, 2 / 255, 20

        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)

    if attack == 'AA':
        if dataset == CIFAR10:
            print('Attack using AA on cifar10 dataset')
            eps = 8 / 255

        elif dataset == 'mnist':
            print('Attack using AA on mnist dataset')
            eps = 0.3

        elif dataset == 'fmnist':
            print('Attack using AA on fmnist dataset')
            eps = 0.2
        elif dataset == 'svhn':
            print('Attack using AA on svhn dataset')
            eps = 4 / 255

        attack = torchattacks.AutoAttack(model, eps=eps, n_classes=10, version='standard')

    if attack == 'BIM':
        if dataset == CIFAR10:
            print('Attack using BIM on cifar10 dataset')
            eps, steps = 8 / 255, 20
            alpha = eps / steps
        elif dataset == 'mnist':
            print('Attack using BIM on mnist dataset')
            eps, alpha, steps = 0.3, 0.03, 100
        elif dataset == 'fmnist':
            print('Attack using BIM on fmnist dataset')
            eps, alpha, steps = 0.2, 0.02, 100
        elif dataset == "svhn":
            print("Attack using BIM on svhn dataset")
            eps, alpha, steps = 4 / 255, 2 / 255, 20

        attack = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)

    return attack
