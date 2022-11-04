import os.path

import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

import wandb
from adversarial_attacks import get_train_attack
from datasets.synthetic_dataset import SyntheticDataset
from datasets.utils import load_dataset
from losses.losses import DBMA_Loss
from metrics import Metric
from options import train_options
from scheduler import get_scheduler
from utils import create_dbma_model

if __name__ == "__main__":

    arg_parser = train_options.train_options()
    args = arg_parser.parse_args()

    wandb.init(project="data_free_black_box_defense_train", name=args.name)
    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')

    # create model
    dbma = create_dbma_model(args=args).to(device)  # TODO convert to data parallel

    # define attack on surrogate model
    attack = get_train_attack(args.dataset, args.attack,
                              nn.Sequential(Normalize(mean=[0.5], std=[0.5]).to(device), dbma.surrogate_model))

    # create synthetic dataset. adversarial images are not already been created the attack is used to create
    synthetic_dataset = SyntheticDataset(args.synthetic_dataset_path, attack)
    train_dataloader = DataLoader(synthetic_dataset, batch_size=args.batch_size, shuffle=True)

    # test dataset
    test_dataset = load_dataset(args.dataset)
    test_loader = test_dataset.test_dataloader(args.batch_size)

    print("number of training images: ", len(train_dataloader) * args.batch_size)

    optimizer = torch.optim.Adam(dbma.parameters(), lr=args.lr, betas=(args.beta1, 0.999))  # define args.beta1
    scheduler = get_scheduler(optimizer, args.lr_policy, args.epoch_count, args.n_epochs, args.n_epochs_decay,
                              args.lr_decay_iters)
    criterion = DBMA_Loss(args.loss)

    normalization = torchvision.transforms.Normalize(mean=(0.5), std=(0.5)).to(device)

    for epoch in range(args.epoch_count, args.n_epochs + args.n_epochs_decay + 1):
        metric = Metric().to(device)
        loss_dict = {}
        for i, data in enumerate(train_dataloader):
            clean_images, adv_images, labels = data

            clean_images = normalization(clean_images).to(device)
            adv_images = normalization(adv_images).to(device)
            labels = labels.to(device)

            predictions = dbma(clean_images, adv_images)

            total_loss, loss_dict = criterion(predictions)

            total_loss.backward()
            optimizer.step()
            metric.update(predictions, labels)
            if i % 250 == 0:
                wandb.log(loss_dict)

        print("end of epoch : ", epoch)

        accuracy = metric.compute()
        wandb.log(accuracy)
        wandb.log(loss_dict)

        scheduler.step()

        if epoch % 5 == 0:
            dbma.eval()
            metric = Metric(train=False).to(device)

            for i, data in enumerate(test_loader):
                clean_images = data[0].to(device)
                labels = data[1].to(device)
                adv_images = attack(clean_images, labels)

                clean_images, adv_images = normalization(clean_images), normalization(adv_images)

                with torch.no_grad():
                    predictions = dbma(clean_images, adv_images)
                metric.update(predictions, labels)
                break
            accuracy = metric.compute()
            wandb.log(accuracy)
            dbma.train()
            state_dict = {
                'epoch': epoch,
                'model_state_dict': dbma.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss.cpu().detach()
            }
            path = "./checkpoints/{}".format(args.name)
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(state_dict, "{}/{}.pth".format(path, epoch))

    wandb.finish()
