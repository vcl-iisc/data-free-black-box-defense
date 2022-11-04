import torch
import torchvision
from torch import nn

import wandb
from adversarial_attacks import get_test_attack
from datasets.utils import load_dataset
from metrics import Metric
from options import test_options
from utils import create_dbma_model

if __name__ == "__main__":

    arg_parser = test_options.test_options()
    args = arg_parser.parse_args()

    wandb.init(project="data_free_black_box_defense_test", name=args.name)

    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')

    dbma = create_dbma_model(args=args)  # convert to data parallel
    state_dict = torch.load("./checkpoints/{}/{}.pth".format(args.name, args.epoch))
    dbma.load_state_dict(state_dict["model_state_dict"])
    dbma.to(device)

    # attack performed on surrogate models
    normalization = torchvision.transforms.Normalize(mean=(0.5), std=(0.5)).to(device)
    attack = get_test_attack(args.dataset, args.attack, nn.Sequential(normalization, dbma.surrogate_model))

    test_dataset = load_dataset(args.dataset)
    test_loader = test_dataset.test_dataloader(batch_size=args.batch_size)

    print("number of test images: ", len(test_loader) * args.batch_size)

    metric = Metric(train=False).to(device)

    for i, data in enumerate(test_loader):
        clean_images, labels = data
        clean_images = clean_images.to(device)
        labels = labels.to(device)

        adv_images = attack(clean_images, labels)
        adv_images = adv_images.to(device)

        clean_images, adv_images = normalization(clean_images), normalization(adv_images)

        predictions = dbma(clean_images, adv_images, train=False)
        metric.update(predictions, labels)

    accuracy = metric.compute()

    wandb.log(accuracy)
    wandb.finish()
