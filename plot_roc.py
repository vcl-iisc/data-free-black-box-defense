import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import wandb
from adversarial_attacks import get_train_attack
from datasets.synthetic_dataset import SyntheticDataset
from metrics import Metric
from options import test_options
from utils import create_dbma_model
import tqdm

if __name__ == "__main__":

    arg_parser = test_options.test_options()
    arg_parser.add_argument("--synthetic_dataset_path", type=str, default="data/alexnet_resnet18_synthetic_data.pth")

    args = arg_parser.parse_args()

    wandb.init(project="data_free_black_box_defense_roc", name=args.name)

    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')

    # attack performed on surrogate models
    normalization = torchvision.transforms.Normalize(mean=(0.5), std=(0.5)).to(device)
    max_k = 30
    LCR = []
    for k in range(1, max_k+2):
        args.keep_percentage = k  # for different values of coefficient percentage

        dbma = create_dbma_model(args=args)  # convert to data parallel
        dbma.to(device)
        dbma.eval()

        attack = get_train_attack(args.dataset, args.attack, nn.Sequential(normalization, dbma.surrogate_model))
        synthetic_dataset = SyntheticDataset(args.synthetic_dataset_path, attack)
        train_dataloader = DataLoader(synthetic_dataset, batch_size=args.batch_size, shuffle=False)


        metric = Metric(train=False).to(device)

        for i, data in tqdm.tqdm(enumerate(train_dataloader)):
            clean_images, adv_images, labels = data
            with torch.no_grad():
                clean_images = normalization(clean_images).to(device)
                adv_images = normalization(adv_images).to(device)
                labels = labels.to(device)
                predictions = dbma(clean_images, adv_images, train=False)
                metric.update(predictions, labels)

        accuracy = metric.compute()
        LCR_A = accuracy["test_wv_adv_acc"]
        LCR_C = accuracy["test_wv_clean_acc"]
        LCR.append(LCR_C + LCR_A)
        

    ROC = {}
    for k in range(1, max_k+1):
        j = k-1
        ROC[k] = LCR[j + 1] - LCR[j]
        print(ROC[k].item())
        #wandb.log({"ROC": ROC[k]})
        #wandb.log({"LOC" : 1})
    
    wandb.finish()
