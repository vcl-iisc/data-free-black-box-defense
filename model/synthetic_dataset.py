import os.path
import torch
from torch.utils.data import Dataset, DataLoader



if __name__ == "__main__":
    d = SyntheticDataset("/media2/inder/GenericRobustness/dfme_dataset/")
    train_dataloader = DataLoader(d, batch_size=64, shuffle=True)
    for i, (images,adv_image, labels) in enumerate(train_dataloader):
        print(i)