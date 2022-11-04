import argparse

import torch

import setup
import trainer
from normalization import Denormalize

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='Experiment setup')
    arg_parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
    arg_parser.add_argument('--optim', type=str, default='adam')
    arg_parser.add_argument('--generator', type=str, default='gan')
    arg_parser.add_argument('--size', type=int, default=32)
    arg_parser.add_argument('--student', type=str, default='half_lenet')
    arg_parser.add_argument('--teacher', type=str, default='lenet')
    arg_parser.add_argument('--true_dataset', type=str, default='split_fmnist')
    arg_parser.add_argument('--gpu_id', type=str, default=0)
    arg_parser.add_argument('--save_path', type=str, default=None)
    arg_parser.add_argument('--regen_name', type=str, default=None)
    arg_parser.add_argument('--epochs', type=str, default='75')
    arg_parser.add_argument('--samples', type=str, default='optimized')
    arg_parser.add_argument('--use_new_optimization', action='store_true')

    env = arg_parser.parse_args()
    device = torch.device("cuda:{}".format(env.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    setup.device = device

    teacher, teacher_dataset, student = setup.prepare_teacher_student(env)
    trainer.evaluate(teacher, teacher_dataset)
    generator = setup.prepare_generator(env)

    student_dataset = setup.prepare_student_dataset(
        env, teacher, teacher_dataset, student, generator
    )

    import os

    if not os.path.exists(env.save_path):
        os.mkdir(env.save_path)

    x = None
    y = None
    denorm = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    import torch

    for i in range(2):
        dataloader = student_dataset.train_dataloader(i)
        for iter_n, batch in enumerate(dataloader):
            if iter_n % 10 == 0:
                print(iter_n)
            images = batch[0].detach().cpu()
            targets = batch[1].detach().cpu()

            images = denorm(images)
            targets = torch.max(targets, dim=1)[1]

            if 0 > torch.min(images) or 1 < torch.max(images):
                print("break error")

            if x is None:
                x = images
                y = targets
            else:
                x = torch.cat((x, images), 0)
                y = torch.cat((y, targets), 0)

    torch.save(x, "{}/images.pt".format(env.save_path))
    torch.save(y, "{}/labels.pt".format(env.save_path))

import torch

x = torch.load("{}/images.pt".format(env.save_path))
y = torch.load("{}/labels.pt".format(env.save_path))

print(x.shape)
print(y.shape)
print(torch.max(x))
print(torch.min(y))
print(torch.max(y))
print(torch.min(y))
