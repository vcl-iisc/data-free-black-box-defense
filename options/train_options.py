import argparse

from constants import CIFAR10


def train_options():
    arg_parser = argparse.ArgumentParser(description='Experiment setup')

    arg_parser.add_argument('--dataset', type=str, default=CIFAR10, help='loss functions')
    arg_parser.add_argument('--attack', type=str, default='PGD', help='loss functions')

    arg_parser.add_argument('--loss', type=str, default='cosim_kl_wc', help='loss functions')
    arg_parser.add_argument('--wvlt', type=str, default='db1', help='type of wavelet function')
    arg_parser.add_argument('--mode', type=str, default='symmetric', help='wavelet mode')
    arg_parser.add_argument('--lr_policy', type=str, default='linear', help='scheduler policy')
    arg_parser.add_argument('--levels', type=int, default=2, help='number of levels')
    arg_parser.add_argument('--keep_percentage', type=int, default=15, help='percentage of coefficients to keep')
    arg_parser.add_argument('--batch_size', type=int, default=128, )
    arg_parser.add_argument('--lr', type=float, default=0.0002)
    arg_parser.add_argument('--beta1', type=float, default=0.5)

    arg_parser.add_argument('--epoch_count', type=int, default=1, )
    arg_parser.add_argument('--n_epochs', type=int, default=100, )
    arg_parser.add_argument('--n_epochs_decay', type=int, default=100, )
    arg_parser.add_argument('--lr_decay_iters', type=int, default=50, )
    arg_parser.add_argument('--gpu_id', type=str, default="0")
    arg_parser.add_argument("--continue_train", action='store_true')

    arg_parser.add_argument("--name", type=str, help="name by which model checkpoints are saved")

    arg_parser.add_argument("--surrogate_model_path", type=str,
                            default="model_stealing/checkpoints/student_resnet18_alexnet.pth")
    arg_parser.add_argument("--surrogate_model_name", type=str, default="resnet18")

    arg_parser.add_argument("--victim_model_name", type=str, default=None)

    arg_parser.add_argument("--synthetic_dataset_path", type=str, default="data/alexnet_resnet18_synthetic_data.pth")

    return arg_parser