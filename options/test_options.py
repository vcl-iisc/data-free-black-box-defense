import argparse

from constants import CIFAR10


def test_options():
    arg_parser = argparse.ArgumentParser(description='Experiment setup')

    arg_parser.add_argument('--dataset', type=str, default=CIFAR10, help='loss functions')
    arg_parser.add_argument('--attack', type=str, default='PGD', help='loss functions')

    arg_parser.add_argument('--wvlt', type=str, default='db1', help='type of wavelet function')
    arg_parser.add_argument('--mode', type=str, default='symmetric', help='wavelet mode')

    arg_parser.add_argument('--levels', type=int, default=2, help='number of levels')
    arg_parser.add_argument('--keep_percentage', type=int, default=15, help='percentage of coefficients to keep')
    arg_parser.add_argument('--batch_size', type=int, default=128, )
    arg_parser.add_argument('--epoch', type=int, default=300, )

    arg_parser.add_argument('--gpu_id', type=str, default="0")

    arg_parser.add_argument("--name", type=str, help="name by which model checkpoints are saved")

    # path to black box victim model path and its architecture name
    arg_parser.add_argument("--victim_model_path", type=str,
                            default="model_stealing/checkpoints/student_resnet18_alexnet.pth")
    arg_parser.add_argument("--victim_model_name", type=str, default="resnet18")

    arg_parser.add_argument("--surrogate_model_path", type=str,
                            default="model_stealing/checkpoints/student_resnet18_alexnet.pth")
    arg_parser.add_argument("--surrogate_model_name", type=str, default="resnet18")

    args = arg_parser.parse_args()
    return args
