if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse

import torch
import wandb

import setup
import trainer

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='Experiment setup')
    # dbma model Parameters
    arg_parser.add_argument('--mode', type=str, default='symmetric', help='wavelet mode')
    arg_parser.add_argument('--wvlt', type=str, default="db1", help="type of wavelet function")
    arg_parser.add_argument('--levels', type=int, default=2)
    arg_parser.add_argument('--input_nc', type=int, default=3)
    arg_parser.add_argument('--keep_percentage', type=int, default=15)
    arg_parser.add_argument('--regen_path', type=str, default=None)
    arg_parser.add_argument('--regen_name', type=str, default="", help="name of regenerator network")

    arg_parser.add_argument('--retrain', action='store_true')
    arg_parser.add_argument("--name", type=str, help="name by which model checkpoints are saved")
    arg_parser.add_argument('--victim_model_name', type=str, default=None)
    arg_parser.add_argument('--victim_model_path', type=str, default=None)

    arg_parser.add_argument('--device', type=str, default="cuda")
    arg_parser.add_argument('--epochs', type=str, default='75')
    arg_parser.add_argument('--generator', type=str, default='gan')
    arg_parser.add_argument('--optim', type=str, default='adam')
    arg_parser.add_argument('--proxy_dataset', type=str, default='cifar10')
    arg_parser.add_argument('--sample_optimization', type=str, default='class')
    arg_parser.add_argument('--samples', type=str, default='optimized')
    arg_parser.add_argument('--size', type=int, default=32)
    arg_parser.add_argument('--student', type=str, default='half_lenet')
    arg_parser.add_argument('--teacher', type=str, default='lenet')
    arg_parser.add_argument('--true_dataset', type=str, default='split_fmnist')
    arg_parser.add_argument('--gpu_id', type=str, default=0)

    arg_parser.add_argument('--use_new_optimization', action='store_true')
    arg_parser.add_argument('--use_wandb', action='store_true')

    env = arg_parser.parse_args()
    if env.use_wandb:
        wandb.init(name="{}_{}_{}".format(env.teacher, env.student, env.epochs), project="black_box")

    device = torch.device("cuda:{}".format(env.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    setup.device = device

    teacher, teacher_dataset, student = setup.prepare_teacher_student(env)

    trainer.evaluate(teacher, teacher_dataset)

    generator = setup.prepare_generator(env)

    student_dataset = setup.prepare_student_dataset(
        env, teacher, teacher_dataset, student, generator
    )

    if env.optim == 'sgd':
        trainer.train_or_restore_predictor(
            student, student_dataset, loss_type='binary',
            n_epochs=int(env.epochs)
        )
    else:
        trainer.train_or_restore_predictor_adam(
            student, student_dataset, loss_type='binary',
            n_epochs=int(env.epochs)
        )
    print("evaluating student")
    trainer.evaluate(student, teacher_dataset)
    if env.use_wandb:
        wandb.finish()
