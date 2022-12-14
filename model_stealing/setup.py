import datasets
import generators
import predictors
import trainer
import utils
from model_stealing.predictors.dbma_defene_model import DBMADefense

device = None
import torch

predictor_dict = {
    'half_lenet': predictors.HalfLeNet,
    'inceptionv3': predictors.InceptionV3,
    'lenet': predictors.LeNet,
    'resnet18': predictors.resnet.ResNet18,
    'vgg': predictors.VGG,
    'alexnet': predictors.Alexnet,
    'alexnet_half': predictors.AlexnetHalf,
    'resnet34': predictors.resnet.ResNet34,
    "dbma": utils.create_dbma_model
}
dataset_dict = {
    'cifar10': datasets.CIFAR10,
    # 'discrepancy': datasets.Discrepancy,
    # 'discrepancy_kl': datasets.Discrepancy_KL,
    # 'curriculum': datasets.Curriculum,
    'fmnist': datasets.FMNIST,
    'optimized': datasets.OptimizedFromGenerator,
    'random': datasets.RandomFromGenerator,
    'split_fmnist': datasets.SplitFMNIST,
    'mnist': datasets.MNIST,
    'svhn': datasets.SVHN,
    # 'two_gans': datasets.TwoGANs,
}
generator_dict = {
    'cifar_10_gan': generators.SNGAN,
    'cifar_100_90_classes_gan': generators.SNGAN,
    'cifar_100_40_classes_gan': generators.SNGAN,
    'cifar_10_vae': generators.VAE,
    'cifar_100_6_classes_gan': generators.Progan,
    'cifar_100_10_classes_gan': generators.Progan,
    'celeb_gan': generators.SNGAN
}
generator_prepare_dict = {
    'cifar_10_gan': trainer.train_or_restore_cifar_10_gan,
    'cifar_100_90_classes_gan': trainer.train_or_restore_cifar_100_90_classes_gan,
    'cifar_100_40_classes_gan': trainer.train_or_restore_cifar_100_40_classes_gan,
    'cifar_10_vae': trainer.train_or_restore_cifar_10_vae,
    'cifar_100_6_classes_gan': trainer.train_or_restore_cifar_100_6_classes_gan,
    'cifar_100_10_classes_gan': trainer.train_or_restore_cifar_100_10_classes_gan,
    'celeb_gan': trainer.celeb_gan
}


def prepare_teacher(env):
    true_dataset = dataset_dict[env.true_dataset](input_size=env.size)
    if env.teacher != "dbma":
        teacher = predictor_dict[env.teacher](
            name=teacher_name(env),
            n_outputs=true_dataset.n_classes,
            input_channels=env.input_nc,
            opt=env
        )
        teacher.to(device)
        if env.optim == 'sgd':
            trainer.train_or_restore_predictor(teacher, true_dataset)
        else:
            trainer.train_or_restore_predictor_adam(teacher, true_dataset)
    else:
        dbma = utils.create_dbma_model(env)
        dbma.load_state_dict(torch.load(env.dbma_path,map_location="cpu")["model_state_dict"])
        victim_model = utils.get_model(env.victim_model_name, env.victim_model_path)
        teacher = DBMADefense(dbma, victim_model)
        teacher.to(device)

    teacher.eval()
    return teacher


def prepare_teacher_student(env):
    true_dataset = dataset_dict[env.true_dataset](input_size=env.size)
    if env.teacher != "dbma":
        teacher = predictor_dict[env.teacher](
            name=teacher_name(env),
            n_outputs=true_dataset.n_classes,
            input_channels=env.input_nc,
        )
        teacher.to(device)
        if env.optim == 'sgd':
            trainer.train_or_restore_predictor(teacher, true_dataset)
        else:
            trainer.train_or_restore_predictor_adam(teacher, true_dataset)
    else:
        dbma = utils.create_dbma_model(env)
        dbma.load_state_dict(torch.load(env.dbma_path, map_location="cpu")["model_state_dict"])
        victim_model = utils.get_model(env.victim_model_name, env.victim_model_path)
        teacher = DBMADefense(dbma, victim_model)
        teacher.to(device)

    teacher.eval()

    student = predictor_dict[env.student](
        name=student_name(env),
        n_outputs=true_dataset.n_classes,
        input_channels=env.input_nc,
    )

    student.to(device)
    return teacher, true_dataset, student


def prepare_generator(env):
    if env.generator == 'combined':
        vae = generator_dict['dcgan']()
        vae = generator_prepare_dict['dcgan'](vae)

        gan = generator_dict['gan']()
        gan = generator_prepare_dict['gan'](gan)

        class CombinedGenerator():
            def __init__(self, vae, gan):
                self.vae = vae
                self.gan = gan

                self.current_generator = self.gan
                self.current_state = 'gan'

            def __call__(self, inputs):
                return self.current_generator(inputs)

            def switch(self):
                self.current_generator = (
                    self.gan if self.current_state == 'vae' else self.vae
                )
                self.current_state = (
                    'gan' if self.current_state == 'vae' else 'vae'
                )

            def encoding_size(self):
                return 128 if 'gan' in self.current_state else 100

        return CombinedGenerator(vae, gan)

    generator = generator_dict[env.generator]()
    generator = generator_prepare_dict[env.generator](generator)

    return generator


def prepare_student_dataset(env, teacher, teacher_dataset, student, generator):
    dataset = dataset_dict[env.samples](
        generator, teacher, student,
        test_dataloader=teacher_dataset.test_dataloader,
        to_grayscale=('gan' in env.generator and ('fmnist' in env.true_dataset or 'mnist' in env.true_dataset)),
        use_new_optimization=env.use_new_optimization
    )
    return dataset


def teacher_name(env):
    return f'teacher_{env.teacher}_for_{env.true_dataset}'


def student_name(env):
    extra_str = ""

    if env.teacher == "dbma":
        extra_str = env.dbma_name+"_"

    name = (f'student_{env.student}_teacher_{env.teacher}_{env.true_dataset}_' +
            f'{env.generator}_' + f'{extra_str}' +
            f'{env.optim}_{env.epochs}'
            )

    print("student name ", name)
    return name
