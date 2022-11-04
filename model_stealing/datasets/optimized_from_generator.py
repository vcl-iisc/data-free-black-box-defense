import sys

sys.path.append('..')

# from itertools import product
# from torch.multiprocessing import Pool, Process, set_start_method
from torch_optimizer import optimize, optimize_new_2, optimize_to_grayscale
import torch


# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


class OptimizedFromGenerator(object):
    def __init__(
            self,
            generator,
            teacher,
            student,
            test_dataloader,
            soft_labels=True,
            device=torch.device('cuda'),
            to_grayscale=True,
            batch_size=64,
            n_classes=10,
            input_size=32,
            use_new_optimization=False
    ):
        self.generator = generator
        self.teacher = teacher
        self.device = device
        self.test_dataloader = test_dataloader
        self.use_soft_labels = soft_labels
        self.to_grayscale = to_grayscale
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_size = input_size
        self.use_new_optimization = use_new_optimization

        # self.input_size = 224   #TODO temporary change
        # self.n_classes = 200  #TODO temporary change

    def train_dataloader(self, *args, **kwargs):
        optimization = optimize_to_grayscale if self.to_grayscale else optimize
        if self.use_new_optimization:
            optimization = optimize_new_2

        for _ in range(100):

            if 'combined' in str(type(self.generator)).lower():
                # with Pool(4) as p:
                #     first_images = torch.cat(
                #         p.starmap(
                #             optimize_to_grayscale,
                #             [(self.teacher, self.generator.gan, 8, 128)] * 4
                #         ), 
                #         axis = 0
                #     )
                # with Pool(4) as p:
                #     second_images = torch.cat(
                #         p.starmap(
                #             optimize_to_grayscale,
                #             [(self.teacher, self.generator.vae, 8, 100)] * 4
                #         ), 
                #         axis = 0
                #     )
                # images = torch.cat(
                #     [first_images, second_images],
                #     axis = 0
                # )
                images = torch.cat((
                    optimize_to_grayscale(
                        self.teacher, self.generator.gan, batch_size=32,
                        encoding_size=128
                    ),
                    optimize_to_grayscale(
                        self.teacher, self.generator.vae, batch_size=32,
                        encoding_size=100
                    ),
                ), axis=0)
            else:
                images = optimization(self.teacher, self.generator, image_size=self.input_size,
                                      num_classes=self.n_classes)

            with torch.no_grad():
                labels = self.teacher(images)
                if not self.use_soft_labels:
                    labels = labels.max(1)[1]
                else:
                    labels = torch.softmax(labels, dim=-1)

            yield (images, labels)
